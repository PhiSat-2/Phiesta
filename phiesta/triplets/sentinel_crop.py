from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from ..remote.catalog_geometry import get_catalog_corners
from .models import SentinelSource, SentinelCropResult
from .sentinel_download import resolve_sentinel_l1c_safe_paths


S2_BANDS_SIM = ["B02", "B03", "B04", "B08", "B05", "B06", "B07"]
S2_BANDS_NAMES = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR_BROAD",
    "RED_EDGE_1",
    "RED_EDGE_2",
    "RED_EDGE_3",
]

XML_BAND_ID_TO_NAME = {
    1: "B02",
    2: "B03",
    3: "B04",
    4: "B05",
    5: "B06",
    6: "B07",
    7: "B08",
}


def _earth_sun_distance_au(date_obj: datetime) -> float:
    day_of_year = date_obj.timetuple().tm_yday
    return float(1.0 - 0.01672 * math.cos(math.radians(0.9856 * (day_of_year - 4))))


def _buffer_lonlat_bounds_from_event(event: Any, buffer_km: float) -> tuple[float, float, float, float]:
    corners = get_catalog_corners(event, order="lonlat")
    if not corners:
        raise ValueError(
            "No catalog corners available on event. "
            "Load the product from Insula or enrich it with catalog_geo first."
        )

    lons = [p[0] for p in corners]
    lats = [p[1] for p in corners]

    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    mean_lat = (min_lat + max_lat) / 2.0
    buffer_deg_lat = buffer_km / 111.32
    buffer_deg_lon = buffer_km / (111.32 * np.cos(np.radians(mean_lat)))

    return (
        min_lon - buffer_deg_lon,
        min_lat - buffer_deg_lat,
        max_lon + buffer_deg_lon,
        max_lat + buffer_deg_lat,
    )


def _find_first_granule_dir(safe_path: str | Path, prefix: str) -> Path:
    granule_dir = Path(safe_path) / "GRANULE"
    if not granule_dir.exists():
        raise FileNotFoundError(f"Missing GRANULE directory in SAFE: {safe_path}")

    candidates = [p for p in granule_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No granule starting with {prefix!r} found in {granule_dir}")

    return candidates[0]


def _find_l1c_band_paths(l1c_paths: list[str]) -> dict[str, list[str]]:
    band_map = {band: [] for band in S2_BANDS_SIM}

    for safe_path in l1c_paths:
        safe = Path(safe_path)
        if not safe.exists():
            raise FileNotFoundError(
                f"Sentinel L1C SAFE path is not accessible locally: {safe}\n"
                "This step must be run on a machine where the SAFE products are mounted."
            )

        granule = _find_first_granule_dir(safe, prefix="L1C_T")
        img_data = granule / "IMG_DATA"
        if not img_data.exists():
            raise FileNotFoundError(f"Missing IMG_DATA directory: {img_data}")

        for band in S2_BANDS_SIM:
            matches = list(img_data.glob(f"*_{band}.jp2"))
            if matches:
                band_map[band].append(str(matches[0]))

    missing = [band for band, paths in band_map.items() if not paths]
    if missing:
        raise FileNotFoundError(f"Missing Sentinel-2 L1C bands: {missing}")

    return band_map


def _extract_simulator_metadata(l1c_path: str | Path, target_datetime: str | None) -> dict:
    """
    Extract the metadata required by the PhiSat-2 simulation pipeline.

    Expected keys:
    - earth_sun_dist
    - solar_irradiances
    - sun_zenith_angles
    """
    meta = {
        "earth_sun_dist": 1.0,
        "solar_irradiances": {},
        "sun_zenith_angles": None,
    }

    if target_datetime is not None:
        try:
            dt = datetime.fromisoformat(str(target_datetime).replace("Z", "+00:00")).replace(tzinfo=None)
            meta["earth_sun_dist"] = _earth_sun_distance_au(dt)
        except Exception:
            pass

    safe_dir = Path(l1c_path)

    xml_main = safe_dir / "MTD_MSIL1C.xml"
    if xml_main.exists():
        try:
            root = ET.parse(xml_main).getroot()
            irradiance_list = root.find(".//Solar_Irradiance_List")
            if irradiance_list is not None:
                for elem in irradiance_list.findall("SOLAR_IRRADIANCE"):
                    band_id = int(elem.get("bandId", -1))
                    if band_id in XML_BAND_ID_TO_NAME:
                        meta["solar_irradiances"][XML_BAND_ID_TO_NAME[band_id]] = float(elem.text)
        except Exception:
            pass

    try:
        granule = _find_first_granule_dir(safe_dir, prefix="L1C_T")
        xml_tile = granule / "MTD_TL.xml"
        if xml_tile.exists():
            root = ET.parse(xml_tile).getroot()
            zenith_grid = root.find(".//Sun_Angles_Grid/Zenith/Values_List")
            if zenith_grid is not None:
                rows = []
                for val_elem in zenith_grid.findall("VALUES"):
                    if val_elem.text:
                        rows.append([float(v) for v in val_elem.text.strip().split()])
                if rows:
                    meta["sun_zenith_angles"] = rows
    except Exception:
        pass

    return meta


def _crop_one_raster_to_lonlat_bbox(src, bbox_4326):
    transformer = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
    native_bbox = shapely_transform(transformer, bbox_4326)
    out_image, out_transform = mask(src, [native_bbox], crop=True)
    return out_image, out_transform


def _merge_band_paths_to_master_grid(
    band_paths: list[str],
    bbox_4326,
    master_crs=None,
    master_transform=None,
    master_shape=None,
):
    datasets = []
    memfiles = []

    try:
        for fp in band_paths:
            with rasterio.open(fp) as src:
                out_image, out_transform = _crop_one_raster_to_lonlat_bbox(src, bbox_4326)

                mem = MemoryFile()
                memfiles.append(mem)

                with mem.open(
                    driver="GTiff",
                    height=out_image.shape[1],
                    width=out_image.shape[2],
                    count=1,
                    dtype=out_image.dtype,
                    crs=src.crs,
                    transform=out_transform,
                ) as dst:
                    dst.write(out_image[0], 1)

                datasets.append(mem.open())

        if not datasets:
            raise ValueError("No raster data could be cropped for this band.")

        with ExitStack() as stack:
            if master_crs is None:
                master_crs = datasets[0].crs

            vrts = [
                stack.enter_context(WarpedVRT(ds, crs=master_crs))
                for ds in datasets
            ]

            merged, merged_transform = merge(vrts, method="first")

        merged_arr = merged[0]

        if master_transform is None:
            master_transform = merged_transform
            master_shape = merged_arr.shape
            return merged_arr, master_crs, master_transform, master_shape

        with MemoryFile() as mem:
            with mem.open(
                driver="GTiff",
                height=merged_arr.shape[0],
                width=merged_arr.shape[1],
                count=1,
                dtype=merged_arr.dtype,
                crs=master_crs,
                transform=merged_transform,
            ) as tmp:
                tmp.write(merged_arr, 1)

            with mem.open() as tmp:
                with WarpedVRT(
                    tmp,
                    crs=master_crs,
                    transform=master_transform,
                    width=master_shape[1],
                    height=master_shape[0],
                    resampling=Resampling.bilinear,
                ) as vrt:
                    aligned = vrt.read(1)

        return aligned, master_crs, master_transform, master_shape

    finally:
        for ds in datasets:
            ds.close()
        for mem in memfiles:
            mem.close()


def create_sentinel_crop(
    event: Any,
    source: SentinelSource,
    output_dir: str | Path,
    buffer_km: float = 10.0,
    overwrite: bool = False,
    verbose: bool = True,
    sentinel_backend: str = "auto",
    sentinel_cache_dir: str | Path = "cache/sentinel2",
    cdse_username: str | None = None,
    cdse_password: str | None = None,
    cdse_access_token: str | None = None,
    overwrite_sentinel_download: bool = False,
) -> SentinelCropResult:
    """
    Create a local Sentinel-2B crop and simulation metadata for one PhiSat-2 event.

    The output GeoTIFF contains 7 bands in this order:
    B02, B03, B04, B08, B05, B06, B07.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    product_id = str(source.product_id)

    crop_path = output_dir / f"{product_id}_s2b_crop_7bands.tif"
    metadata_path = output_dir / f"{product_id}_s2b_metadata.json"

    if crop_path.exists() and metadata_path.exists() and not overwrite:
        return SentinelCropResult(
            crop_path=str(crop_path),
            metadata_path=str(metadata_path),
            cloud_mask_path=None,
            buffer_km=float(buffer_km),
            resolution_m=10.0,
            bands=list(S2_BANDS_SIM),
            metadata={
                "status": "ALREADY_EXISTS",
                "source": source.to_dict() if hasattr(source, "to_dict") else {},
            },
        )
 
    if verbose:
        print(f"[Phiesta] Creating Sentinel-2B crop: {crop_path}")

    min_lon, min_lat, max_lon, max_lat = _buffer_lonlat_bounds_from_event(
        event,
        buffer_km=buffer_km,
    )
    bbox_4326 = box(min_lon, min_lat, max_lon, max_lat)

    local_l1c_paths = resolve_sentinel_l1c_safe_paths(
        source,
        backend=sentinel_backend,
        cache_dir=sentinel_cache_dir,
        cdse_username=cdse_username,
        cdse_password=cdse_password,
        cdse_access_token=cdse_access_token,
        overwrite=overwrite_sentinel_download,
        verbose=verbose,
    )

    band_map = _find_l1c_band_paths(local_l1c_paths)

    final_channels = []
    master_crs = None
    master_transform = None
    master_shape = None

    with rasterio.Env(GDAL_NUM_THREADS="1", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for band in S2_BANDS_SIM:
            if verbose:
                print(f"[Phiesta] Cropping S2 band {band}")

            arr, master_crs, master_transform, master_shape = _merge_band_paths_to_master_grid(
                band_paths=band_map[band],
                bbox_4326=bbox_4326,
                master_crs=master_crs,
                master_transform=master_transform,
                master_shape=master_shape,
            )
            final_channels.append(arr)

    stack = np.stack(final_channels, axis=0).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": stack.shape[0],
        "height": stack.shape[1],
        "width": stack.shape[2],
        "crs": master_crs,
        "transform": master_transform,
        "compress": "deflate",
        "bigtiff": "if_safer",
    }

    with rasterio.open(crop_path, "w", **profile) as dst:
        dst.write(stack)
        dst.descriptions = tuple(S2_BANDS_NAMES)

    metadata = _extract_simulator_metadata(
        l1c_path=local_l1c_paths[0],
        target_datetime=source.s2_datetime,
    )
    metadata.update(
        {
            "product_id": product_id,
            "satellite": source.satellite,
            "s2_datetime": source.s2_datetime,
            "delta_days": source.delta_days,
            "cloud_cover": source.cloud_cover,
            "coverage": source.coverage,
            "buffer_km": float(buffer_km),
            "bands": list(S2_BANDS_SIM),
            "band_names": list(S2_BANDS_NAMES),
            "crop_path": str(crop_path),
            "l1c_paths": list(source.l1c_paths),
            "l1c_local_paths": list(local_l1c_paths),
            "l2a_paths": list(source.l2a_paths),
            "sentinel_backend": str(sentinel_backend),
            "sentinel_cache_dir": str(sentinel_cache_dir),
        }
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"[Phiesta] Sentinel crop saved: {crop_path}")
        print(f"[Phiesta] Metadata saved: {metadata_path}")

    return SentinelCropResult(
        crop_path=str(crop_path),
        metadata_path=str(metadata_path),
        cloud_mask_path=None,
        buffer_km=float(buffer_km),
        resolution_m=10.0,
        bands=list(S2_BANDS_SIM),
        metadata={
            "status": "SUCCESS",
            "shape": tuple(stack.shape),
            "crs": str(master_crs),
            "bounds_lonlat": [min_lon, min_lat, max_lon, max_lat],
        },
    )