from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds, transform as window_transform

from .models import SentinelMosaic


def _radius_km_to_latlon_bbox(center_latlon: Tuple[float, float], radius_km: float):
    lat, lon = float(center_latlon[0]), float(center_latlon[1])
    dlat = radius_km / 111.32
    dlon = radius_km / max(111.32 * math.cos(math.radians(lat)), 1e-6)
    min_lat = lat - dlat
    max_lat = lat + dlat
    min_lon = lon - dlon
    max_lon = lon + dlon
    return (min_lon, min_lat, max_lon, max_lat)


def _bbox_4326_to_dataset_crs(bbox_lonlat, ds_crs):
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    if str(ds_crs).upper().endswith("4326"):
        return (min_lon, min_lat, max_lon, max_lat)
    transformer = Transformer.from_crs("EPSG:4326", ds_crs, always_xy=True)
    xs, ys = transformer.transform(
        [min_lon, max_lon, max_lon, min_lon],
        [min_lat, min_lat, max_lat, max_lat],
    )
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_dataset_to_4326(bbox_xy, ds_crs):
    min_x, min_y, max_x, max_y = bbox_xy
    if str(ds_crs).upper().endswith("4326"):
        return (min_x, min_y, max_x, max_y)
    transformer = Transformer.from_crs(ds_crs, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(
        [min_x, max_x, max_x, min_x],
        [min_y, min_y, max_y, max_y],
    )
    return (min(lons), min(lats), max(lons), max(lats))


@dataclass
class LocalRasterProvider:
    mosaic_path: str
    band_indices: Sequence[int] = (1, 2, 3, 4)  # rasterio is 1-based
    acquisition_datetime: Optional[str] = None

    def fetch_mosaic(
        self,
        center_latlon: Tuple[float, float],
        radius_km: float,
        acquisition_datetime: Optional[str],
        time_window_days: int,
        reference_product: str = "l2a",
    ) -> SentinelMosaic:
        with rasterio.open(self.mosaic_path) as ds:
            if ds.count < max(self.band_indices):
                raise ValueError(
                    f"Raster has {ds.count} bands but band_indices={self.band_indices}"
                )

            bbox_4326 = _radius_km_to_latlon_bbox(center_latlon, radius_km)
            bbox_ds = _bbox_4326_to_dataset_crs(bbox_4326, ds.crs)
            win = from_bounds(*bbox_ds, transform=ds.transform)
            arr = ds.read(
                indexes=list(self.band_indices),
                window=win,
                boundless=True,
                fill_value=np.nan,
                out_dtype="float32",
            )
            arr = np.moveaxis(arr, 0, -1)  # HWC

            tfm = window_transform(win, ds.transform)
            bbox_crop_ds = window_bounds(win, ds.transform)
            bbox_crop_4326 = _bbox_dataset_to_4326(bbox_crop_ds, ds.crs)

            return SentinelMosaic(
                image=arr.astype(np.float32),
                transform=tfm,
                crs=ds.crs,
                bbox_latlon=bbox_crop_4326,
                acquisition_datetime=acquisition_datetime or self.acquisition_datetime,
                source_path=self.mosaic_path,
            )
