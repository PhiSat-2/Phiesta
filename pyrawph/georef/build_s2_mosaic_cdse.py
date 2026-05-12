from __future__ import annotations

import argparse
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple, Optional

import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import from_bounds
from pyproj import Transformer


TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


def parse_date_utc(date_str: str) -> datetime:
    """
    Accepts:
      - YYYY-MM-DD
      - YYYY-MM-DDTHH:MM:SS
      - YYYY-MM-DDTHH:MM:SSZ
    Returns timezone-aware UTC datetime.
    """
    s = date_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(date_str, "%Y-%m-%d")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def isoformat_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_access_token(client_id: str, client_secret: str) -> str:
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["access_token"]


def utm_epsg_from_latlon(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def make_search_bbox_utm(
    lat: float,
    lon: float,
    radius_km: float,
) -> Tuple[Tuple[float, float, float, float], int]:
    """
    Returns:
        bbox_utm = (minx, miny, maxx, maxy)
        utm_epsg
    """
    epsg = utm_epsg_from_latlon(lat, lon)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = to_utm.transform(lon, lat)

    r = radius_km * 1000.0
    bbox = (x - r, y - r, x + r, y + r)
    return bbox, epsg

def make_bbox_utm_from_bbox_latlon(
    bbox_latlon: Tuple[float, float, float, float],
) -> Tuple[Tuple[float, float, float, float], int]:
    """
    Convert a lon/lat bbox (EPSG:4326) into a UTM bbox.

    Input:
        bbox_latlon = (min_lon, min_lat, max_lon, max_lat)

    Returns:
        bbox_utm = (minx, miny, maxx, maxy)
        utm_epsg
    """
    min_lon, min_lat, max_lon, max_lat = bbox_latlon
    center_lat = 0.5 * (min_lat + max_lat)
    center_lon = 0.5 * (min_lon + max_lon)

    epsg = utm_epsg_from_latlon(center_lat, center_lon)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    xs, ys = to_utm.transform(
        [min_lon, max_lon, max_lon, min_lon],
        [min_lat, min_lat, max_lat, max_lat],
    )

    bbox_utm = (min(xs), min(ys), max(xs), max(ys))
    return bbox_utm, epsg

def split_bbox_utm(
    bbox_utm: Tuple[float, float, float, float],
    tile_size_m: float = 25000.0,
) -> List[Tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = bbox_utm

    tiles = []
    x = minx
    while x < maxx:
        x2 = min(x + tile_size_m, maxx)
        y = miny
        while y < maxy:
            y2 = min(y + tile_size_m, maxy)
            tiles.append((x, y, x2, y2))
            y = y2
        x = x2
    return tiles


def bbox_size_from_resolution(
    bbox: Tuple[float, float, float, float],
    resolution_m: float = 10.0,
) -> Tuple[int, int]:
    minx, miny, maxx, maxy = bbox
    width = max(1, int(round((maxx - minx) / resolution_m)))
    height = max(1, int(round((maxy - miny) / resolution_m)))
    return width, height


def build_evalscript() -> str:
    # Output band order: BLUE, GREEN, RED, NIR
    return """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08", "dataMask"]
    }],
    output: {
      bands: 4,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  if (sample.dataMask == 0) {
    return [0, 0, 0, 0];
  }
  return [sample.B02, sample.B03, sample.B04, sample.B08];
}
""".strip()


def make_process_payload(
    bbox_utm: Tuple[float, float, float, float],
    epsg: int,
    date_from: str,
    date_to: str,
    width: int,
    height: int,
    max_cloud_coverage: int,
    data_collection: str = "sentinel-2-l1c",
    mosaicking_order: str = "leastCC",
) -> dict:
    return {
        "input": {
            "bounds": {
                "bbox": list(bbox_utm),
                "properties": {
                    "crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"
                },
            },
            "data": [
                {
                    "type": str(data_collection),
                    "dataFilter": {
                        "timeRange": {
                            "from": date_from,
                            "to": date_to,
                        },
                        "mosaickingOrder": str(mosaicking_order),
                        "maxCloudCoverage": int(max_cloud_coverage),
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    },
                }
            ],
        },
        "evalscript": build_evalscript(),
    }


def download_tile_tiff(
    token: str,
    payload: dict,
) -> bytes:
    resp = requests.post(
        PROCESS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "image/tiff",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Process API failed with status {resp.status_code}: {resp.text[:1000]}"
        )
    return resp.content


def save_tile_geotiff(
    tiff_bytes: bytes,
    bbox_utm: Tuple[float, float, float, float],
    epsg: int,
    out_path: Path,
) -> None:
    with MemoryFile(tiff_bytes) as memfile:
        with memfile.open() as src:
            arr = src.read()  # (C, H, W)

    transform = from_bounds(*bbox_utm, width=arr.shape[2], height=arr.shape[1])

    profile = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": arr.shape[0],
        "dtype": str(arr.dtype),
        "crs": f"EPSG:{epsg}",
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "nodata": 0.0,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)


def merge_tiles(tile_paths: List[Path], out_path: Path) -> None:
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, out_transform = merge(srcs)

        profile = srcs[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=mosaic.shape[0],
            transform=out_transform,
            compress="lzw",
            tiled=True,
            nodata=0.0,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for s in srcs:
            s.close()


def build_s2_mosaic(
    phi_date: str,
    out_tif: str,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    bbox_latlon: Optional[Tuple[float, float, float, float]] = None,
    tile_size_km: float = 25.0,
    resolution_m: float = 10.0,
    time_window_days: int = 45,
    max_cloud_coverage: int = 60,
    data_collection: str = "sentinel-2-l1c",
    mosaicking_order: str = "leastCC",
) -> Path:
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing CDSE credentials. Set environment variables "
            "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET first."
        )

    token = get_access_token(client_id, client_secret)

    center_dt = parse_date_utc(phi_date)
    date_from = isoformat_z(center_dt - timedelta(days=time_window_days))
    date_to = isoformat_z(center_dt + timedelta(days=time_window_days))

    if bbox_latlon is not None:
        bbox_utm, epsg = make_bbox_utm_from_bbox_latlon(bbox_latlon)
    else:
        if center_lat is None or center_lon is None or radius_km is None:
            raise ValueError(
                "Provide either bbox_latlon, or center_lat + center_lon + radius_km."
            )
        bbox_utm, epsg = make_search_bbox_utm(
            center_lat,
            center_lon,
            radius_km=radius_km,
        )

    tiles = split_bbox_utm(bbox_utm, tile_size_m=tile_size_km * 1000.0)

    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    tile_paths: List[Path] = []

    with tempfile.TemporaryDirectory(prefix="s2_tiles_") as tmpdir:
        tmpdir = Path(tmpdir)

        for i, tile_bbox in enumerate(tiles):
            width, height = bbox_size_from_resolution(tile_bbox, resolution_m=resolution_m)

            payload = make_process_payload(
                bbox_utm=tile_bbox,
                epsg=epsg,
                date_from=date_from,
                date_to=date_to,
                width=width,
                height=height,
                max_cloud_coverage=max_cloud_coverage,
                data_collection=data_collection,
                mosaicking_order=mosaicking_order,
            )

            print(f"[{i+1}/{len(tiles)}] downloading tile {tile_bbox} ({width}x{height})")
            tiff_bytes = download_tile_tiff(token, payload)

            tile_path = tmpdir / f"tile_{i:02d}.tif"
            save_tile_geotiff(
                tiff_bytes=tiff_bytes,
                bbox_utm=tile_bbox,
                epsg=epsg,
                out_path=tile_path,
            )
            tile_paths.append(tile_path)

        print(f"Merging {len(tile_paths)} tiles into {out_tif} ...")
        merge_tiles(tile_paths, out_tif)

    print(f"Done: {out_tif}")
    return out_tif


def main():
    parser = argparse.ArgumentParser(description="Build a local Sentinel-2 L2A 4-band mosaic for PhiSat-2 matching.")
    parser.add_argument("--lat", type=float, required=True, help="Approximate center latitude")
    parser.add_argument("--lon", type=float, required=True, help="Approximate center longitude")
    parser.add_argument("--date", type=str, required=True, help="PhiSat-2 date, e.g. 2026-01-26 or 2026-01-26T17:40:10Z")
    parser.add_argument("--out", type=str, required=True, help="Output GeoTIFF path")
    parser.add_argument("--radius-km", type=float, default=25.0, help="Search radius in km")
    parser.add_argument("--tile-size-km", type=float, default=25.0, help="Tile size in km")
    parser.add_argument("--resolution-m", type=float, default=10.0, help="Output resolution in meters")
    parser.add_argument("--time-window-days", type=int, default=45, help="Temporal search window around PhiSat-2 date")
    parser.add_argument("--max-cloud", type=int, default=60, help="Maximum cloud coverage percentage")
    args = parser.parse_args()

    build_s2_mosaic(
        center_lat=args.lat,
        center_lon=args.lon,
        phi_date=args.date,
        out_tif=args.out,
        radius_km=args.radius_km,
        tile_size_km=args.tile_size_km,
        resolution_m=args.resolution_m,
        time_window_days=args.time_window_days,
        max_cloud_coverage=args.max_cloud,
    )


if __name__ == "__main__":
    main()