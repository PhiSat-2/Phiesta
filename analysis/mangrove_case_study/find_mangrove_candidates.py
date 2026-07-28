from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

from pyrawph import connect_insula
from pyrawph.remote.constants import PHISAT2_L1_COLLECTION
from pyrawph.remote.catalog_geometry import (
    catalog_geo_from_feature,
    get_catalog_bbox_lonlat,
    get_catalog_center,
    get_catalog_identifier,
)

WORLDCOVER_BASE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
    "v200/2021/map"
)

MANGROVE_CLASS = 95


def _extract_acq_id(text: Any) -> str | None:
    if text is None:
        return None
    s = str(text)
    m = re.search(r"PHISAT-2_L[01]_0*(\d+)_", s)
    if m:
        return str(int(m.group(1)))
    m = re.search(r"\b0*(\d{3,8})\b", s)
    if m:
        return str(int(m.group(1)))
    return None


def _worldcover_tile_name(lat: float, lon: float) -> str:
    """
    ESA WorldCover v200 tile naming, 3x3 degree tiles.
    Example:
        ESA_WorldCover_10m_2021_v200_N36W006_Map.tif
    """
    lat0 = math.floor(lat / 3.0) * 3
    lon0 = math.floor(lon / 3.0) * 3

    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"

    return (
        f"ESA_WorldCover_10m_2021_v200_"
        f"{ns}{abs(int(lat0)):02d}{ew}{abs(int(lon0)):03d}_Map.tif"
    )


def _tiles_for_bbox(bbox_lonlat: tuple[float, float, float, float]) -> list[str]:
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat

    # small epsilon avoids including an extra tile exactly on max boundary
    eps = 1e-9
    lat_start = math.floor(min_lat / 3.0) * 3
    lat_end = math.floor((max_lat - eps) / 3.0) * 3
    lon_start = math.floor(min_lon / 3.0) * 3
    lon_end = math.floor((max_lon - eps) / 3.0) * 3

    names = []
    lat = lat_start
    while lat <= lat_end:
        lon = lon_start
        while lon <= lon_end:
            names.append(_worldcover_tile_name(lat, lon))
            lon += 3
        lat += 3

    return sorted(set(names))


def _download_worldcover_tile(tile_name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / tile_name
    if out.exists() and out.stat().st_size > 0:
        return out

    url = f"{WORLDCOVER_BASE}/{tile_name}"
    print(f"[WorldCover] downloading {tile_name}")
    r = requests.get(url, timeout=300)
    if not r.ok:
        raise RuntimeError(f"Failed to download {url}: HTTP {r.status_code}")

    out.write_bytes(r.content)
    return out


def _count_mangrove_in_tile(
    tile_path: Path,
    bbox_lonlat: tuple[float, float, float, float],
) -> dict[str, Any]:
    with rasterio.open(tile_path) as src:
        # bbox_lonlat is EPSG:4326. WorldCover tiles are usually EPSG:4326,
        # but keep this robust.
        if str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            bounds = transform_bounds("EPSG:4326", src.crs, *bbox_lonlat, densify_pts=21)
        else:
            bounds = bbox_lonlat

        win = from_bounds(*bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()

        # clamp window
        col0 = max(0, int(win.col_off))
        row0 = max(0, int(win.row_off))
        col1 = min(src.width, col0 + int(win.width))
        row1 = min(src.height, row0 + int(win.height))

        if col1 <= col0 or row1 <= row0:
            return {"pixels": 0, "mangrove_pixels": 0}

        win = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
        arr = src.read(1, window=win)

    valid = arr != 0
    mangrove = arr == MANGROVE_CLASS

    return {
        "pixels": int(valid.sum()),
        "mangrove_pixels": int(mangrove.sum()),
    }


def _feature_to_row(
    feature: dict[str, Any],
    cache_dir: Path,
    min_mangrove_pixels: int = 1,
) -> dict[str, Any] | None:
    catalog_geo = catalog_geo_from_feature(feature)
    if not catalog_geo:
        return None

    bbox = get_catalog_bbox_lonlat(catalog_geo)
    center = get_catalog_center(catalog_geo, order="latlon")
    identifier = get_catalog_identifier(catalog_geo)

    acq_id = (
        _extract_acq_id(identifier)
        or _extract_acq_id(feature.get("id"))
        or _extract_acq_id(json.dumps(feature)[:2000])
    )

    tile_names = _tiles_for_bbox(bbox)

    total_pixels = 0
    total_mangrove = 0
    used_tiles = []

    for tile_name in tile_names:
        try:
            tile_path = _download_worldcover_tile(tile_name, cache_dir)
            counts = _count_mangrove_in_tile(tile_path, bbox)
            total_pixels += counts["pixels"]
            total_mangrove += counts["mangrove_pixels"]
            used_tiles.append(tile_name)
        except Exception as exc:
            print(f"[WARN] {tile_name}: {type(exc).__name__}: {exc}")

    if total_mangrove < min_mangrove_pixels:
        return None

    frac = total_mangrove / max(total_pixels, 1)

    return {
        "product_id": acq_id or "",
        "identifier": identifier or "",
        "center_lat": center[0] if center else "",
        "center_lon": center[1] if center else "",
        "bbox_min_lon": bbox[0],
        "bbox_min_lat": bbox[1],
        "bbox_max_lon": bbox[2],
        "bbox_max_lat": bbox[3],
        "worldcover_tiles": ";".join(used_tiles),
        "valid_worldcover_pixels": total_pixels,
        "mangrove_pixels": total_mangrove,
        "mangrove_fraction": frac,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Find PHISAT-2 L1 acquisitions whose Insula catalog footprint overlaps ESA WorldCover mangroves."
    )
    parser.add_argument("--pages", type=int, default=20, help="Number of Insula search pages to scan.")
    parser.add_argument("--results-per-page", type=int, default=100, help="Insula results per page.")
    parser.add_argument("--out", type=str, default="outputs/mangrove_candidates.csv")
    parser.add_argument("--cache-dir", type=str, default="cache/worldcover")
    parser.add_argument("--min-mangrove-pixels", type=int, default=1)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)

    client = connect_insula()

    rows = []

    for page in range(int(args.pages)):
        print(f"\n[Insula] page {page}")

        data = client.search_ref_data(
            ref_data_collection=PHISAT2_L1_COLLECTION,
            page=page,
            results_per_page=int(args.results_per_page),
        )

        features = data.get("features", [])
        print(f"[Insula] features: {len(features)}")

        if not features:
            break

        for i, feature in enumerate(features):
            try:
                row = _feature_to_row(
                    feature,
                    cache_dir=cache_dir,
                    min_mangrove_pixels=int(args.min_mangrove_pixels),
                )
                if row is not None:
                    print(
                        "[MATCH]",
                        row["product_id"],
                        "mangrove_pixels=",
                        row["mangrove_pixels"],
                        "frac=",
                        f"{row['mangrove_fraction']:.6f}",
                    )
                    rows.append(row)
            except Exception as exc:
                print(f"[WARN] feature {i}: {type(exc).__name__}: {exc}")

    rows = sorted(
        rows,
        key=lambda r: (int(r["mangrove_pixels"]), float(r["mangrove_fraction"])),
        reverse=True,
    )

    fieldnames = [
        "product_id",
        "identifier",
        "center_lat",
        "center_lon",
        "bbox_min_lon",
        "bbox_min_lat",
        "bbox_max_lon",
        "bbox_max_lat",
        "worldcover_tiles",
        "valid_worldcover_pixels",
        "mangrove_pixels",
        "mangrove_fraction",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone")
    print(f"Candidates: {len(rows)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
