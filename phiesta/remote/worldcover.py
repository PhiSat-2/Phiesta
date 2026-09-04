from __future__ import annotations

from typing import Any
import time

import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

from .catalog_geometry import catalog_geo_from_feature
from .constants import PHISAT2_L1_COLLECTION


PLANETARY_COMPUTER_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PLANETARY_COMPUTER_DATA = "https://planetarycomputer.microsoft.com/api/data/v1"
WORLDCOVER_COLLECTION = "esa-worldcover"
WORLDCOVER_DATETIME = "2021-01-01T00:00:00Z/2021-12-31T23:59:59Z"

WORLDCOVER_CLASSES = {
    "tree_cover": 10,
    "shrubland": 20,
    "grassland": 30,
    "cropland": 40,
    "built_up": 50,
    "bare_sparse_vegetation": 60,
    "snow_and_ice": 70,
    "permanent_water": 80,
    "herbaceous_wetland": 90,
    "mangroves": 95,
    "moss_and_lichen": 100,
}

_WORLDCOVER_ALIASES = {
    "tree": "tree_cover",
    "trees": "tree_cover",
    "forest": "tree_cover",
    "forests": "tree_cover",
    "built": "built_up",
    "builtup": "built_up",
    "urban": "built_up",
    "bare": "bare_sparse_vegetation",
    "bare_sparse": "bare_sparse_vegetation",
    "water": "permanent_water",
    "water_body": "permanent_water",
    "water_bodies": "permanent_water",
    "wetland": "herbaceous_wetland",
    "wetlands": "herbaceous_wetland",
    "mangrove": "mangroves",
    "moss_lichen": "moss_and_lichen",
}


def _normalize_class_name(value: str) -> str:
    s = str(value).strip().lower()
    for ch in ("-", " ", "/", "\\"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def resolve_worldcover_class(value: str | int) -> tuple[int, str]:
    if isinstance(value, (int, np.integer)):
        code = int(value)
        for name, known_code in WORLDCOVER_CLASSES.items():
            if known_code == code:
                return code, name
        raise ValueError(
            f"Unknown WorldCover class code {code}. "
            f"Known codes: {sorted(WORLDCOVER_CLASSES.values())}"
        )

    s = _normalize_class_name(str(value))
    if s.isdigit():
        return resolve_worldcover_class(int(s))

    s = _WORLDCOVER_ALIASES.get(s, s)
    if s not in WORLDCOVER_CLASSES:
        known = ", ".join(sorted(WORLDCOVER_CLASSES))
        raise ValueError(f"Unknown WorldCover class {value!r}. Known classes: {known}")

    return WORLDCOVER_CLASSES[s], s


def _buffer_lonlat_geometry_km(geometry, distance_km: float):
    distance_km = float(distance_km)
    if distance_km < 0:
        raise ValueError("spatial_tolerance_km must be >= 0.")
    if distance_km == 0:
        return geometry

    center = geometry.centroid
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={float(center.y):.12f} "
        f"+lon_0={float(center.x):.12f} +datum=WGS84 +units=m +no_defs"
    )
    forward = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    backward = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)

    local = shapely_transform(forward.transform, geometry)
    buffered = local.buffer(distance_km * 1000.0)
    return shapely_transform(backward.transform, buffered)


def _count_class_in_dataset(src, geometry_lonlat, class_code: int):
    """
    Local raster helper retained for deterministic unit tests.
    The production WorldCover backend below does not use Rasterio remote I/O.
    """
    geom = geometry_lonlat
    if src.crs is not None and CRS.from_user_input(src.crs) != CRS.from_epsg(4326):
        transformer = Transformer.from_crs(
            "EPSG:4326",
            src.crs,
            always_xy=True,
        )
        geom = shapely_transform(transformer.transform, geometry_lonlat)

    minx, miny, maxx, maxy = geom.bounds
    left = max(float(minx), float(src.bounds.left))
    bottom = max(float(miny), float(src.bounds.bottom))
    right = min(float(maxx), float(src.bounds.right))
    top = min(float(maxy), float(src.bounds.top))

    if left >= right or bottom >= top:
        return 0, 0

    win = from_bounds(left, bottom, right, top, transform=src.transform)
    win = win.round_offsets().round_lengths()

    col0 = max(0, int(win.col_off))
    row0 = max(0, int(win.row_off))
    col1 = min(src.width, col0 + int(win.width))
    row1 = min(src.height, row0 + int(win.height))

    if col1 <= col0 or row1 <= row0:
        return 0, 0

    win = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
    arr = src.read(1, window=win, masked=True)
    data = np.asarray(arr.data)
    data_mask = np.ma.getmaskarray(arr)

    inside = geometry_mask(
        [mapping(geom)],
        out_shape=data.shape,
        transform=src.window_transform(win),
        invert=True,
        all_touched=False,
    )

    valid = inside & (~data_mask) & (data != 0)
    target = valid & (data == int(class_code))
    return int(valid.sum()), int(target.sum())


def _count_class_in_local_tile(tile_path, geometry_lonlat, class_code: int):
    with rasterio.open(tile_path) as src:
        return _count_class_in_dataset(
            src,
            geometry_lonlat,
            class_code=class_code,
        )


def _pc_search_worldcover_items(
    geometry_lonlat,
    *,
    timeout: int = 60,
) -> list[dict[str, Any]]:
    """
    Find the WorldCover 2021 STAC items intersecting one geometry.

    The Planetary Computer STAC API is public and requires no account/token.
    """
    payload = {
        "collections": [WORLDCOVER_COLLECTION],
        "intersects": mapping(geometry_lonlat),
        "datetime": WORLDCOVER_DATETIME,
        "limit": 100,
    }

    response = requests.post(
        f"{PLANETARY_COMPUTER_STAC}/search",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return list(data.get("features", []))


def _find_band_statistics(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the first rio-tiler BandStatistics dictionary from a Planetary
    Computer GeoJSON statistics response.
    """
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
        if not features:
            return {}
        payload = features[0]

    props = payload.get("properties") or {}
    stats = props.get("statistics") or {}

    # Most item-statistics responses are {"statistics": {"map": {...}}}
    # or {"statistics": {"b1": {...}}}. Keep the parser tolerant.
    stack = [stats]
    while stack:
        obj = stack.pop()
        if not isinstance(obj, dict):
            continue
        if "histogram" in obj and (
            "valid_pixels" in obj or "count" in obj
        ):
            return obj
        stack.extend(v for v in obj.values() if isinstance(v, dict))

    return {}


def _categorical_count_from_stats(
    stats: dict[str, Any],
    class_code: int,
) -> tuple[int, int]:
    """
    Return (valid_pixels, class_pixels) from rio-tiler categorical statistics.
    """
    valid_pixels = int(round(float(
        stats.get("valid_pixels", stats.get("count", 0))
    )))

    histogram = stats.get("histogram")
    if not isinstance(histogram, (list, tuple)) or len(histogram) != 2:
        return valid_pixels, 0

    counts, categories = histogram
    if not isinstance(counts, (list, tuple)) or not isinstance(
        categories, (list, tuple)
    ):
        return valid_pixels, 0

    class_pixels = 0
    for i, value in enumerate(categories):
        if i >= len(counts):
            break
        try:
            if float(value) == float(class_code):
                class_pixels += int(round(float(counts[i])))
        except (TypeError, ValueError):
            continue

    return valid_pixels, class_pixels


def _pc_item_worldcover_stats(
    item: dict[str, Any],
    geometry_lonlat,
    class_code: int,
    *,
    statistics_max_size: int | None = 1024,
    timeout: int = 120,
) -> tuple[int, int]:
    """
    Ask the Planetary Computer Data API to compute categorical statistics
    server-side for one WorldCover item and one GeoJSON geometry.

    No raster bytes are stored locally and no local GDAL remote-I/O backend
    is involved.
    """
    item_id = str(item["id"])

    item_geom = shape(item["geometry"])
    query_geom = geometry_lonlat.intersection(item_geom)
    if query_geom.is_empty:
        return 0, 0

    params: list[tuple[str, Any]] = [
        ("collection", WORLDCOVER_COLLECTION),
        ("item", item_id),
        ("assets", "map"),
        ("categorical", "true"),
        ("c", int(class_code)),
        ("nodata", 0),
    ]
    if statistics_max_size is not None:
        params.append(("max_size", int(statistics_max_size)))

    # The runtime endpoint expects the GeoJSON Feature itself as the
    # request body, not an outer {"geojson": feature} wrapper.
    body = {
        "type": "Feature",
        "properties": {},
        "geometry": mapping(query_geom),
    }

    url = f"{PLANETARY_COMPUTER_DATA}/item/statistics"
    retry_statuses = {429, 500, 502, 503, 504}
    response = None

    for attempt in range(4):
        response = requests.post(
            url,
            params=params,
            json=body,
            timeout=timeout,
        )

        if response.ok:
            break

        if response.status_code not in retry_statuses or attempt == 3:
            text = response.text[:1000]
            raise RuntimeError(
                "Planetary Computer WorldCover statistics request failed: "
                f"HTTP {response.status_code} for item {item_id}. {text}"
            )

        time.sleep(1.5 * (2 ** attempt))

    if response is None or not response.ok:
        raise RuntimeError(
            f"Planetary Computer WorldCover statistics request failed for {item_id}."
        )

    stats = _find_band_statistics(response.json())
    if not stats:
        raise RuntimeError(
            "Planetary Computer returned no usable WorldCover statistics "
            f"for item {item_id}."
        )

    return _categorical_count_from_stats(stats, class_code)


def worldcover_stats_for_feature(
    feature: dict[str, Any],
    worldcover: str | int,
    *,
    spatial_tolerance_km: float = 30.0,
    statistics_max_size: int | None = 1024,
) -> dict[str, Any]:
    """
    Compute WorldCover class presence/fraction for one PhiSat-2 catalog feature.

    The production backend uses Microsoft Planetary Computer's public STAC and
    Data APIs. All raster work is performed server-side, avoiding full tile
    downloads and Rasterio/GDAL remote-I/O portability issues on Windows.
    """
    class_code, class_name = resolve_worldcover_class(worldcover)

    catalog_geo = catalog_geo_from_feature(
        feature,
        ref_data_collection=PHISAT2_L1_COLLECTION,
    )

    geom = shape(catalog_geo["geometry_geojson"])
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        raise ValueError("Empty catalog geometry.")

    search_geom = _buffer_lonlat_geometry_km(
        geom,
        distance_km=float(spatial_tolerance_km),
    )

    items = _pc_search_worldcover_items(search_geom)

    valid_pixels = 0
    class_pixels = 0
    used_items: list[str] = []

    for item in items:
        valid, target = _pc_item_worldcover_stats(
            item,
            search_geom,
            class_code=class_code,
            statistics_max_size=statistics_max_size,
        )
        valid_pixels += int(valid)
        class_pixels += int(target)
        used_items.append(str(item["id"]))

    fraction = (
        float(class_pixels) / float(valid_pixels)
        if valid_pixels > 0
        else 0.0
    )

    props = feature.get("properties", {})
    center = catalog_geo.get("center_lonlat") or [None, None]

    return {
        "product_id": catalog_geo.get("identifier"),
        "product_identifier": props.get("productIdentifier"),
        "filename": props.get("filename"),
        "start_datetime": catalog_geo.get("start_datetime"),
        "center_lon": center[0],
        "center_lat": center[1],
        "worldcover_class": class_name,
        "worldcover_code": int(class_code),
        "worldcover_fraction": float(fraction),
        "worldcover_pixels": int(class_pixels),
        "worldcover_valid_pixels": int(valid_pixels),
        "spatial_tolerance_km": float(spatial_tolerance_km),
        "worldcover_items": ";".join(used_items),
    }


def search_l1_worldcover(
    client,
    worldcover: str | int,
    *,
    min_fraction: float = 1e-6,
    spatial_tolerance_km: float = 30.0,
    statistics_max_size: int | None = 1024,
    results_per_page: int = 100,
    max_catalog_products: int | None = None,
    include_uncertain: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    High-recall catalog prefilter by ESA WorldCover content.

    Defaults:
      - min_fraction = 1e-6
      - spatial_tolerance_km = 30
      - statistics_max_size = 1024
      - include_uncertain = True

    WorldCover raster processing is delegated to the public Planetary Computer
    Data API; no WorldCover tiles or PhiSat-2 products are downloaded locally.

    If the external WorldCover service still fails for one acquisition after
    its internal retries, the default high-recall behavior is conservative:
    the acquisition is retained as ``worldcover_status="uncertain"`` rather
    than being silently discarded or aborting the whole catalog scan.

    Set ``include_uncertain=False`` to preserve fail-fast behavior.
    Set ``min_fraction=0`` for literal non-zero class presence.
    """
    min_fraction = float(min_fraction)
    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError("min_fraction must be between 0 and 1.")

    spatial_tolerance_km = float(spatial_tolerance_km)
    if spatial_tolerance_km < 0:
        raise ValueError("spatial_tolerance_km must be >= 0.")

    class_code, class_name = resolve_worldcover_class(worldcover)

    rows: list[dict[str, Any]] = []
    scanned = 0
    uncertain_count = 0

    for feature in client.iter_ref_data(
        ref_data_collection=PHISAT2_L1_COLLECTION,
        results_per_page=int(results_per_page),
    ):
        scanned += 1

        try:
            stats = worldcover_stats_for_feature(
                feature,
                class_code,
                spatial_tolerance_km=spatial_tolerance_km,
                statistics_max_size=statistics_max_size,
            )
        except Exception as exc:
            if not include_uncertain:
                raise

            try:
                catalog_geo = catalog_geo_from_feature(
                    feature,
                    ref_data_collection=PHISAT2_L1_COLLECTION,
                )
                props = feature.get("properties", {})
                center = catalog_geo.get("center_lonlat") or [None, None]
                product_id = catalog_geo.get("identifier")
                product_identifier = props.get("productIdentifier")
                filename = props.get("filename")
                start_datetime = catalog_geo.get("start_datetime")
            except Exception:
                props = feature.get("properties", {})
                product_id = None
                product_identifier = props.get("productIdentifier")
                filename = props.get("filename")
                start_datetime = props.get("startDate")
                center = [None, None]

            error_text = f"{type(exc).__name__}: {exc}"
            if len(error_text) > 1200:
                error_text = error_text[:1200] + "..."

            rows.append({
                "product_id": product_id,
                "product_identifier": product_identifier,
                "filename": filename,
                "start_datetime": start_datetime,
                "center_lon": center[0],
                "center_lat": center[1],
                "worldcover_class": class_name,
                "worldcover_code": int(class_code),
                "worldcover_fraction": np.nan,
                "worldcover_pixels": np.nan,
                "worldcover_valid_pixels": np.nan,
                "spatial_tolerance_km": float(spatial_tolerance_km),
                "worldcover_items": "",
                "worldcover_status": "uncertain",
                "worldcover_error": error_text,
            })
            uncertain_count += 1

            if verbose:
                print(
                    "[WorldCover uncertain]",
                    product_id or product_identifier or scanned,
                    class_name,
                    error_text.splitlines()[0],
                )

            if verbose and scanned % 100 == 0:
                print(
                    f"[WorldCover] scanned={scanned}, candidates={len(rows)}, "
                    f"uncertain={uncertain_count}, class={class_name}"
                )

            if (
                max_catalog_products is not None
                and scanned >= int(max_catalog_products)
            ):
                break

            continue

        if (
            stats["worldcover_pixels"] > 0
            and stats["worldcover_fraction"] >= min_fraction
        ):
            stats = dict(stats)
            stats["worldcover_status"] = "matched"
            stats["worldcover_error"] = ""
            rows.append(stats)

            if verbose:
                print(
                    "[WorldCover match]",
                    stats["product_id"],
                    class_name,
                    f"fraction={stats['worldcover_fraction']:.8f}",
                    f"pixels={stats['worldcover_pixels']}",
                )

        if verbose and scanned % 100 == 0:
            print(
                f"[WorldCover] scanned={scanned}, candidates={len(rows)}, "
                f"uncertain={uncertain_count}, class={class_name}"
            )

        if (
            max_catalog_products is not None
            and scanned >= int(max_catalog_products)
        ):
            break

    columns = [
        "product_id",
        "product_identifier",
        "filename",
        "start_datetime",
        "center_lon",
        "center_lat",
        "worldcover_class",
        "worldcover_code",
        "worldcover_fraction",
        "worldcover_pixels",
        "worldcover_valid_pixels",
        "spatial_tolerance_km",
        "worldcover_items",
        "worldcover_status",
        "worldcover_error",
    ]

    table = pd.DataFrame(rows, columns=columns)

    if not table.empty:
        table["_status_rank"] = table["worldcover_status"].map(
            {"matched": 0, "uncertain": 1}
        ).fillna(2)
        table = (
            table.sort_values(
                ["_status_rank", "worldcover_fraction", "worldcover_pixels"],
                ascending=[True, False, False],
                na_position="last",
            )
            .drop(columns=["_status_rank"])
            .reset_index(drop=True)
        )

    if verbose:
        matched_count = int(
            (table["worldcover_status"] == "matched").sum()
        ) if not table.empty else 0
        print(
            f"[WorldCover] done: scanned={scanned}, matched={matched_count}, "
            f"uncertain={uncertain_count}, candidates={len(table)}, "
            f"class={class_name}, min_fraction={min_fraction:g}, "
            f"tolerance_km={spatial_tolerance_km:g}"
        )

    return table
