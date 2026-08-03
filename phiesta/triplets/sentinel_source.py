from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.wkt import loads

from ..remote.catalog_geometry import (
    get_catalog_acquisition_datetime,
    get_catalog_corners,
    get_catalog_identifier,
)
from .models import SentinelSource


CDSE_PRODUCTS_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def _buffer_lonlat_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    buffer_km: float,
):
    mean_lat = (min_lat + max_lat) / 2.0
    buffer_deg_lat = buffer_km / 111.32
    buffer_deg_lon = buffer_km / (111.32 * np.cos(np.radians(mean_lat)))

    return (
        min_lon - buffer_deg_lon,
        min_lat - buffer_deg_lat,
        max_lon + buffer_deg_lon,
        max_lat + buffer_deg_lat,
    )


def _bbox_to_wkt(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> str:
    return (
        f"POLYGON(("
        f"{min_lon} {min_lat}, "
        f"{max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, "
        f"{min_lon} {max_lat}, "
        f"{min_lon} {min_lat}"
        f"))"
    )


def _get_product_attr(product: dict, name: str, default=None):
    for attr in product.get("Attributes", []):
        if attr.get("Name") == name:
            return attr.get("Value", default)
    return default


def _clean_cdse_wkt(raw: Any) -> str | None:
    if not raw:
        return None

    text = str(raw)

    if "SRID=4326;" in text:
        text = text.split("SRID=4326;", 1)[1]

    text = text.replace("'", "").replace("geography", "").strip()
    return text or None


def _fetch_l1c_twin(
    session: requests.Session,
    l2a_product: dict
) -> dict | None:
    exact_time = l2a_product["ContentDate"]["Start"]
    tile_id = l2a_product["Name"].split("_")[5]

    name = l2a_product["Name"]

    if "S2A_" in name:
        satellite = "S2A"
    elif "S2B_" in name:
        satellite = "S2B"

    query = (
        "Collection/Name eq 'SENTINEL-2' "
        f"and ContentDate/Start eq {exact_time} "
        f"and contains(Name, '_{tile_id}_') "
        f"and contains(Name, '{satellite}_') "
        "and Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'S2MSI1C')"
    )

    res = session.get(
        CDSE_PRODUCTS_URL,
        params={"$filter": query, "$select": "Id,Name,S3Path"},
        timeout=60,
    ).json().get("value", [])

    if not res:
        return None

    item = res[0]

    return {
        "l1c": item.get("S3Path", "").rstrip("/"),
        "l1c_id": item.get("Id"),
        "l1c_name": item.get("Name"),
        "l2a": l2a_product.get("S3Path", "").rstrip("/"),
        "l2a_id": l2a_product.get("Id"),
        "l2a_name": l2a_product.get("Name"),
        "tile_id": tile_id,
        "time": exact_time,
    }


def _verify_mosaic_coverage(
    session: requests.Session,
    candidate_products: list[dict],
    target_box,
) -> float:
    if not candidate_products:
        return 0.0

    names_filter = " or ".join(
        [f"Name eq '{p['Name']}'" for p in candidate_products]
    )

    res = session.get(
        CDSE_PRODUCTS_URL,
        params={"$filter": names_filter},
        timeout=60,
    ).json().get("value", [])

    if not res:
        return 0.0

    polygons = []

    for item in res:
        raw_wkt = (
            item.get("Footprint")
            or item.get("GeoFootprint")
            or item.get("OData.CSC.Footprint")
            or ""
        )
        wkt = _clean_cdse_wkt(raw_wkt)
        if not wkt:
            continue

        try:
            polygons.append(loads(wkt))
        except Exception:
            pass

    if not polygons:
        return 0.0

    mosaic_geom = unary_union(polygons)
    return float(mosaic_geom.intersection(target_box).area / target_box.area)


def find_best_sentinel_source_for_bbox(
    *,
    product_id: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    target_datetime: str,
    satellite: str | list[str] | None = None,
    buffer_km: float = 10.0,
    window_days: int = 15,
    max_cloud_cover: float = 20.0,
    min_coverage: float = 0.85,
    w_time: float = 0.0,
    w_cloud: float = 1.0,
    max_candidates_to_verify: int = 5,
    session: requests.Session | None = None,
) -> SentinelSource:
    """
    Select the best Sentinel-2 source around a PhiSat-2 catalog footprint.

    The selection currently favors low cloud cover and sufficient spatial coverage.
    The temporal search window is symmetric around the PhiSat-2 acquisition date.
    """
    session = session or requests.Session()

    if satellite is None:
        satellite = ["S2A", "S2B"]

    min_lon_b, min_lat_b, max_lon_b, max_lat_b = _buffer_lonlat_bbox(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        buffer_km=buffer_km,
    )

    target_box = box(min_lon_b, min_lat_b, max_lon_b, max_lat_b)
    area_wkt = _bbox_to_wkt(min_lon_b, min_lat_b, max_lon_b, max_lat_b)

    target_dt = pd.to_datetime(target_datetime).tz_localize(None)
    start_str = (target_dt - timedelta(days=window_days)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )
    end_str = (target_dt + timedelta(days=window_days)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )

    satellite_filter = " or ".join([f"contains(Name, '{s}_')" for s in satellite])

    query_l2a = (
        "Collection/Name eq 'SENTINEL-2' "
        f"and ({satellite_filter}) "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{area_wkt}') "
        f"and ContentDate/Start gt {start_str} "
        f"and ContentDate/Start lt {end_str} "
        "and Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
    )

    res_l2a = session.get(
        CDSE_PRODUCTS_URL,
        params={
            "$filter": query_l2a,
            "$expand": "Attributes",
            "$select": "Name,ContentDate,S3Path,Attributes",
            "$top": 500,
        },
        timeout=90,
    ).json().get("value", [])

    if not res_l2a:
        raise ValueError(
            f"No {satellite} L2A product found for product_id={product_id} "
            f"within ±{window_days} days and buffer_km={buffer_km}."
        )

    # Keep only products under the requested cloud threshold.
    filtered = []
    for product in res_l2a:
        cloud = float(_get_product_attr(product, "cloudCover", 100.0))
        if cloud <= float(max_cloud_cover):
            filtered.append(product)

    if not filtered:
        raise ValueError(
            f"{len(res_l2a)} {satellite} L2A products found, but none below "
            f"max_cloud_cover={max_cloud_cover}%."
        )

    datatakes: dict[str, list[dict]] = defaultdict(list)
    for product in filtered:
        day = product["ContentDate"]["Start"][:10]
        datatakes[day].append(product)

    candidates = []

    for day, products in datatakes.items():
        clouds = [
            float(_get_product_attr(p, "cloudCover", 100.0))
            for p in products
        ]
        avg_cloud = float(np.mean(clouds))

        s2_dt = pd.to_datetime(products[0]["ContentDate"]["Start"]).tz_localize(None)
        delta_days = abs((s2_dt - target_dt).total_seconds()) / 86400.0

        candidates.append(
            {
                "day": day,
                "products": products,
                "cloud_cover": avg_cloud,
                "delta_days": float(delta_days),
                "base_score": float(delta_days * w_time + avg_cloud * w_cloud),
            }
        )

    candidates.sort(key=lambda x: x["base_score"])
    candidates = candidates[:max_candidates_to_verify]

    best = None
    best_score = float("inf")

    for candidate in candidates:
        coverage_fraction = _verify_mosaic_coverage(
            session=session,
            candidate_products=candidate["products"],
            target_box=target_box,
        )

        if coverage_fraction < float(min_coverage):
            continue

        candidate["coverage_fraction"] = float(coverage_fraction)

        missing_data_pct = (1.0 - coverage_fraction) * 100.0
        effective_bad_pixels = missing_data_pct + candidate["cloud_cover"]

        score = float(candidate["delta_days"] * w_time + effective_bad_pixels * w_cloud)

        if score < best_score:
            best_score = score
            best = candidate

    if best is None:
        raise ValueError(
            f"No valid {satellite} mosaic candidate reached min_coverage={min_coverage:.2f}."
        )

    final_pairs = []

    with ThreadPoolExecutor(max_workers=max(1, len(best["products"]))) as executor:
        futures = [
            executor.submit(_fetch_l1c_twin, session, product)
            for product in best["products"]
        ]

        for future in futures:
            pair = future.result()
            if pair is not None:
                final_pairs.append(pair)

    if not final_pairs:
        raise ValueError(
            f"Found {satellite} L2A products but no matching L1C twins."
        )
    
    selected_satellites = set()

    for p in best["products"]:
        if "S2A_" in p["Name"]:
            selected_satellites.add("S2A")
        elif "S2B_" in p["Name"]:
            selected_satellites.add("S2B")

    if len(selected_satellites) == 1:
        selected_satellite = selected_satellites.pop()
    else:
        selected_satellite = "+".join(sorted(selected_satellites))

    return SentinelSource(
        product_id=str(product_id),
        satellite=selected_satellite,
        s2_datetime=best["products"][0]["ContentDate"]["Start"],
        delta_days=round(float(best["delta_days"]), 3),
        cloud_cover=round(float(best["cloud_cover"]), 3),
        coverage=round(float(best["coverage_fraction"]) * 100.0, 3),
        l1c_paths=[p["l1c"] for p in final_pairs],
        l2a_paths=[p["l2a"] for p in final_pairs],
        metadata={
            "s2_day": best["day"],
            "buffer_km": float(buffer_km),
            "window_days": int(window_days),
            "max_cloud_cover": float(max_cloud_cover),
            "min_coverage": float(min_coverage),
            "coverage_fraction": float(best["coverage_fraction"]),
            "num_tiles": len(best["products"]),
            "pairs": final_pairs,
        },
    )


def find_best_sentinel_source(
    event: Any,
    *,
    product_id: str | None = None,
    satellite: str | list[str] | None = None,
    buffer_km: float = 10.0,
    window_days: int = 15,
    max_cloud_cover: float = 20.0,
    min_coverage: float = 0.85,
    session: requests.Session | None = None,
) -> SentinelSource:
    """
    Select the best Sentinel-2 source for a PhiSat-2 event using catalog geometry.
    """
    corners_lonlat = get_catalog_corners(event, order="lonlat")
    if not corners_lonlat:
        raise ValueError(
            "No catalog corners available on event. "
            "Load the product from Insula or enrich it with catalog_geo first."
        )

    lons = [p[0] for p in corners_lonlat]
    lats = [p[1] for p in corners_lonlat]

    target_datetime = get_catalog_acquisition_datetime(event)
    if target_datetime is None:
        raise ValueError(
            "Could not infer PhiSat-2 acquisition datetime from catalog metadata."
        )

    pid = product_id or get_catalog_identifier(event) or "unknown"

    return find_best_sentinel_source_for_bbox(
        product_id=str(pid),
        min_lon=min(lons),
        min_lat=min(lats),
        max_lon=max(lons),
        max_lat=max(lats),
        target_datetime=target_datetime,
        satellite=satellite,
        buffer_km=buffer_km,
        window_days=window_days,
        max_cloud_cover=max_cloud_cover,
        min_coverage=min_coverage,
        session=session,
    )