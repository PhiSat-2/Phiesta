from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from pyrawph.remote.catalog_geometry import (
    catalog_geo_from_feature,
    get_catalog_center,
    get_catalog_identifier,
)


def _iter_features(search_result: Any) -> list[dict]:
    """
    Normalize an Insula search result into a list of GeoJSON-like features.
    """
    if search_result is None:
        return []

    if isinstance(search_result, dict):
        if "features" in search_result:
            return list(search_result.get("features", []))
        if "content" in search_result:
            return list(search_result.get("content", []))

    if isinstance(search_result, list):
        return search_result

    return []


def _collect_strings(obj: Any):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _collect_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _collect_strings(value)
    elif isinstance(obj, str):
        yield obj


def _extract_product_id(text: Any) -> str | None:
    if text is None:
        return None

    s = str(text)

    match = re.search(r"PHISAT-2_L[01]_0*(\d+)_", s)
    if match:
        return str(int(match.group(1)))

    match = re.search(r"\b0*(\d{3,8})\b", s)
    if match:
        return str(int(match.group(1)))

    return None


def _extract_full_filename(feature: dict) -> str:
    strings = list(_collect_strings(feature))

    for s in strings:
        match = re.search(r"PHISAT-2_L[01]_[A-Za-z0-9_]+(?:\.zip)?", s)
        if match:
            return match.group(0)

    for s in strings:
        if "PHISAT-2_L1" in s or "PHISAT-2_L0" in s:
            return s

    return ""


def _properties(feature: dict) -> dict:
    if not isinstance(feature, dict):
        return {}
    props = feature.get("properties")
    return props if isinstance(props, dict) else {}


def _first_existing(props: dict, keys: list[str]):
    for key in keys:
        value = props.get(key)
        if value not in (None, ""):
            return value
    return None


def _row_from_feature(feature: dict) -> dict:
    props = _properties(feature)
    catalog_geo = catalog_geo_from_feature(feature)

    filename = _extract_full_filename(feature)
    product_id = (
        _extract_product_id(filename)
        or _extract_product_id(get_catalog_identifier(catalog_geo) if catalog_geo else None)
        or _extract_product_id(feature.get("id"))
        or _extract_product_id(json.dumps(feature)[:5000])
    )

    center = None
    corners = []
    polygon = []

    if catalog_geo:
        center = get_catalog_center(catalog_geo, order="lonlat")
        corners = catalog_geo.get("corners_lonlat") or []
        polygon = catalog_geo.get("polygon_lonlat") or []

    row = {
        "product_id": product_id,
        "filename": filename,
        "feature_id": feature.get("id"),
        "identifier": get_catalog_identifier(catalog_geo) if catalog_geo else None,
        "start_datetime": (
            _first_existing(props, ["start_datetime", "startDate", "start", "datetime"])
            or (catalog_geo.get("start_datetime") if catalog_geo else None)
        ),
        "completion_datetime": (
            _first_existing(props, ["completion_datetime", "completionDate", "end"])
            or (catalog_geo.get("completion_datetime") if catalog_geo else None)
        ),
        "center_lon": center[0] if center else None,
        "center_lat": center[1] if center else None,
        "geometry_type": catalog_geo.get("geometry_type") if catalog_geo else None,
        "corners_lonlat": json.dumps(corners, separators=(",", ":")) if corners else "",
        "polygon_lonlat": json.dumps(polygon, separators=(",", ":")) if polygon else "",
    }

    for idx in range(4):
        if idx < len(corners):
            lon, lat = corners[idx]
            row[f"corner_{idx + 1}_lon"] = lon
            row[f"corner_{idx + 1}_lat"] = lat
        else:
            row[f"corner_{idx + 1}_lon"] = None
            row[f"corner_{idx + 1}_lat"] = None

    return row


def search_result_to_dataframe(search_result: Any) -> pd.DataFrame:
    """
    Convert an Insula search result into a compact pandas DataFrame.

    The resulting table is designed for users: product id, filename, datetime,
    center, and footprint corners are made explicit.
    """
    features = _iter_features(search_result)
    rows = [_row_from_feature(feature) for feature in features]
    return pd.DataFrame(rows)


def search_result_to_records(search_result: Any) -> list[dict]:
    """
    Convert an Insula search result into a list of compact dictionaries.
    """
    return search_result_to_dataframe(search_result).to_dict(orient="records")


def export_search_result_csv(search_result: Any, out_csv: str) -> str:
    """
    Export an Insula search result to a compact CSV.
    """
    df = search_result_to_dataframe(search_result)
    df.to_csv(out_csv, index=False)
    return out_csv
