from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from phiesta.remote.catalog_geometry import (
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


def add_footprint_bbox_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add footprint bbox columns derived from corner coordinates.

    Adds:
    - footprint_min_lon
    - footprint_min_lat
    - footprint_max_lon
    - footprint_max_lat
    """
    out = df.copy()

    lon_cols = [f"corner_{i}_lon" for i in range(1, 5) if f"corner_{i}_lon" in out.columns]
    lat_cols = [f"corner_{i}_lat" for i in range(1, 5) if f"corner_{i}_lat" in out.columns]

    if not lon_cols or not lat_cols:
        out["footprint_min_lon"] = None
        out["footprint_min_lat"] = None
        out["footprint_max_lon"] = None
        out["footprint_max_lat"] = None
        return out

    out["footprint_min_lon"] = out[lon_cols].min(axis=1)
    out["footprint_max_lon"] = out[lon_cols].max(axis=1)
    out["footprint_min_lat"] = out[lat_cols].min(axis=1)
    out["footprint_max_lat"] = out[lat_cols].max(axis=1)

    return out


def filter_dataframe_by_bbox(
    df: pd.DataFrame,
    bbox_lonlat: tuple[float, float, float, float],
) -> pd.DataFrame:
    """
    Filter a product table by intersection with a lon/lat bbox.

    Args:
        df: DataFrame returned by search_result_to_dataframe or search_l1_table.
        bbox_lonlat: (min_lon, min_lat, max_lon, max_lat).

    Returns:
        Filtered DataFrame.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat

    out = add_footprint_bbox_columns(df)

    intersects = (
        (out["footprint_max_lon"] >= min_lon)
        & (out["footprint_min_lon"] <= max_lon)
        & (out["footprint_max_lat"] >= min_lat)
        & (out["footprint_min_lat"] <= max_lat)
    )

    return out[intersects].copy()


def search_bbox_table(
    client,
    *,
    level: str = "L1",
    bbox_lonlat: tuple[float, float, float, float],
    pages: int = 40,
    results_per_page: int = 100,
    **search_kwargs,
) -> pd.DataFrame:
    """
    Search several Insula pages and return products intersecting a lon/lat bbox.

    This uses catalog footprints from Insula. It is intended for discovery and
    approximate filtering, not precise pixel-level georeferencing.

    Args:
        client: InsulaClient.
        level: "L1" or "L0".
        bbox_lonlat: (min_lon, min_lat, max_lon, max_lat).
        pages: maximum number of pages to scan.
        results_per_page: number of products per Insula page.
        **search_kwargs: extra arguments forwarded to search_l1/search_l0.

    Returns:
        Filtered compact DataFrame.
    """
    all_tables = []

    level = level.upper()

    for page in range(int(pages)):
        kwargs = dict(search_kwargs)
        kwargs["page"] = page
        kwargs["results_per_page"] = results_per_page

        if level == "L1":
            result = client.search_l1(**kwargs)
        elif level == "L0":
            result = client.search_l0(**kwargs)
        else:
            raise ValueError(f"Unsupported level: {level!r}. Use 'L1' or 'L0'.")

        df_page = search_result_to_dataframe(result)

        if df_page.empty:
            break

        all_tables.append(df_page)

    if not all_tables:
        return pd.DataFrame()

    df = pd.concat(all_tables, ignore_index=True)
    df = filter_dataframe_by_bbox(df, bbox_lonlat=bbox_lonlat)

    if "start_datetime" in df.columns:
        df = df.sort_values("start_datetime", ascending=False, na_position="last")

    return df.reset_index(drop=True)


def _corners_from_row(row) -> list[list[float]]:
    """
    Extract footprint corners from a search table row.
    """
    corners = []

    for idx in range(1, 5):
        lon_key = f"corner_{idx}_lon"
        lat_key = f"corner_{idx}_lat"

        if lon_key in row and lat_key in row:
            lon = row[lon_key]
            lat = row[lat_key]

            if pd.notna(lon) and pd.notna(lat):
                corners.append([float(lon), float(lat)])

    if corners:
        return corners

    if "corners_lonlat" in row and isinstance(row["corners_lonlat"], str) and row["corners_lonlat"]:
        try:
            return json.loads(row["corners_lonlat"])
        except Exception:
            return []

    return []


def search_table_to_geojson(df: pd.DataFrame) -> dict:
    """
    Convert a compact search table into a GeoJSON FeatureCollection.

    The geometry uses the Insula catalog footprint corners.
    This is intended for visualization and approximate spatial filtering.
    """
    features = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        corners = _corners_from_row(row_dict)

        if len(corners) < 4:
            continue

        polygon = corners + [corners[0]]

        properties = {}
        for key, value in row_dict.items():
            if key.startswith("corner_"):
                continue
            if key in {"corners_lonlat", "polygon_lonlat"}:
                continue

            try:
                if pd.isna(value):
                    value = None
            except Exception:
                pass

            properties[key] = value

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon],
            },
            "properties": properties,
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def export_search_table_geojson(df: pd.DataFrame, out_geojson: str) -> str:
    """
    Export a compact search table to GeoJSON.
    """
    from pathlib import Path

    out = Path(out_geojson)
    out.parent.mkdir(parents=True, exist_ok=True)

    geojson = search_table_to_geojson(df)

    with out.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    return str(out)


def load_products_from_table(
    client,
    table: pd.DataFrame,
    *,
    level: str = "L1",
    limit: int | None = None,
    product_id_col: str = "product_id",
    continue_on_error: bool = True,
    **load_kwargs,
) -> list[dict]:
    """
    Load/download products listed in a compact search table.

    Args:
        client: InsulaClient.
        table: DataFrame containing product ids.
        level: "L1" or "L0".
        limit: optional max number of rows to process.
        product_id_col: column containing product ids.
        continue_on_error: if True, keep going when one product fails.
        **load_kwargs: forwarded to client.load_l1/load_l0.

    Returns:
        List of dictionaries with product_id, status, and object/error.
    """
    if product_id_col not in table.columns:
        raise ValueError(f"Missing column {product_id_col!r} in table.")

    rows = table.head(limit).copy() if limit is not None else table.copy()
    results = []

    level = level.upper()

    for _, row in rows.iterrows():
        product_id = str(row[product_id_col])

        if product_id.endswith(".0"):
            product_id = product_id[:-2]

        try:
            if level == "L1":
                obj = client.load_l1(product_id, **load_kwargs)
            elif level == "L0":
                obj = client.load_l0(product_id, **load_kwargs)
            else:
                raise ValueError(f"Unsupported level: {level!r}. Use 'L1' or 'L0'.")

            results.append({
                "product_id": product_id,
                "status": "SUCCESS",
                "object": obj,
                "error": "",
            })

        except Exception as exc:
            record = {
                "product_id": product_id,
                "status": "FAILED",
                "object": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(record)

            if not continue_on_error:
                raise

    return results
