from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import re

LonLat = Tuple[float, float]
LatLon = Tuple[float, float]

import math

def extract_phisat_acquisition_id(text: Any) -> Optional[str]:
    """
    Extract the numeric PhiSat-2 acquisition id from a bare id, filename,
    productIdentifier, product folder name, or zip name.

    Examples:
        "5095" -> "5095"
        "000005095" -> "5095"
        "PHISAT-2_L1_000005095_..." -> "5095"
        "phisat_PHISAT-2_L1_000005095_....zip" -> "5095"
    """
    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None

    if s.isdigit():
        return str(int(s))

    m = re.search(r"PHISAT-2_L[01]_0*(\d+)_", s, flags=re.IGNORECASE)
    if m is not None:
        return str(int(m.group(1)))

    return None


# Backward-compatible internal alias.
_parse_identifier_from_product_identifier = extract_phisat_acquisition_id


def _parse_iso_datetime_from_product_identifier(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = str(text)
    m = re.search(r"_(\d{14})_", s)
    if m is None:
        return None
    t = m.group(1)
    return f"{t[0:4]}-{t[4:6]}-{t[6:8]}T{t[8:10]}:{t[10:12]}:{t[12:14]}Z"


def _as_lonlat_pair(x: Any) -> LonLat:
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise ValueError(f"Invalid coordinate pair: {x!r}")
    lon = float(x[0])
    lat = float(x[1])
    return (lon, lat)


def _lonlat_to_latlon(pt: LonLat) -> LatLon:
    lon, lat = pt
    return (lat, lon)


def _extract_polygon_ring_lonlat(feature: Dict[str, Any]) -> List[LonLat]:
    geom = feature.get("geometry")
    if not isinstance(geom, dict):
        raise ValueError("Feature has no geometry dictionary.")

    geom_type = geom.get("type")
    coords = geom.get("coordinates")

    if geom_type != "Polygon":
        raise ValueError(f"Expected Polygon geometry, got {geom_type!r}")

    if not isinstance(coords, list) or len(coords) == 0:
        raise ValueError("Polygon geometry has no coordinates.")

    ring = coords[0]
    if not isinstance(ring, list) or len(ring) < 4:
        raise ValueError("Polygon outer ring is invalid.")

    return [_as_lonlat_pair(pt) for pt in ring]


def _strip_closed_ring(ring: List[LonLat]) -> List[LonLat]:
    if len(ring) >= 2 and ring[0] == ring[-1]:
        return ring[:-1]
    return ring


def _extract_corners_from_ring_lonlat(ring: List[LonLat]) -> List[LonLat]:
    ring_open = _strip_closed_ring(ring)
    if len(ring_open) != 4:
        raise ValueError(
            f"Expected a 4-corner polygon, got {len(ring_open)} corner(s)."
        )
    return ring_open


def _extract_centroid_lonlat(feature: Dict[str, Any]) -> Optional[LonLat]:
    props = feature.get("properties", {})
    centroid = props.get("centroid")

    if not isinstance(centroid, dict):
        return None
    if centroid.get("type") != "Point":
        return None

    coords = centroid.get("coordinates")
    if not isinstance(coords, (list, tuple)) or len(coords) != 2:
        return None

    return _as_lonlat_pair(coords)


def _mean_center_lonlat(corners_lonlat: List[LonLat]) -> LonLat:
    lon = sum(pt[0] for pt in corners_lonlat) / len(corners_lonlat)
    lat = sum(pt[1] for pt in corners_lonlat) / len(corners_lonlat)
    return (lon, lat)


def catalog_geo_from_feature(
    feature: Dict[str, Any],
    ref_data_collection: str | None = None,
) -> Dict[str, Any]:
    """
    Build a lightweight, serialization-friendly catalog-geometry dictionary
    from one Insula feature.

    Notes:
    - GeoJSON coordinates are kept in [lon, lat] order.
    - Convenience conversions in (lat, lon) order are also stored for Folium / Leaflet.
    - This object is intentionally separated from raster georeferencing fields
      such as crs / transform / bounds.
    """
    props = feature.get("properties", {})
    ring_lonlat = _extract_polygon_ring_lonlat(feature)
    corners_lonlat = _extract_corners_from_ring_lonlat(ring_lonlat)

    center_lonlat = _extract_centroid_lonlat(feature)
    if center_lonlat is None:
        center_lonlat = _mean_center_lonlat(corners_lonlat)

    ring_latlon = [_lonlat_to_latlon(pt) for pt in ring_lonlat]
    corners_latlon = [_lonlat_to_latlon(pt) for pt in corners_lonlat]
    center_latlon = _lonlat_to_latlon(center_lonlat)

    return {
        "source": "insula_catalog_search",
        "feature_id": feature.get("id"),
        "ref_data_collection": ref_data_collection,
        "product_identifier": props.get("productIdentifier"),
        "filename": props.get("filename"),
        "geometry_type": feature.get("geometry", {}).get("type"),
        "geometry_geojson": feature.get("geometry"),
        "centroid_geojson": props.get("centroid"),
        "polygon_lonlat": [list(pt) for pt in ring_lonlat],
        "polygon_latlon": [list(pt) for pt in ring_latlon],
        "corners_lonlat": [list(pt) for pt in corners_lonlat],
        "corners_latlon": [list(pt) for pt in corners_latlon],
        "center_lonlat": list(center_lonlat),
        "center_latlon": list(center_latlon),
        "identifier": _parse_identifier_from_product_identifier(
            props.get("productIdentifier") or props.get("filename")
        ),
        "start_datetime": props.get("startDate"),
        "completion_datetime": props.get("completionDate"),
    }

def get_catalog_identifier(obj: Any) -> Optional[str]:
    cg = _catalog_geo_from_any(obj)
    ident = cg.get("identifier")
    if ident is not None:
        return str(ident)

    pid = cg.get("product_identifier") or cg.get("filename")
    return _parse_identifier_from_product_identifier(pid)


def get_catalog_acquisition_datetime(obj: Any) -> Optional[str]:
    cg = _catalog_geo_from_any(obj)

    dt = cg.get("start_datetime")
    if dt:
        return str(dt)

    dt = cg.get("completion_datetime")
    if dt:
        return str(dt)

    pid = cg.get("product_identifier") or cg.get("filename")
    return _parse_iso_datetime_from_product_identifier(pid)


def enrich_meta_with_insula_feature(
    meta: Dict[str, Any],
    feature: Dict[str, Any],
    ref_data_collection: str | None = None,
) -> None:
    """
    Mutate an event metadata dictionary in-place using one Insula feature.

    This keeps the current `insula_*` convention already used in the codebase,
    and adds one new nested block: `meta["catalog_geo"]`.
    """
    props = feature.get("properties", {})

    meta["insula_filename"] = props.get("filename")
    meta["insula_product_identifier"] = props.get("productIdentifier")
    meta["insula_download_url"] = props.get("_links", {}).get("download", {}).get("href")
    meta["insula_platform_url"] = props.get("platformUrl")
    meta["catalog_geo"] = catalog_geo_from_feature(
        feature=feature,
        ref_data_collection=ref_data_collection,
    )

def _catalog_geo_from_any(obj: Any) -> Dict[str, Any]:
    """
    Accept either:
    - a raw catalog_geo dict,
    - an event exposing get_meta(),
    - or a metadata dict containing "catalog_geo".
    """
    if hasattr(obj, "get_meta"):
        meta = obj.get_meta()
        if "catalog_geo" not in meta:
            raise KeyError("No 'catalog_geo' found in event metadata.")
        return meta["catalog_geo"]

    if isinstance(obj, dict):
        if "catalog_geo" in obj and isinstance(obj["catalog_geo"], dict):
            return obj["catalog_geo"]
        if "corners_latlon" in obj or "corners_lonlat" in obj:
            return obj

    raise TypeError(
        "Expected an event with get_meta(), a metadata dict containing "
        "'catalog_geo', or a raw catalog_geo dictionary."
    )


def _normalize_order(order: str) -> str:
    s = str(order).strip().lower()
    if s not in {"latlon", "lonlat"}:
        raise ValueError("order must be 'latlon' or 'lonlat'.")
    return s


def get_catalog_corners(obj: Any, order: str = "latlon") -> List[Tuple[float, float]]:
    """
    Return the 4 catalog corners in the requested coordinate order.

    Returns:
        A list of 4 tuples:
        - (lat, lon) if order='latlon'
        - (lon, lat) if order='lonlat'
    """
    cg = _catalog_geo_from_any(obj)
    order = _normalize_order(order)
    key = "corners_latlon" if order == "latlon" else "corners_lonlat"
    vals = cg.get(key)
    if vals is None:
        raise KeyError(f"Missing '{key}' in catalog_geo.")
    return [tuple(map(float, pt)) for pt in vals]


def get_catalog_center(obj: Any, order: str = "latlon") -> Tuple[float, float]:
    """
    Return the catalog center in the requested coordinate order.

    Returns:
        - (lat, lon) if order='latlon'
        - (lon, lat) if order='lonlat'
    """
    cg = _catalog_geo_from_any(obj)
    order = _normalize_order(order)
    key = "center_latlon" if order == "latlon" else "center_lonlat"
    vals = cg.get(key)
    if vals is None:
        raise KeyError(f"Missing '{key}' in catalog_geo.")
    return tuple(map(float, vals))


def get_catalog_polygon(obj: Any, order: str = "latlon", closed: bool = True) -> List[Tuple[float, float]]:
    """
    Return the full catalog polygon ring in the requested coordinate order.

    Args:
        closed: If False, drop the final repeated point when present.
    """
    cg = _catalog_geo_from_any(obj)
    order = _normalize_order(order)
    key = "polygon_latlon" if order == "latlon" else "polygon_lonlat"
    vals = cg.get(key)
    if vals is None:
        raise KeyError(f"Missing '{key}' in catalog_geo.")

    pts = [tuple(map(float, pt)) for pt in vals]

    if not closed and len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]

    return pts


def format_catalog_geo(
    obj: Any,
    order: str = "latlon",
    decimals: int = 6,
) -> str:
    """
    Return a compact human-readable string with center + 4 corners.
    """
    order = _normalize_order(order)
    corners = get_catalog_corners(obj, order=order)
    center = get_catalog_center(obj, order=order)

    if order == "latlon":
        center_label = "center_latlon"
        corners_label = "corners_latlon"
    else:
        center_label = "center_lonlat"
        corners_label = "corners_lonlat"

    def _fmt_pair(pt: Tuple[float, float]) -> str:
        return f"({pt[0]:.{decimals}f}, {pt[1]:.{decimals}f})"

    corner_lines = ",\n  ".join(_fmt_pair(pt) for pt in corners)

    return (
        f"{center_label}: {_fmt_pair(center)}\n"
        f"{corners_label}: [\n  {corner_lines}\n]"
    )


def print_catalog_geo(
    obj: Any,
    order: str = "latlon",
    decimals: int = 6,
) -> None:
    """
    Pretty-print the catalog center and corners.
    """
    print(format_catalog_geo(obj=obj, order=order, decimals=decimals))



def bbox_to_wkt(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    decimals: int = 10,
) -> str:
    """
    Convert a lon/lat bounding box into a WKT polygon.

    Args:
        min_lon: Minimum longitude.
        min_lat: Minimum latitude.
        max_lon: Maximum longitude.
        max_lat: Maximum latitude.
        decimals: Number of decimals used when formatting coordinates.

    Returns:
        A WKT polygon string in lon/lat order.

    Raises:
        ValueError: If the bbox is invalid.
    """
    min_lon = float(min_lon)
    min_lat = float(min_lat)
    max_lon = float(max_lon)
    max_lat = float(max_lat)

    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError(
            "Invalid bbox: expected min_lon < max_lon and min_lat < max_lat."
        )

    pts = [
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ]

    coord_str = ", ".join(f"{lon:.{decimals}f} {lat:.{decimals}f}" for lon, lat in pts)
    return f"POLYGON(({coord_str}))"


def point_buffer_to_wkt(
    lon: float,
    lat: float,
    radius_km: float,
    n_points: int = 64,
    decimals: int = 10,
) -> str:
    """
    Build an approximate circular AOI around a lon/lat point as a WKT polygon.

    The construction uses a local geographic approximation:
    - latitude scale ~= 110.574 km / degree
    - longitude scale ~= 111.320 * cos(latitude) km / degree

    This is sufficient for moderate search radii and catalogue queries.

    Args:
        lon: Center longitude.
        lat: Center latitude.
        radius_km: Radius in kilometers.
        n_points: Number of vertices used to approximate the circle.
        decimals: Number of decimals used when formatting coordinates.

    Returns:
        A WKT polygon string in lon/lat order.

    Raises:
        ValueError: If radius_km <= 0 or n_points < 8.
    """
    lon = float(lon)
    lat = float(lat)
    radius_km = float(radius_km)

    if radius_km <= 0:
        raise ValueError("radius_km must be > 0.")
    if n_points < 8:
        raise ValueError("n_points must be >= 8.")

    lat_km_per_deg = 110.574
    lon_km_per_deg = 111.320 * math.cos(math.radians(lat))
    lon_km_per_deg = max(abs(lon_km_per_deg), 1e-9)

    pts: List[LonLat] = []
    for i in range(n_points):
        theta = 2.0 * math.pi * i / n_points
        dlon = (radius_km * math.cos(theta)) / lon_km_per_deg
        dlat = (radius_km * math.sin(theta)) / lat_km_per_deg
        pts.append((lon + dlon, lat + dlat))

    pts.append(pts[0])

    coord_str = ", ".join(f"{x:.{decimals}f} {y:.{decimals}f}" for x, y in pts)
    return f"POLYGON(({coord_str}))"


def get_catalog_bbox_lonlat(obj: Any) -> Tuple[float, float, float, float]:
    """
    Return the catalogue bounding box as (min_lon, min_lat, max_lon, max_lat).
    """
    poly = get_catalog_polygon(obj, order="lonlat", closed=False)
    lons = [pt[0] for pt in poly]
    lats = [pt[1] for pt in poly]
    return (min(lons), min(lats), max(lons), max(lats))