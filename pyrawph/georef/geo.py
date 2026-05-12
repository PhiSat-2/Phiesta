from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio import Affine


def affine_2x3_to_rasterio_affine(m: np.ndarray) -> Affine:
    m = np.asarray(m, dtype=np.float64)
    return Affine(
        float(m[0, 0]), float(m[0, 1]), float(m[0, 2]),
        float(m[1, 0]), float(m[1, 1]), float(m[1, 2]),
    )


def project_template_corners(shape_hw, affine_2x3: np.ndarray) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    corners = np.array(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
        dtype=np.float32,
    )
    corners_h = np.hstack([corners, np.ones((4, 1), dtype=np.float32)])
    return (np.asarray(affine_2x3, dtype=np.float32) @ corners_h.T).T.astype(np.float32)


def pixels_to_latlon(corners_px: np.ndarray, transform, crs) -> List[Tuple[float, float]]:
    xs = []
    ys = []
    for x, y in corners_px:
        gx, gy = rasterio.transform.xy(transform, y, x, offset="center")
        xs.append(float(gx))
        ys.append(float(gy))

    if str(crs).upper().endswith("4326"):
        return [(lat, lon) for lon, lat in zip(xs, ys)]

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)
    return [(float(lat), float(lon)) for lat, lon in zip(lats, lons)]


def corners_list_to_dict(corners_latlon: List[Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    return {
        "ul": tuple(corners_latlon[0]),
        "ur": tuple(corners_latlon[1]),
        "lr": tuple(corners_latlon[2]),
        "ll": tuple(corners_latlon[3]),
    }


def corners_dict_to_array(corners_latlon: Dict[str, Tuple[float, float]]) -> np.ndarray:
    return np.array(
        [corners_latlon["ul"], corners_latlon["ur"], corners_latlon["lr"], corners_latlon["ll"]],
        dtype=np.float64,
    )


def compose_world_transform(s2_transform, phi_to_s2_affine_2x3: np.ndarray):
    phi_to_s2 = affine_2x3_to_rasterio_affine(phi_to_s2_affine_2x3)
    return s2_transform * phi_to_s2


def world_bounds_from_transform(shape_hw, world_transform):
    h, w = int(shape_hw[0]), int(shape_hw[1])
    pts = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    xs = []
    ys = []
    for x, y in pts:
        gx, gy = world_transform * (x, y)
        xs.append(float(gx))
        ys.append(float(gy))
    return (min(xs), min(ys), max(xs), max(ys))


def center_from_corners_latlon(corners_latlon: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    vals = np.asarray(list(corners_latlon.values()), dtype=np.float64)
    lat = float(vals[:, 0].mean())
    lon = float(vals[:, 1].mean())
    return (lat, lon)


def haversine_m(lat1, lon1, lat2, lon2):
    import math

    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
