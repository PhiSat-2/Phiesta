from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol


from .features import norm01, resize_to_gsd
from .geo import haversine_m
from .models import SentinelMosaic
from .sentinel_provider import LocalRasterProvider
from .build_s2_mosaic_cdse import build_s2_mosaic

from ..utils.display import (
    prepare_event_display_image as _prepare_event_display_image,
)

import cv2

from ..remote.catalog_geometry import (
    get_catalog_center,
    get_catalog_corners,
    get_catalog_polygon,
    get_catalog_identifier,
    get_catalog_acquisition_datetime,
)


_COMPARE_PRESETS = {
    "balanced": {
        "search_schedule_days": (0, 5, 10, 15, 20, 30, 45),
        "max_cloud_coverage": 20,
        "margin_km": 1.0,
        "resolution_m": 10.0,
        "flip_horizontal": True,
        "grid_steps": 6,
        "phi_gsd_m": 4.75,
        "match_output_shape": True,
        "tile_size_km": None,
        "ensure_nonempty": True,
        "min_valid_fraction": 0.25,
        "band_indices": (1, 2, 3, 4),
        "figsize": (12, 6),
        "out_png": None,
        "verbose": True,
        "data_collection": "sentinel-2-l1c",
        "mosaicking_order": "leastCC",
    },
    "strict": {
        "search_schedule_days": (0, 5, 10, 15, 20, 30),
        "max_cloud_coverage": 15,
        "margin_km": 1.0,
        "resolution_m": 10.0,
        "flip_horizontal": True,
        "grid_steps": 6,
        "phi_gsd_m": 4.75,
        "match_output_shape": True,
        "tile_size_km": None,
        "ensure_nonempty": True,
        "min_valid_fraction": 0.35,
        "band_indices": (1, 2, 3, 4),
        "figsize": (12, 6),
        "out_png": None,
        "verbose": True,
        "data_collection": "sentinel-2-l1c",
        "mosaicking_order": "leastCC",
    },
    "relaxed": {
        "search_schedule_days": (0, 5, 10, 15, 20, 30, 45, 60),
        "max_cloud_coverage": 60,
        "margin_km": 2.0,
        "resolution_m": 10.0,
        "flip_horizontal": True,
        "grid_steps": 6,
        "phi_gsd_m": 4.75,
        "match_output_shape": True,
        "tile_size_km": None,
        "ensure_nonempty": True,
        "min_valid_fraction": 0.10,
        "band_indices": (1, 2, 3, 4),
        "figsize": (12, 6),
        "out_png": None,
        "verbose": True,
        "data_collection": "sentinel-2-l1c",
        "mosaicking_order": "leastCC",
    },
}


_COMPARE_COMMON_BANDS = ("BLUE", "GREEN", "RED", "NIR")

_PHISAT_INDEX_TO_COMMON_ALIAS = {
    1: "BLUE",
    2: "GREEN",
    3: "RED",
    7: "NIR",
}

_SENTINEL_MOSAIC_BAND_INDEX = {
    "BLUE": 1,
    "GREEN": 2,
    "RED": 3,
    "NIR": 4,
}


def _compare_band_alias(band):
    if isinstance(band, str):
        b = band.upper()
        if b == "ALL":
            return "all"
        if b in _SENTINEL_MOSAIC_BAND_INDEX:
            return b
        raise ValueError(
            f"Band {band!r} cannot be compared with the current Sentinel mosaic. "
            f"Use one of {_COMPARE_COMMON_BANDS}, or 'all'."
        )

    if isinstance(band, int):
        if band in _PHISAT_INDEX_TO_COMMON_ALIAS:
            return _PHISAT_INDEX_TO_COMMON_ALIAS[band]
        raise ValueError(
            f"PhiSat-2 band index {band} has no direct equivalent in the current "
            "Sentinel mosaic. Use BLUE/GREEN/RED/NIR."
        )

    raise ValueError(f"Unsupported band selector: {band!r}")


def _resolve_compare_bands(bands):
    if bands is None:
        bands = ("RED", "GREEN", "BLUE")

    if isinstance(bands, str) and bands.upper() == "ALL":
        return "grid", list(_COMPARE_COMMON_BANDS)

    if isinstance(bands, (str, int)):
        return "single", [_compare_band_alias(bands)]

    aliases = [_compare_band_alias(b) for b in list(bands)]

    if len(aliases) == 3:
        return "rgb", aliases

    return "grid", aliases


def _sentinel_indexes_for_aliases(aliases):
    return tuple(_SENTINEL_MOSAIC_BAND_INDEX[a] for a in aliases)


def _cv2_interpolation(name: str):
    import cv2

    name = str(name).lower()
    if name == "nearest":
        return cv2.INTER_NEAREST
    if name == "bilinear":
        return cv2.INTER_LINEAR
    if name == "bicubic":
        return cv2.INTER_CUBIC

    raise ValueError(
        f"Unsupported interpolation={name!r}. "
        "Use 'nearest', 'bilinear', or 'bicubic'."
    )


def _resize_display_image(img: np.ndarray, target_shape_hw, interpolation="bilinear") -> np.ndarray:
    import cv2

    h, w = int(target_shape_hw[0]), int(target_shape_hw[1])
    interp = _cv2_interpolation(interpolation)

    img = np.asarray(img, dtype=np.float32)

    if img.ndim == 2:
        return cv2.resize(img, (w, h), interpolation=interp)

    return cv2.resize(img, (w, h), interpolation=interp)


def _as_rgb_display(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)

    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)

    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)

    return img


def _resolve_compare_config(preset: str = "balanced", overrides: Optional[dict] = None) -> dict:
    if preset not in _COMPARE_PRESETS:
        raise ValueError(f"Unknown preset={preset!r}. Allowed: {list(_COMPARE_PRESETS)}")

    cfg = dict(_COMPARE_PRESETS[preset])
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    return cfg


def _ensure_cache_dirs(base_dir: str | Path):
    base = Path(base_dir)
    mosaics = base / "mosaics"
    mosaics.mkdir(parents=True, exist_ok=True)
    return base, mosaics


def _safe_datetime_token(dt: str) -> str:
    return (
        str(dt)
        .replace("-", "")
        .replace(":", "")
        .replace("T", "_")
        .replace("Z", "")
    )


def _expand_bbox_latlon_km(
    bbox_latlon: Tuple[float, float, float, float],
    margin_km: float,
) -> Tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox_latlon
    center_lat = 0.5 * (min_lat + max_lat)

    dlat = float(margin_km) / 111.32
    dlon = float(margin_km) / max(111.32 * np.cos(np.radians(center_lat)), 1e-6)

    return (
        min_lon - dlon,
        min_lat - dlat,
        max_lon + dlon,
        max_lat + dlat,
    )


def catalog_bbox_latlon(
    obj: Any,
    margin_km: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    Return the minimal lon/lat bbox covering the catalog polygon,
    optionally expanded by a margin in km.
    """
    poly = np.asarray(get_catalog_polygon(obj, order="latlon", closed=False), dtype=np.float64)
    lats = poly[:, 0]
    lons = poly[:, 1]

    bbox = (
        float(np.min(lons)),
        float(np.min(lats)),
        float(np.max(lons)),
        float(np.max(lats)),
    )
    if margin_km > 0:
        bbox = _expand_bbox_latlon_km(bbox, margin_km=margin_km)
    return bbox


def _bbox_center_radius_km(
    bbox_latlon: Tuple[float, float, float, float],
) -> Tuple[Tuple[float, float], float]:
    min_lon, min_lat, max_lon, max_lat = bbox_latlon
    center_lat = 0.5 * (min_lat + max_lat)
    center_lon = 0.5 * (min_lon + max_lon)

    corners = [
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, max_lon),
        (max_lat, min_lon),
    ]

    radius_km = max(
        haversine_m(center_lat, center_lon, lat, lon) for lat, lon in corners
    ) / 1000.0

    return (center_lat, center_lon), float(radius_km)


def _bbox_size_km(
    bbox_latlon: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox_latlon
    mid_lat = 0.5 * (min_lat + max_lat)
    mid_lon = 0.5 * (min_lon + max_lon)

    width_km = haversine_m(mid_lat, min_lon, mid_lat, max_lon) / 1000.0
    height_km = haversine_m(min_lat, mid_lon, max_lat, mid_lon) / 1000.0
    return float(width_km), float(height_km)


def _max_safe_tile_size_km(
    resolution_m: float,
    max_pixels: int = 2500,
    safety_margin_px: int = 100,
) -> float:
    usable_pixels = max(256, int(max_pixels) - int(safety_margin_px))
    return usable_pixels * float(resolution_m) / 1000.0


def _tile_size_for_bbox(
    bbox_latlon: Tuple[float, float, float, float],
    resolution_m: float,
    max_pixels: int = 2500,
    safety_margin_px: int = 100,
) -> float:
    width_km, height_km = _bbox_size_km(bbox_latlon)
    safe_max = _max_safe_tile_size_km(
        resolution_m=resolution_m,
        max_pixels=max_pixels,
        safety_margin_px=safety_margin_px,
    )
    return min(max(width_km, height_km), safe_max)


def _dataset_bounds_to_4326(bounds, crs):
    min_x, min_y, max_x, max_y = map(float, bounds)
    if str(crs).upper().endswith("4326"):
        return (min_x, min_y, max_x, max_y)

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(
        [min_x, max_x, max_x, min_x],
        [min_y, min_y, max_y, max_y],
    )
    return (min(lons), min(lats), max(lons), max(lats))

def _resolve_display_bands(
    bands: str | int | Sequence[int] | None,
    count: int,
) -> list[int]:
    """
    Resolve user-facing Sentinel display bands.

    Rasterio uses 1-based band indexing.
    """
    if bands is None:
        bands = (3, 2, 1)

    if isinstance(bands, str):
        if bands.lower() != "all":
            raise ValueError(
                f"Unsupported bands string: {bands!r}. Use 'all', an int, or a sequence."
            )
        return list(range(1, count + 1))

    if isinstance(bands, int):
        indexes = [int(bands)]
    else:
        indexes = [int(b) for b in bands]

    if not indexes:
        raise ValueError("bands must not be empty.")

    invalid = [b for b in indexes if b < 1 or b > count]
    if invalid:
        raise ValueError(
            f"Invalid band index(es): {invalid}. "
            f"This mosaic has {count} band(s), indexed from 1 to {count}."
        )

    return indexes


def _open_local_mosaic_with_bands(
    mosaic_path: str | Path,
    bands: str | int | Sequence[int] | None = (3, 2, 1),
) -> tuple[SentinelMosaic, list[int]]:
    with rasterio.open(str(mosaic_path)) as ds:
        indexes = _resolve_display_bands(bands, count=ds.count)
        arr = ds.read(indexes=indexes, out_dtype="float32")
        img = np.moveaxis(arr, 0, -1)
        bbox_latlon = _dataset_bounds_to_4326(ds.bounds, ds.crs)

        mosaic = SentinelMosaic(
            image=img.astype(np.float32),
            transform=ds.transform,
            crs=ds.crs,
            bbox_latlon=bbox_latlon,
            acquisition_datetime=None,
            source_path=str(mosaic_path),
        )

    return mosaic, indexes

def open_local_mosaic(
    mosaic_path: str | Path,
    bands: str | int | Sequence[int] | None = (3, 2, 1),
    band_indices: Sequence[int] | None = None,
) -> SentinelMosaic:
    """
    Open a local Sentinel mosaic.

    Args:
        mosaic_path: Path to the Sentinel mosaic GeoTIFF.
        bands: Bands to read. Use one int, a 3-band RGB tuple, a longer sequence, or "all".
        band_indices: Deprecated alias for bands.
    """
    if band_indices is not None:
        bands = band_indices

    mosaic, _ = _open_local_mosaic_with_bands(mosaic_path, bands=bands)
    return mosaic

def _rectified_valid_fraction(
    obj: Any,
    mosaic_path: str | Path,
    band_indices: Sequence[int] = (1, 2, 3, 4),
    flip_horizontal: bool = True,
) -> float:
    """
    Fraction of valid pixels inside the rectified catalog footprint.
    A pixel is considered valid if at least one band is finite and nonzero.
    """
    mosaic = open_local_mosaic(mosaic_path=mosaic_path, band_indices=band_indices)
    rect = extract_rectified_catalog_crop(
        obj=obj,
        mosaic=mosaic,
        flip_horizontal=flip_horizontal,
    )

    raw = np.asarray(rect["rectified_raw"], dtype=np.float32)
    valid = np.isfinite(raw) & (raw != 0)
    valid_px = np.any(valid, axis=2)
    return float(valid_px.mean())



def _phi_rgb_from_event(event) -> np.ndarray:
    """
    Build a display RGB image from a PhiSat-2 event using RED/GREEN/BLUE bands.
    """
    r = event.get_band("RED").astype(np.float32)
    g = event.get_band("GREEN").astype(np.float32)
    b = event.get_band("BLUE").astype(np.float32)
    return np.stack([norm01(r), norm01(g), norm01(b)], axis=-1)

def _prepare_phi_rgb_for_compare(
    l1_event,
    dst_gsd_m: float = 10.0,
    src_gsd_m: float = 4.75,
    match_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Build a PhiSat-2 RGB image and resample it to the requested spatial resolution.

    Optionally resize again to match an exact display shape.
    """
    phi_rgb = _phi_rgb_from_event(l1_event).astype(np.float32)

    # 1) put PhiSat-2 at the same spatial resolution as Sentinel
    phi_rgb = resize_to_gsd(
        phi_rgb,
        src_gsd_m=src_gsd_m,
        dst_gsd_m=dst_gsd_m,
    )

    # 2) optional tiny display adjustment to match the exact output shape
    if match_shape is not None:
        out_h, out_w = int(match_shape[0]), int(match_shape[1])
        if phi_rgb.shape[:2] != (out_h, out_w):
            phi_rgb = cv2.resize(
                phi_rgb.astype(np.float32),
                (out_w, out_h),
                interpolation=cv2.INTER_LINEAR,
            )

    return np.clip(phi_rgb, 0.0, 1.0)



def _order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    Input shape: (4, 2), coordinates in image pixel space (x, y).
    """
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _sentinel_rgb_from_mosaic_image(mosaic_image: np.ndarray) -> np.ndarray:
    """
    Sentinel mosaic image is expected as HxWx4 in order [BLUE, GREEN, RED, NIR].
    Convert to display RGB.
    """
    return norm01(mosaic_image[..., [2, 1, 0]].astype(np.float32))


def _draw_grid(ax, img_shape, grid_steps: int = 6, color="yellow", alpha=0.35, lw=0.8):
    """Draw a regular grid on an axes for visual comparison."""
    h, w = img_shape[:2]

    for i in range(1, grid_steps):
        x = i * w / grid_steps
        y = i * h / grid_steps
        ax.axvline(x=x, color=color, alpha=alpha, linewidth=lw)
        ax.axhline(y=y, color=color, alpha=alpha, linewidth=lw)

def extract_rectified_catalog_crop(
    obj: Any,
    mosaic: SentinelMosaic,
    flip_horizontal: bool = True,
) -> dict:
    """
    Extract the catalog footprint quadrilateral from the Sentinel mosaic
    and warp it into a rectified rectangle.

    Returns both raw 4-band data and display RGB.
    """
    poly_px = catalog_polygon_pixels(obj, mosaic)
    poly_px = np.asarray(poly_px, dtype=np.float32)

    if poly_px.shape[0] == 5:
        poly_px = poly_px[:4]

    src = _order_points_tl_tr_br_bl(poly_px)

    width_top = np.linalg.norm(src[1] - src[0])
    width_bottom = np.linalg.norm(src[2] - src[3])
    height_left = np.linalg.norm(src[3] - src[0])
    height_right = np.linalg.norm(src[2] - src[1])

    out_w = int(round((width_top + width_bottom) / 2.0))
    out_h = int(round((height_left + height_right) / 2.0))

    out_w = max(out_w, 32)
    out_h = max(out_h, 32)

    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(src, dst)

    raw = cv2.warpPerspective(
        mosaic.image.astype(np.float32),
        H,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
    )

    rgb = _sentinel_rgb_from_mosaic_image(raw)

    if flip_horizontal:
        raw = np.fliplr(raw)
        rgb = np.fliplr(rgb)

    return {
        "rectified_raw": raw,
        "rectified_rgb": rgb,
        "src_quad_px": src,
        "output_size": (out_h, out_w),
        "homography": H,
    }

def ensure_nearest_valid_cdse_mosaic_for_catalog(
    obj: Any,
    cache_dir: str | Path = "georef_cache",
    preset: str = "balanced",
    acquisition_datetime: Optional[str] = None,
    force_rebuild: bool = False,
    **overrides,
) -> Path:
    """
    Build or reuse a Sentinel-2 mosaic for the catalog footprint using a
    progressive temporal search schedule and validation inside the footprint
    itself (not the whole mosaic).
    """
    cfg = _resolve_compare_config(preset=preset, overrides=overrides)

    acq_dt = acquisition_datetime or get_catalog_acquisition_datetime(obj)
    if acq_dt is None:
        raise ValueError(
            "Could not infer ΦSat-2 acquisition datetime from catalog metadata. "
            "Pass acquisition_datetime=... explicitly."
        )

    bbox_latlon = catalog_bbox_latlon(obj, margin_km=cfg["margin_km"])

    _, mosaics_dir = _ensure_cache_dirs(cache_dir)

    ident = get_catalog_identifier(obj) or "unknown"
    dt_token = _safe_datetime_token(acq_dt)

    tile_size_km = cfg["tile_size_km"]
    if tile_size_km is None:
        tile_size_km = _tile_size_for_bbox(
            bbox_latlon=bbox_latlon,
            resolution_m=cfg["resolution_m"],
        )

    last_error = None

    for days in cfg["search_schedule_days"]:
        out_tif = mosaics_dir / (
            f"s2_mosaic_{ident}_{dt_token}"
            f"_d{int(days)}_cc{int(cfg['max_cloud_coverage'])}"
            f"_res{int(cfg['resolution_m'])}m.tif"
        )

        if out_tif.exists() and not force_rebuild:
            vf = _rectified_valid_fraction(
                obj=obj,
                mosaic_path=out_tif,
                band_indices=cfg["band_indices"],
                flip_horizontal=cfg["flip_horizontal"],
            )

            if cfg["verbose"]:
                print(
                    f"[PyRawPh] Reusing cached Sentinel mosaic: {out_tif} "
                    f"(rectified valid fraction = {vf:.3f})"
                )

            if (not cfg["ensure_nonempty"]) or (vf >= cfg["min_valid_fraction"]):
                if cfg["verbose"]:
                    print(f"[PyRawPh] Selected Sentinel mosaic: {out_tif}")
                return out_tif

            if cfg["verbose"]:
                print(
                    f"[PyRawPh] Cached mosaic rejected: valid fraction {vf:.3f} "
                    f"< required {cfg['min_valid_fraction']:.3f}"
                )

        if out_tif.exists() and force_rebuild:
            try:
                out_tif.unlink()
            except Exception:
                pass

        if cfg["verbose"]:
            print(
                f"[PyRawPh] Trying Sentinel search window ±{int(days)} day(s), "
                f"cloud<={int(cfg['max_cloud_coverage'])}%"
            )

        try:
            if cfg["verbose"]:
                build_s2_mosaic(
                    phi_date=acq_dt,
                    out_tif=str(out_tif),
                    bbox_latlon=bbox_latlon,
                    tile_size_km=float(tile_size_km),
                    resolution_m=float(cfg["resolution_m"]),
                    time_window_days=int(days),
                    max_cloud_coverage=int(cfg["max_cloud_coverage"]),
                    data_collection=str(cfg["data_collection"]),
                    mosaicking_order=str(cfg["mosaicking_order"]),
                )
            else:
                import contextlib
                import io
                import logging

                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    logging.getLogger("rasterio").setLevel(logging.ERROR)

                    build_s2_mosaic(
                        phi_date=acq_dt,
                        out_tif=str(out_tif),
                        bbox_latlon=bbox_latlon,
                        tile_size_km=float(tile_size_km),
                        resolution_m=float(cfg["resolution_m"]),
                        time_window_days=int(days),
                        max_cloud_coverage=int(cfg["max_cloud_coverage"]),
                        data_collection=str(cfg["data_collection"]),
                        mosaicking_order=str(cfg["mosaicking_order"]),
                    )
        except Exception as exc:
            last_error = exc
            continue

        if not out_tif.exists():
            continue

        vf = _rectified_valid_fraction(
            obj=obj,
            mosaic_path=out_tif,
            band_indices=cfg["band_indices"],
            flip_horizontal=cfg["flip_horizontal"],
        )

        if cfg["verbose"]:
            print(
                f"[PyRawPh] Mosaic rejected: valid fraction {vf:.3f} "
                f"< required {cfg['min_valid_fraction']:.3f}"
            )

        if (not cfg["ensure_nonempty"]) or (vf >= cfg["min_valid_fraction"]):
            if cfg["verbose"]:
                print(f"[PyRawPh] Selected Sentinel mosaic: {out_tif}")
            return out_tif

    raise RuntimeError(
        "Could not build a non-empty Sentinel mosaic inside the catalog footprint. "
        f"Last error: {last_error}"
    )

def _resolve_catalog_mosaic_path(
    obj: Any,
    mosaic_path: str | Path | None = None,
    cache_dir: str | Path = "georef_cache",
    preset: str = "balanced",
    acquisition_datetime: Optional[str] = None,
    force_rebuild: bool = False,
    **overrides,
) -> Path:
    """
    Resolve a Sentinel mosaic path for the given catalog object.

    If `mosaic_path` is provided, reuse it directly.
    Otherwise, build or reuse a valid mosaic through the robust catalog-footprint search.
    """
    if mosaic_path is not None:
        return Path(mosaic_path)

    return ensure_nearest_valid_cdse_mosaic_for_catalog(
        obj=obj,
        cache_dir=cache_dir,
        preset=preset,
        acquisition_datetime=acquisition_datetime,
        force_rebuild=force_rebuild,
        **overrides,
    )

def compare_catalog_rectified(
    obj: Any,
    mosaic_path: str | Path | None = None,
    cache_dir: str | Path = "georef_cache",
    preset: str = "balanced",
    acquisition_datetime: Optional[str] = None,
    force_rebuild: bool = False,
    bands=("RED", "GREEN", "BLUE"),
    registered: bool = False,
    registration_master="RED",
    resize: str = "phisat_to_sentinel",
    interpolation: str = "bilinear",
    grid: bool = True,
    normalize: bool = True,
    normalization: str = "percentile",
    percentiles=(2, 98),
    per_band: bool = True,
    **overrides,
):
    """
    Compare PhiSat-2 with a rectified Sentinel-2 crop using catalog footprint geometry.

    The comparison supports:
    - RGB comparison with bands=("RED", "GREEN", "BLUE")
    - single-band comparison, e.g. bands="NIR"
    - common-band grid comparison with bands="all"

    The current Sentinel mosaic contains only BLUE, GREEN, RED, and NIR.
    """
    cfg = _resolve_compare_config(preset=preset, overrides=overrides)

    mode, aliases = _resolve_compare_bands(bands)

    resolved_mosaic_path = _resolve_catalog_mosaic_path(
        obj=obj,
        mosaic_path=mosaic_path,
        cache_dir=cache_dir,
        preset=preset,
        acquisition_datetime=acquisition_datetime,
        force_rebuild=force_rebuild,
        **cfg,
    )

    if hasattr(obj, "_registered_for_display") and registered:
        phi_obj = obj._registered_for_display(
            master_band=registration_master,
            max_shifts=overrides.get("max_shifts", (80, 80)),
            force=overrides.get("force_registration", False),
        )
    else:
        phi_obj = obj

    def _prepare_pair(pair_aliases):
        s2_indexes = _sentinel_indexes_for_aliases(pair_aliases)

        mosaic = open_local_mosaic(
            mosaic_path=resolved_mosaic_path,
            bands=s2_indexes,
        )

        rect = extract_rectified_catalog_crop(
            obj=obj,
            mosaic=mosaic,
            flip_horizontal=cfg["flip_horizontal"],
        )

        s2_img = _as_rgb_display(rect["rectified_rgb"])

        phi_bands = pair_aliases if len(pair_aliases) == 3 else pair_aliases[0]

        prepared_phi = _prepare_event_display_image(
            phi_obj,
            bands=phi_bands,
            normalize=normalize,
            normalization=normalization,
            percentiles=percentiles,
            per_band=per_band,
        )

        if prepared_phi["mode"] == "gray":
            phi_img = _as_rgb_display(prepared_phi["image"])
        elif prepared_phi["mode"] == "rgb":
            phi_img = _as_rgb_display(prepared_phi["image"])
        else:
            raise ValueError("Internal error: compare pair should be gray or RGB.")

        if resize == "phisat_to_sentinel":
            phi_img = _resize_display_image(
                phi_img,
                target_shape_hw=s2_img.shape[:2],
                interpolation=interpolation,
            )
        elif resize == "sentinel_to_phisat":
            s2_img = _resize_display_image(
                s2_img,
                target_shape_hw=phi_img.shape[:2],
                interpolation=interpolation,
            )
        else:
            raise ValueError(
                f"Unsupported resize={resize!r}. "
                "Use 'phisat_to_sentinel' or 'sentinel_to_phisat'."
            )

        return {
            "mosaic": mosaic,
            "rect": rect,
            "phi_img": phi_img,
            "s2_img": s2_img,
            "aliases": pair_aliases,
            "prepared_phi": prepared_phi,
        }

    # RGB or single-band comparison
    if mode in ("rgb", "single"):
        pair = _prepare_pair(aliases)

        fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])

        axes[0].imshow(pair["phi_img"])
        axes[0].set_title(
            f"PhiSat-2 {tuple(aliases) if mode == 'rgb' else aliases[0]} "
            f"— {'registered' if registered else 'raw'}"
        )
        if grid:
            _draw_grid(axes[0], pair["phi_img"].shape, grid_steps=cfg["grid_steps"])
        axes[0].axis("off")

        axes[1].imshow(pair["s2_img"])
        axes[1].set_title(
            f"Sentinel-2 rectified crop — max cloud ≤ {cfg['max_cloud_coverage']}%"
        )
        if grid:
            _draw_grid(axes[1], pair["s2_img"].shape, grid_steps=cfg["grid_steps"])
        axes[1].axis("off")

        plt.suptitle(
            f"PhiSat-2 vs Sentinel-2 — {mode} — resize={resize}, interp={interpolation}"
        )
        plt.tight_layout()

        if cfg["out_png"] is not None:
            plt.savefig(cfg["out_png"], dpi=180, bbox_inches="tight")

        plt.show()

        return {
            "config": cfg,
            "mode": mode,
            "bands": aliases,
            "registered": registered,
            "resize": resize,
            "interpolation": interpolation,
            "mosaic_path": str(resolved_mosaic_path),
            "mosaic": pair["mosaic"],
            "phi_img": pair["phi_img"],
            "sentinel_rectified_img": pair["s2_img"],
            "src_quad_px": pair["rect"]["src_quad_px"],
            "output_size": pair["rect"]["output_size"],
            "homography": pair["rect"]["homography"],
        }

    # Grid comparison: one row per common band
    pairs = [_prepare_pair([alias]) for alias in aliases]

    fig, axes = plt.subplots(
        len(pairs),
        2,
        figsize=(cfg["figsize"][0], max(3 * len(pairs), cfg["figsize"][1])),
        squeeze=False,
    )

    for i, pair in enumerate(pairs):
        alias = pair["aliases"][0]

        axes[i, 0].imshow(pair["phi_img"])
        axes[i, 0].set_title(f"PhiSat-2 {alias} — {'registered' if registered else 'raw'}")
        if grid:
            _draw_grid(axes[i, 0], pair["phi_img"].shape, grid_steps=cfg["grid_steps"])
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pair["s2_img"])
        axes[i, 1].set_title(f"Sentinel-2 {alias} rectified")
        if grid:
            _draw_grid(axes[i, 1], pair["s2_img"].shape, grid_steps=cfg["grid_steps"])
        axes[i, 1].axis("off")

    plt.suptitle(
        f"PhiSat-2 vs Sentinel-2 common bands — resize={resize}, interp={interpolation}"
    )
    plt.tight_layout()

    if cfg["out_png"] is not None:
        plt.savefig(cfg["out_png"], dpi=180, bbox_inches="tight")

    plt.show()

    return {
        "config": cfg,
        "mode": "grid",
        "bands": aliases,
        "registered": registered,
        "resize": resize,
        "interpolation": interpolation,
        "mosaic_path": str(resolved_mosaic_path),
        "pairs": pairs,
    }






def _latlon_to_dataset_xy(lat: float, lon: float, crs) -> Tuple[float, float]:
    if str(crs).upper().endswith("4326"):
        return float(lon), float(lat)

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def catalog_polygon_pixels(
    obj: Any,
    mosaic: SentinelMosaic,
) -> np.ndarray:
    """
    Convert the catalog polygon to pixel coordinates in the fetched Sentinel crop.

    Returns:
        array of shape (N, 2) with columns [x, y] in crop pixel coordinates.
    """
    poly_latlon = get_catalog_polygon(obj, order="latlon", closed=True)

    xs = []
    ys = []
    for lat, lon in poly_latlon:
        x, y = _latlon_to_dataset_xy(lat, lon, mosaic.crs)
        xs.append(x)
        ys.append(y)

    rows, cols = rowcol(mosaic.transform, xs, ys)
    cols = np.asarray(cols, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.float32)
    return np.stack([cols, rows], axis=1)


def catalog_center_pixel(
    obj: Any,
    mosaic: SentinelMosaic,
) -> Tuple[float, float]:
    """
    Return the catalog center as (x, y) pixel coordinates in the Sentinel crop.
    """
    lat, lon = get_catalog_center(obj, order="latlon")
    x, y = _latlon_to_dataset_xy(lat, lon, mosaic.crs)
    row, col = rowcol(mosaic.transform, x, y)
    return float(col), float(row)


def sentinel_rgb_from_mosaic(mosaic: SentinelMosaic) -> np.ndarray:
    """
    Build a display RGB image from a Sentinel mosaic.

    Rules:
    - 1 band: grayscale repeated into RGB
    - 3 bands: direct RGB in the order selected by the user
    """
    img = np.asarray(mosaic.image, dtype=np.float32)

    if img.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {img.shape}")

    if img.shape[2] == 1:
        gray = norm01(img[..., 0])
        return np.repeat(gray[..., None], 3, axis=2)

    if img.shape[2] == 3:
        return norm01(img)

    raise ValueError(
        "sentinel_rgb_from_mosaic only supports 1 or 3 selected bands. "
        "Use show_catalog_geo_on_sentinel(..., bands='all') to display bands separately."
    )

def show_catalog_geo_on_sentinel(
    obj: Any,
    mosaic_path: str | Path,
    bands: str | int | Sequence[int] | None = (3, 2, 1),
    figsize=(8, 8),
    title: Optional[str] = None,
    out_png: str | Path | None = None,
):
    """
    Display the catalog footprint on an existing local Sentinel mosaic.

    Args:
        obj: PhiSat-2 event, metadata dict, or raw catalog_geo dict.
        mosaic_path: Path to the Sentinel mosaic GeoTIFF.
        bands:
            - int: display one band in grayscale
            - tuple/list of length 3: display RGB in the given order
            - "all": display all bands separately
            - tuple/list of any other length: display selected bands separately
    """
    mosaic, used_bands = _open_local_mosaic_with_bands(
        mosaic_path=mosaic_path,
        bands=bands,
    )

    img = np.asarray(mosaic.image, dtype=np.float32)
    poly_px = catalog_polygon_pixels(obj, mosaic)
    cx, cy = catalog_center_pixel(obj, mosaic)

    def _draw_overlay(ax):
        ax.plot(poly_px[:, 0], poly_px[:, 1], linewidth=2)
        ax.scatter(poly_px[:, 0], poly_px[:, 1], s=35)
        ax.scatter([cx], [cy], s=60, marker="x")
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.axis("off")

    # Case 1: one band -> grayscale
    if img.shape[2] == 1:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(norm01(img[..., 0]), cmap="gray")
        _draw_overlay(ax)
        ax.set_title(title or f"Catalog footprint on Sentinel-2 - band {used_bands[0]}")
        plt.tight_layout()

        if out_png is not None:
            plt.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.show()

        return {
            "mosaic": mosaic,
            "display": norm01(img[..., 0]),
            "polygon_px": poly_px,
            "center_px": (cx, cy),
            "bands": used_bands,
            "mode": "gray",
        }

    # Case 2: exactly three bands -> RGB in given order
    if img.shape[2] == 3:
        rgb = norm01(img)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb)
        _draw_overlay(ax)
        ax.set_title(title or f"Catalog footprint on Sentinel-2 - RGB {tuple(used_bands)}")
        plt.tight_layout()

        if out_png is not None:
            plt.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.show()

        return {
            "mosaic": mosaic,
            "rgb": rgb,
            "polygon_px": poly_px,
            "center_px": (cx, cy),
            "bands": used_bands,
            "mode": "rgb",
        }

    # Case 3: all / several bands -> grid of individual bands
    n = img.shape[2]
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize[0] * ncols / 2.0, figsize[1] * nrows / 2.0),
        squeeze=False,
    )

    for i in range(nrows * ncols):
        ax = axes.flat[i]
        if i >= n:
            ax.axis("off")
            continue

        ax.imshow(norm01(img[..., i]), cmap="gray")
        _draw_overlay(ax)
        ax.set_title(f"Band {used_bands[i]}")

    plt.suptitle(title or "Catalog footprint on Sentinel-2 bands")
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.show()

    return {
        "mosaic": mosaic,
        "bands_images": img,
        "polygon_px": poly_px,
        "center_px": (cx, cy),
        "bands": used_bands,
        "mode": "grid",
    }

def ensure_cdse_mosaic_for_catalog(
    obj: Any,
    cache_dir: str | Path = "georef_cache",
    acquisition_datetime: Optional[str] = None,
    radius_km: Optional[float] = None,
    margin_km: float = 2.0,
    tile_size_km: Optional[float] = None,
    resolution_m: float = 10.0,
    time_window_days: int = 7,
    max_cloud_coverage: int = 20,
    force_rebuild: bool = False,
    verbose: bool = True,
    preset: str = "balanced",
) -> Path:
    """
    Backward-compatible wrapper around ensure_nearest_valid_cdse_mosaic_for_catalog().

    New code should prefer ensure_nearest_valid_cdse_mosaic_for_catalog() directly.
    """
    overrides = {
        "margin_km": margin_km,
        "resolution_m": resolution_m,
        "max_cloud_coverage": max_cloud_coverage,
        "verbose": verbose,
    }

    if tile_size_km is not None:
        overrides["tile_size_km"] = tile_size_km

    if time_window_days is not None:
        overrides["search_schedule_days"] = tuple(sorted({0, int(time_window_days)}))

    return ensure_nearest_valid_cdse_mosaic_for_catalog(
        obj=obj,
        cache_dir=cache_dir,
        preset=preset,
        acquisition_datetime=acquisition_datetime,
        force_rebuild=force_rebuild,
        **overrides,
    )


def show_catalog_geo_in_sentinel(
    obj: Any,
    mosaic_path: str | Path | None = None,
    bands: str | int | Sequence[int] | None = (3, 2, 1),
    cache_dir: str | Path = "georef_cache",
    preset: str = "balanced",
    acquisition_datetime: Optional[str] = None,
    margin_km: float = 1.0,
    resolution_m: float = 10.0,
    max_cloud_coverage: int = 20,
    figsize=(8, 8),
    title: Optional[str] = None,
    out_png: str | Path | None = None,
    force_rebuild: bool = False,
    verbose: bool = True,
):
    """
    Build/reuse a Sentinel-2 mosaic and display the catalog footprint on it.
    """
    resolved_mosaic_path = _resolve_catalog_mosaic_path(
        obj=obj,
        mosaic_path=mosaic_path,
        cache_dir=cache_dir,
        preset=preset,
        acquisition_datetime=acquisition_datetime,
        force_rebuild=force_rebuild,
        margin_km=margin_km,
        resolution_m=resolution_m,
        max_cloud_coverage=max_cloud_coverage,
        verbose=verbose,
    )

    return show_catalog_geo_on_sentinel(
        obj=obj,
        mosaic_path=resolved_mosaic_path,
        bands=bands,
        figsize=figsize,
        title=title,
        out_png=out_png,
    )


def show_coordinates_in_sentinel(*args, **kwargs):
    return show_catalog_geo_in_sentinel(*args, **kwargs)




