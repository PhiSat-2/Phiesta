from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

from .anatomy import _as_path, product_card


def _preferred_display_raster(root: Path) -> Path | None:
    candidates = [
        root / "bands" / "scene_0_BC_multiband.tiff",
        root / "bands" / "scene_0_RC_multiband.tiff",
        root / "bands" / "scene_0_BC_RGB.tiff",
        root / "bands" / "scene_0_RC_RGB.tiff",
    ]

    for p in candidates:
        if p.exists():
            return p

    bands_dir = root / "bands"
    rasters = sorted(bands_dir.glob("*.tif*")) if bands_dir.exists() else []
    return rasters[0] if rasters else None


def _robust01(x: np.ndarray, percentiles=(2, 98)) -> np.ndarray:
    x = x.astype(np.float32)
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.zeros_like(x, dtype=np.float32)

    vals = x[mask]
    lo, hi = np.percentile(vals, percentiles)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _select_display_bands(arr: np.ndarray) -> np.ndarray:
    """
    arr shape: C, H, W.

    For 8-band PhiSat-2 stacks, use display RGB from zero-based bands [3, 2, 1],
    corresponding to rasterio read bands [4, 3, 2].
    """
    c = arr.shape[0]

    if c >= 4:
        idx = [3, 2, 1]
    elif c >= 3:
        idx = [2, 1, 0]
    elif c == 2:
        idx = [1, 0, 0]
    else:
        idx = [0, 0, 0]

    return arr[idx]


def _read_display_sample(path: Path, max_side: int = 1024) -> tuple[np.ndarray, dict[str, Any]]:
    if rasterio is None:
        raise ImportError("rasterio is required for quality_report().")

    with rasterio.open(path) as src:
        scale = max(src.width, src.height) / max_side
        if scale < 1:
            out_h, out_w = src.height, src.width
        else:
            out_h = max(1, int(round(src.height / scale)))
            out_w = max(1, int(round(src.width / scale)))

        indexes = list(range(1, min(src.count, 8) + 1))
        arr = src.read(indexes, out_shape=(len(indexes), out_h, out_w)).astype(np.float32)

        meta = {
            "raster": str(path),
            "width": src.width,
            "height": src.height,
            "sample_width": out_w,
            "sample_height": out_h,
            "count": src.count,
            "dtype": src.dtypes[0] if src.dtypes else None,
            "crs": str(src.crs) if src.crs else "",
            "nodata": src.nodata,
        }

    rgb_raw = _select_display_bands(arr)
    rgb = np.stack([_robust01(ch) for ch in rgb_raw], axis=-1)
    return rgb, meta


def _edge_density(gray: np.ndarray, threshold: float = 0.08) -> float:
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0

    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)

    g = np.zeros_like(gray, dtype=np.float32)
    g[:-1, :] += np.abs(gy)
    g[:, :-1] += np.abs(gx)

    return float(np.mean(g > threshold))


def quality_report(product: Any, *, max_side: int = 1024) -> dict[str, Any]:
    """
    Heuristic visual/product screening report.

    This is not a certified physical product-quality metric. It is designed
    to help users quickly decide whether a PhiSat-2 product is visually usable
    for inspection, galleries, patch extraction, alignment, or diagnostics.
    """
    root = _as_path(product)
    card = product_card(root)
    raster = _preferred_display_raster(root)

    report: dict[str, Any] = {
        "product_id": card.get("product_id"),
        "level": card.get("level"),
        "folder": str(root),
        "has_bands": bool(card.get("has_bands")),
        "has_geolocation": bool(card.get("has_geolocation")),
        "crs_values": card.get("crs_values", []),
        "n_files": card.get("n_files"),
        "n_rasters": card.get("n_rasters"),
        "total_mb": card.get("total_mb"),
        "screening_note": (
            "Heuristic visual screening report; not a certified physical product-quality metric."
        ),
    }

    flags: list[str] = []

    if not card.get("has_bands"):
        flags.append("missing_bands")
    if card.get("level") in {"L1", "L1C"} and not card.get("has_geolocation"):
        flags.append("missing_geolocation_for_l1c")
    if card.get("level") in {"L1", "L1C"} and not card.get("crs_values"):
        flags.append("missing_crs_for_l1c")
    if raster is None:
        flags.append("missing_display_raster")
        report.update({
            "raster": None,
            "score": 0.0,
            "recommendation": "inspect_manually",
            "flags": flags,
        })
        return report

    try:
        rgb, raster_meta = _read_display_sample(raster, max_side=max_side)
    except Exception as e:
        flags.append("raster_read_failed")
        report.update({
            "raster": str(raster),
            "score": 0.0,
            "recommendation": "inspect_manually",
            "flags": flags,
            "error": f"{type(e).__name__}: {e}",
        })
        return report

    gray = np.nanmean(rgb, axis=-1)

    finite_fraction = float(np.mean(np.isfinite(rgb)))
    dark_fraction = float(np.mean(gray < 0.05))
    bright_fraction = float(np.mean(gray > 0.95))
    texture_score = float(np.nanstd(gray))
    edge_density = _edge_density(gray)

    texture_component = min(texture_score / 0.18, 1.0)
    edge_component = min(edge_density / 0.12, 1.0)
    dark_component = 1.0 - min(dark_fraction / 0.70, 1.0)
    bright_component = 1.0 - min(bright_fraction / 0.70, 1.0)

    score = (
        0.35 * texture_component
        + 0.35 * edge_component
        + 0.15 * dark_component
        + 0.15 * bright_component
    )

    # Conservative display-screening flags.
    # These are not physical cloud/quality labels; they indicate whether the
    # product is a good visual/diagnostic candidate.
    if finite_fraction < 0.98:
        flags.append("many_non_finite_pixels")
        score *= 0.5
    if dark_fraction > 0.25:
        flags.append("high_display_dark_fraction")
    if bright_fraction > 0.20:
        flags.append("high_display_bright_fraction")
    if texture_score < 0.16:
        flags.append("low_display_texture")
    if edge_density < 0.12:
        flags.append("low_display_edge_density")

    # Hard caps: weak visual structure should not be called a clean candidate.
    if dark_fraction > 0.25 or bright_fraction > 0.20:
        score = min(score, 0.65)
    if edge_density < 0.12:
        score = min(score, 0.65)
    if texture_score < 0.16:
        score = min(score, 0.70)
    if "missing_bands" in flags or "missing_display_raster" in flags:
        score = min(score, 0.30)
    if "missing_geolocation_for_l1c" in flags or "missing_crs_for_l1c" in flags:
        score = min(score, 0.45)

    if score >= 0.75 and not flags:
        recommendation = "good_display_candidate"
    elif score >= 0.50:
        recommendation = "usable_with_caution"
    else:
        recommendation = "inspect_manually"

    report.update({
        **raster_meta,
        "finite_fraction": round(finite_fraction, 4),
        "dark_fraction": round(dark_fraction, 4),
        "bright_fraction": round(bright_fraction, 4),
        "texture_score": round(texture_score, 4),
        "edge_density": round(edge_density, 4),
        "score": round(float(score), 4),
        "recommendation": recommendation,
        "flags": flags,
    })

    return report


def quality_table(products: Iterable[Any], *, max_side: int = 1024) -> pd.DataFrame:
    rows = [quality_report(p, max_side=max_side) for p in products]
    return pd.DataFrame(rows)
