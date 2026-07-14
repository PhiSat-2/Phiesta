from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import cv2


def _norm01(x: np.ndarray, mask: np.ndarray | None = None, percentiles=(2, 98)) -> np.ndarray:
    x = x.astype("float32")
    if mask is None:
        mask = np.isfinite(x)
    vals = x[mask]
    if vals.size < 100:
        return np.zeros_like(x, dtype="float32")
    lo, hi = np.percentile(vals, percentiles)
    if hi <= lo:
        return np.zeros_like(x, dtype="float32")
    return np.clip((x - lo) / (hi - lo), 0, 1).astype("float32")


def _pearson(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    x = a[mask].astype("float64")
    y = b[mask].astype("float64")
    if x.size < 100:
        return None
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    if den == 0:
        return None
    return float((x * y).sum() / den)


def _global_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    x = a[mask].astype("float64")
    y = b[mask].astype("float64")
    if x.size < 100:
        return None

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mux, muy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mux) * (y - muy)).mean()

    return float(
        ((2 * mux * muy + c1) * (2 * cov + c2))
        / ((mux * mux + muy * muy + c1) * (vx + vy + c2))
    )


def _edge_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    ax = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    bx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)

    ea = np.sqrt(ax * ax + ay * ay)
    eb = np.sqrt(bx * bx + by * by)

    return _pearson(ea, eb, mask)


def _phase_shift(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(a.astype("float32"), b.astype("float32"), win)
    dx, dy = float(shift[0]), float(shift[1])
    mag = float((dx * dx + dy * dy) ** 0.5)
    return {
        "dx_px": dx,
        "dy_px": dy,
        "shift_mag_px": mag,
        "response": float(response),
    }


def _desc_map(descriptions: tuple[Any, ...]) -> dict[str, int]:
    out: dict[str, int] = {}

    for i, d in enumerate(descriptions):
        s = (d or "").upper()

        if "PAN" in s:
            out["PAN"] = i
        elif "BLUE" in s or "B02" in s:
            out["BLUE"] = i
        elif "GREEN" in s or "B03" in s:
            out["GREEN"] = i
        elif s == "RED" or "B04" in s:
            out["RED"] = i
        elif s == "NIR" or "B08" in s:
            out["NIR"] = i
        elif "RED_EDGE_1" in s or "B05" in s or "RE1" in s:
            out["RE1"] = i
        elif "RED_EDGE_2" in s or "B06" in s or "RE2" in s:
            out["RE2"] = i
        elif "RED_EDGE_3" in s or "B07" in s or "RE3" in s:
            out["RE3"] = i

    return out


def _read_downsampled(path: str | Path, band_idx0: int, step: int) -> np.ndarray:
    with rasterio.open(path) as src:
        x = src.read(band_idx0 + 1).astype("float32")
    return x[::step, ::step]


def evaluate_real_sim_alignment(
    real_path: str | Path,
    simulated_path: str | Path,
    *,
    downsample: int = 4,
    percentiles=(2, 98),
) -> dict[str, Any]:
    """
    Compute simple scene-level real-vs-simulated alignment diagnostics.

    These metrics are diagnostic only. They combine alignment quality,
    radiometric differences, clouds, temporal changes and scene content.
    """
    real_path = Path(real_path)
    simulated_path = Path(simulated_path)

    with rasterio.open(real_path) as rsrc, rasterio.open(simulated_path) as ssrc:
        rmap = _desc_map(tuple(rsrc.descriptions))
        smap = _desc_map(tuple(ssrc.descriptions))

    bands = ["PAN", "BLUE", "GREEN", "RED", "NIR", "RE1", "RE2", "RE3"]
    out: dict[str, Any] = {
        "real_path": str(real_path),
        "simulated_path": str(simulated_path),
        "downsample": int(downsample),
        "percentiles": list(percentiles),
        "band_metrics": {},
    }

    for band in bands:
        if band not in rmap or band not in smap:
            continue

        r = _read_downsampled(real_path, rmap[band], downsample)
        s = _read_downsampled(simulated_path, smap[band], downsample)

        mask = np.isfinite(r) & np.isfinite(s) & (r != 0) & (s != 0)

        rn = _norm01(r, mask, percentiles=percentiles)
        sn = _norm01(s, mask, percentiles=percentiles)

        bm = {
            "corr": _pearson(rn, sn, mask),
            "ssim_global": _global_ssim(rn, sn, mask),
            "edge_corr": _edge_corr(rn, sn, mask),
            "valid_fraction": float(mask.mean()),
        }

        if band == "PAN":
            ph = _phase_shift(rn, sn)
            bm["phase_dx_px_downsampled"] = ph["dx_px"]
            bm["phase_dy_px_downsampled"] = ph["dy_px"]
            bm["phase_shift_mag_px_downsampled"] = ph["shift_mag_px"]
            bm["phase_response"] = ph["response"]
            bm["phase_shift_mag_px_fullres_approx"] = ph["shift_mag_px"] * float(downsample)

        out["band_metrics"][band] = bm

    mb = []
    mb_ssim = []

    for band in ["BLUE", "GREEN", "RED", "NIR", "RE1", "RE2", "RE3"]:
        bm = out["band_metrics"].get(band)
        if not bm:
            continue
        if bm.get("corr") is not None:
            mb.append(float(bm["corr"]))
        if bm.get("ssim_global") is not None:
            mb_ssim.append(float(bm["ssim_global"]))

    out["summary"] = {
        "PAN_corr": out["band_metrics"].get("PAN", {}).get("corr"),
        "PAN_ssim_global": out["band_metrics"].get("PAN", {}).get("ssim_global"),
        "PAN_edge_corr": out["band_metrics"].get("PAN", {}).get("edge_corr"),
        "PAN_phase_shift_mag_px_fullres_approx": out["band_metrics"].get("PAN", {}).get("phase_shift_mag_px_fullres_approx"),
        "PAN_phase_response": out["band_metrics"].get("PAN", {}).get("phase_response"),
        "NIR_corr": out["band_metrics"].get("NIR", {}).get("corr"),
        "NIR_ssim_global": out["band_metrics"].get("NIR", {}).get("ssim_global"),
        "MB_corr_mean": float(np.mean(mb)) if mb else None,
        "MB_ssim_global_mean": float(np.mean(mb_ssim)) if mb_ssim else None,
    }

    return out
