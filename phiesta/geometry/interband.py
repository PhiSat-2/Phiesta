from __future__ import annotations

from typing import Any, Iterable

import re
import numpy as np
import pandas as pd

from phiesta.utils.l0_l1_registration import (
    prep_for_phase_corr,
    phase_correlation_shift,
)


def _event_folder(event: Any) -> str:
    for attr in ("product_folder", "root", "path"):
        if hasattr(event, attr):
            v = getattr(event, attr)
            try:
                v = v() if callable(v) else v
            except TypeError:
                pass
            if v:
                return str(v)
    return ""


def _product_level(event: Any) -> str | None:
    folder = _event_folder(event)

    if "PHISAT-2_L1A_" in folder:
        return "L1A"
    if "PHISAT-2_L1_" in folder:
        return "L1C"
    if "PHISAT-2_L0_" in folder:
        return "L0"

    return None


def _infer_product_id(event: Any) -> str | None:
    folder = _event_folder(event)
    m = re.search(r"_(\d{9})_", folder)
    if m:
        return str(int(m.group(1)))

    for attr in ("meta", "get_meta"):
        if hasattr(event, attr):
            v = getattr(event, attr)
            try:
                meta = v() if callable(v) else dict(v)
            except Exception:
                meta = {}
            for key in ("product_id", "id", "identifier", "filename"):
                if key in meta:
                    m = re.search(r"(\d{4,9})", str(meta[key]))
                    if m:
                        return str(int(m.group(1)))

    return None


def _downsample(x: np.ndarray, max_side: int = 1024) -> tuple[np.ndarray, float]:
    h, w = x.shape
    scale = max(h, w) / max_side
    if scale <= 1:
        return x, 1.0

    step = max(1, int(round(scale)))
    return x[::step, ::step], float(step)


def _get_band(event: Any, band: Any) -> np.ndarray:
    if hasattr(event, "get_band"):
        return event.get_band(band)
    raise TypeError("Expected an event-like object with get_band(...).")


def _parse_shift_result(result: Any) -> tuple[float, float, float | None]:
    """
    Parse existing Phiesta phase-correlation return formats.

    Supported:
    - dict with dx/dy/response-like keys;
    - tuple/list like (shift, response), where shift=(dy, dx);
    - tuple/list like (dy, dx);
    - tensor/array with at least two values.
    """
    if isinstance(result, dict):
        dx = result.get("dx", result.get("shift_x", result.get("x", np.nan)))
        dy = result.get("dy", result.get("shift_y", result.get("y", np.nan)))
        response = result.get("response", result.get("peak", result.get("score", None)))
        return float(dx), float(dy), None if response is None else float(response)

    if isinstance(result, (tuple, list)):
        if len(result) == 2:
            a, b = result

            # Common convention: (shift, response), shift=(dy, dx)
            if hasattr(a, "detach"):
                a = a.detach().cpu().numpy()
            if hasattr(b, "detach"):
                b = b.detach().cpu().numpy()

            if isinstance(a, (tuple, list, np.ndarray)):
                arr = np.asarray(a).ravel()
                if arr.size >= 2:
                    dy, dx = float(arr[0]), float(arr[1])
                    resp_arr = np.asarray(b).ravel()
                    response = float(resp_arr[0]) if resp_arr.size else None
                    return dx, dy, response

            # Convention: (dy, dx)
            return float(a), float(b), None

        if len(result) >= 3:
            return float(result[0]), float(result[1]), float(result[2])

    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()

    arr = np.asarray(result).ravel()
    if arr.size >= 2:
        # Convention: [dy, dx]
        return float(arr[1]), float(arr[0]), None

    raise ValueError(f"Cannot parse phase-correlation result: {result!r}")


def _phase_shift_2d(target: np.ndarray, master: np.ndarray):
    """
    Use the existing Phiesta phase-correlation helper with proper preprocessing.
    """
    target_t = prep_for_phase_corr(target.astype(np.float32))
    master_t = prep_for_phase_corr(master.astype(np.float32))
    return phase_correlation_shift(target_t, master_t)


def interband_shift_table(
    event: Any,
    *,
    master_band: Any = 2,
    target_bands: Iterable[Any] | str = "all",
    max_side: int = 1024,
) -> pd.DataFrame:
    """
    Estimate global inter-band translations using the existing Phiesta
    phase-correlation utility.

    This is a fast diagnostic wrapper, not a certified geometric calibration product.
    """
    master_full = _get_band(event, master_band).astype(np.float32)
    master, scale = _downsample(master_full, max_side=max_side)

    if target_bands == "all":
        if hasattr(event, "to_cube"):
            cube = event.to_cube(bands="all", band_axis=0, copy=False)
            n_bands = cube.shape[0]
        elif hasattr(event, "as_numpy"):
            arr = event.as_numpy()
            n_bands = arr.shape[0] if arr.ndim == 3 else 1
        else:
            n_bands = 8

        target_bands = list(range(n_bands))

    rows = []
    for band in target_bands:
        if band == master_band:
            continue

        try:
            target_full = _get_band(event, band).astype(np.float32)
            target, _ = _downsample(target_full, max_side=max_side)

            result = _phase_shift_2d(target, master)
            dx_ds, dy_ds, response = _parse_shift_result(result)

            dx = dx_ds * scale
            dy = dy_ds * scale
            shift_px = float((dx ** 2 + dy ** 2) ** 0.5)

            status = "ok"
            error = None

        except Exception as e:
            dx = dy = shift_px = np.nan
            dx_ds = dy_ds = np.nan
            response = np.nan
            status = "failed"
            error = f"{type(e).__name__}: {e}"

        rows.append({
            "product_id": _infer_product_id(event),
            "level": _product_level(event),
            "master_band": master_band,
            "target_band": band,
            "dx_px": dx,
            "dy_px": dy,
            "shift_px": shift_px,
            "dx_px_downsampled": dx_ds,
            "dy_px_downsampled": dy_ds,
            "scale": scale,
            "response": response,
            "max_side": max_side,
            "status": status,
            "error": error,
        })

    return pd.DataFrame(rows)
