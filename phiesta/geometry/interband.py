from __future__ import annotations

from typing import Any, Iterable

import re
import numpy as np
import pandas as pd

from phiesta.utils.l0_l1_registration import (
    prep_for_phase_corr,
    phase_correlation_shift,
    warp_np_by_shift,
)


def _event_folder(event: Any) -> str:
    for attr in ("product_folder", "root", "path"):
        if hasattr(event, attr):
            value = getattr(event, attr)
            try:
                value = value() if callable(value) else value
            except TypeError:
                pass
            if value:
                return str(value)
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
    match = re.search(r"_(\d{9})_", folder)
    if match:
        return str(int(match.group(1)))

    for attr in ("meta", "get_meta"):
        if not hasattr(event, attr):
            continue
        value = getattr(event, attr)
        try:
            meta = value() if callable(value) else dict(value)
        except Exception:
            meta = {}
        for key in ("product_id", "id", "identifier", "filename"):
            if key in meta:
                match = re.search(r"(\d{4,9})", str(meta[key]))
                if match:
                    return str(int(match.group(1)))
    return None


def _get_band(event: Any, band: Any) -> np.ndarray:
    if not hasattr(event, "get_band"):
        raise TypeError("Expected an event-like object with get_band(...).")
    return np.asarray(event.get_band(band))


def _downsample(
    image: np.ndarray,
    max_side: int = 1024,
) -> tuple[np.ndarray, float]:
    h, w = image.shape
    scale = max(h, w) / float(max_side)
    if scale <= 1:
        return image, 1.0
    step = max(1, int(round(scale)))
    return image[::step, ::step], float(step)


def _parse_shift_result(result: Any) -> tuple[float, float, float | None]:
    """Return ``(dx, dy, response)`` from supported phase-correlation outputs."""
    if isinstance(result, dict):
        dx = result.get("dx", result.get("shift_x", result.get("x", np.nan)))
        dy = result.get("dy", result.get("shift_y", result.get("y", np.nan)))
        response = result.get(
            "response",
            result.get("peak", result.get("score", None)),
        )
        return float(dx), float(dy), None if response is None else float(response)

    if isinstance(result, (tuple, list)):
        if len(result) == 2:
            first, second = result
            if hasattr(first, "detach"):
                first = first.detach().cpu().numpy()
            if hasattr(second, "detach"):
                second = second.detach().cpu().numpy()

            if isinstance(first, (tuple, list, np.ndarray)):
                arr = np.asarray(first).ravel()
                if arr.size >= 2:
                    dy, dx = float(arr[0]), float(arr[1])
                    response_arr = np.asarray(second).ravel()
                    response = float(response_arr[0]) if response_arr.size else None
                    return dx, dy, response

            return float(first), float(second), None

        if len(result) >= 3:
            return float(result[0]), float(result[1]), float(result[2])

    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()

    arr = np.asarray(result).ravel()
    if arr.size >= 2:
        dy, dx = float(arr[0]), float(arr[1])
        return dx, dy, None

    raise ValueError(f"Cannot parse phase-correlation result: {result!r}")


def _phase_shift_target_to_master(
    target: np.ndarray,
    master: np.ndarray,
    *,
    max_shifts=None,
):
    """
    Estimate the shift to apply to ``target`` so it aligns with ``master``.

    The sign convention is deliberately explicit and is shared by the global
    and local geometry diagnostics.
    """
    master_t = prep_for_phase_corr(master.astype(np.float32))
    target_t = prep_for_phase_corr(target.astype(np.float32))
    return phase_correlation_shift(
        master_t,
        target_t,
        max_shifts=max_shifts,
    )


def _finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _common_shape(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def interband_shift_table(
    event: Any,
    *,
    master_band: Any = 2,
    target_bands: Iterable[Any] | str = "all",
    max_side: int = 1024,
    max_shifts=None,
) -> pd.DataFrame:
    """
    Estimate global inter-band translations.

    ``dx_px`` and ``dy_px`` are the shifts to apply to each target band so it
    aligns with ``master_band``. Positive ``dx`` moves the target right;
    positive ``dy`` moves it down.

    This is a diagnostic measurement, not a certified geometric calibration.
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
            target, target_scale = _downsample(
                target_full,
                max_side=max_side,
            )
            master_use, target_use = _common_shape(master, target)

            result = _phase_shift_target_to_master(
                target_use,
                master_use,
                max_shifts=max_shifts,
            )
            dx_ds, dy_ds, response = _parse_shift_result(result)

            effective_scale = float(max(scale, target_scale))
            dx = dx_ds * effective_scale
            dy = dy_ds * effective_scale
            shift_px = float(np.hypot(dx, dy))

            corr_before = _finite_corr(master_use, target_use)
            aligned = warp_np_by_shift(
                target_use,
                np.array([dy_ds, dx_ds], dtype=np.float32),
            )
            corr_after = _finite_corr(master_use, aligned)

            status = "ok"
            error = None

        except Exception as exc:
            dx = dy = shift_px = np.nan
            dx_ds = dy_ds = np.nan
            response = np.nan
            corr_before = corr_after = np.nan
            effective_scale = np.nan
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "product_id": _infer_product_id(event),
                "level": _product_level(event),
                "master_band": master_band,
                "target_band": band,
                "dx_px": dx,
                "dy_px": dy,
                "shift_px": shift_px,
                "dx_px_downsampled": dx_ds,
                "dy_px_downsampled": dy_ds,
                "scale": effective_scale,
                "response": response,
                "corr_before": corr_before,
                "corr_after": corr_after,
                "corr_gain": (
                    corr_after - corr_before
                    if np.isfinite(corr_after) and np.isfinite(corr_before)
                    else np.nan
                ),
                "max_side": max_side,
                "status": status,
                "error": error,
            }
        )

    return pd.DataFrame(rows)


def local_interband_shift_field(
    event: Any,
    *,
    master_band: Any,
    target_band: Any,
    window_size: int = 512,
    stride: int | None = None,
    max_shifts=(80, 80),
    min_std: float = 1e-6,
) -> pd.DataFrame:
    """
    Estimate a spatial field of local target→master translations.

    Windows with nearly constant content are marked ``low_texture`` rather than
    producing an arbitrary phase-correlation displacement.
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0.")
    if stride is None:
        stride = window_size
    if stride <= 0:
        raise ValueError("stride must be > 0.")

    master = _get_band(event, master_band).astype(np.float32)
    target = _get_band(event, target_band).astype(np.float32)
    master, target = _common_shape(master, target)
    h, w = master.shape

    if window_size > h or window_size > w:
        raise ValueError(
            f"window_size={window_size} exceeds common band shape {(h, w)}."
        )

    rows = []
    for y0 in range(0, h - window_size + 1, stride):
        for x0 in range(0, w - window_size + 1, stride):
            y1 = y0 + window_size
            x1 = x0 + window_size

            master_win = master[y0:y1, x0:x1]
            target_win = target[y0:y1, x0:x1]

            row = {
                "product_id": _infer_product_id(event),
                "level": _product_level(event),
                "master_band": master_band,
                "target_band": target_band,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "x_center": x0 + window_size / 2.0,
                "y_center": y0 + window_size / 2.0,
                "window_size": window_size,
                "stride": stride,
            }

            if (
                float(np.nanstd(master_win)) <= min_std
                or float(np.nanstd(target_win)) <= min_std
            ):
                row.update(
                    {
                        "dx_px": np.nan,
                        "dy_px": np.nan,
                        "shift_px": np.nan,
                        "corr_before": np.nan,
                        "corr_after": np.nan,
                        "corr_gain": np.nan,
                        "status": "low_texture",
                        "error": None,
                    }
                )
                rows.append(row)
                continue

            try:
                result = _phase_shift_target_to_master(
                    target_win,
                    master_win,
                    max_shifts=max_shifts,
                )
                dx, dy, _ = _parse_shift_result(result)
                aligned = warp_np_by_shift(
                    target_win,
                    np.array([dy, dx], dtype=np.float32),
                )

                corr_before = _finite_corr(master_win, target_win)
                corr_after = _finite_corr(master_win, aligned)

                row.update(
                    {
                        "dx_px": dx,
                        "dy_px": dy,
                        "shift_px": float(np.hypot(dx, dy)),
                        "corr_before": corr_before,
                        "corr_after": corr_after,
                        "corr_gain": (
                            corr_after - corr_before
                            if np.isfinite(corr_after)
                            and np.isfinite(corr_before)
                            else np.nan
                        ),
                        "status": "ok",
                        "error": None,
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "dx_px": np.nan,
                        "dy_px": np.nan,
                        "shift_px": np.nan,
                        "corr_before": np.nan,
                        "corr_after": np.nan,
                        "corr_gain": np.nan,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


def _edge_strength(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    gy, gx = np.gradient(arr)
    return np.hypot(gx, gy)


def _robust_unit_interval(
    image: np.ndarray,
    percentile: float = 99.5,
) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    scale = float(np.percentile(finite, percentile))
    if scale <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / scale, 0.0, 1.0)


def edge_overlay(
    event: Any,
    *,
    band_a: Any,
    band_b: Any,
    align: bool = False,
    max_shifts=(80, 80),
    percentile: float = 99.5,
) -> np.ndarray:
    """
    Return a red/cyan edge overlay for visual band-registration inspection.

    Red encodes edges from ``band_a``; cyan encodes edges from ``band_b``.
    Coincident edges therefore appear approximately white.
    """
    a = _get_band(event, band_a).astype(np.float32)
    b = _get_band(event, band_b).astype(np.float32)
    a, b = _common_shape(a, b)

    if align:
        result = _phase_shift_target_to_master(
            b,
            a,
            max_shifts=max_shifts,
        )
        dx, dy, _ = _parse_shift_result(result)
        b = warp_np_by_shift(
            b,
            np.array([dy, dx], dtype=np.float32),
        )

    ea = _robust_unit_interval(_edge_strength(a), percentile=percentile)
    eb = _robust_unit_interval(_edge_strength(b), percentile=percentile)

    return np.stack([ea, eb, eb], axis=-1).astype(np.float32)


def plot_shift_map(
    field: pd.DataFrame,
    *,
    ax=None,
    scale: float | None = None,
):
    """Plot a local shift field as image-coordinate arrows."""
    import matplotlib.pyplot as plt

    required = {"x_center", "y_center", "dx_px", "dy_px", "status"}
    missing = required - set(field.columns)
    if missing:
        raise ValueError(f"Shift field is missing columns: {sorted(missing)}")

    ok = field[
        (field["status"] == "ok")
        & field["dx_px"].notna()
        & field["dy_px"].notna()
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    ax.quiver(
        ok["x_center"],
        ok["y_center"],
        ok["dx_px"],
        ok["dy_px"],
        angles="xy",
        scale_units="xy",
        scale=scale,
    )
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_title("Local inter-band displacement field")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    return fig, ax
