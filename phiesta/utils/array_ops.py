from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _event_band_count(event) -> int:
    meta = getattr(event, "meta", None) or getattr(event, "_meta", {}) or {}

    count = meta.get("count")
    if count is not None:
        return int(count)

    arr = getattr(event, "_arr", None)
    if arr is not None and hasattr(arr, "shape") and len(arr.shape) >= 1:
        return int(arr.shape[0])

    raise ValueError("Could not determine event band count.")


def resolve_band_selectors(event, bands: Any = "all") -> list[Any]:
    """
    Resolve a user band selection into a list of selectors.

    Accepted inputs:
    - "all" or None: all bands;
    - one selector: "NIR", 7, 842;
    - a sequence of selectors: ("RED", "GREEN", "BLUE").
    """
    if bands is None or bands == "all":
        return list(range(_event_band_count(event)))

    if isinstance(bands, (str, int, float)):
        return [bands]

    if isinstance(bands, Sequence):
        return list(bands)

    raise TypeError(f"Unsupported band selector: {bands!r}")


def to_cube(
    event,
    bands: Any = "all",
    band_axis: int = 0,
    dtype: Any | None = None,
    copy: bool = True,
) -> np.ndarray:
    """
    Return selected event bands as a NumPy cube.

    Args:
        event: L0_event or L1_event-like object supporting get_band.
        bands: "all", one selector, or a sequence of selectors.
        band_axis: output band axis. Use 0 for (B, H, W), -1 for (H, W, B).
        dtype: optional output dtype.
        copy: whether to return a copy.

    Returns:
        NumPy array containing selected bands.
    """
    selectors = resolve_band_selectors(event, bands)

    arrays = []
    for selector in selectors:
        band = event.get_band(selector)
        arrays.append(np.asarray(band))

    cube = np.stack(arrays, axis=0)

    if band_axis not in (0, -1, 2):
        cube = np.moveaxis(cube, 0, band_axis)
    elif band_axis in (-1, 2):
        cube = np.moveaxis(cube, 0, -1)

    if dtype is not None:
        cube = cube.astype(dtype, copy=False)

    if copy:
        cube = cube.copy()

    return cube


def _resolve_window(
    height_total: int,
    width_total: int,
    x_min: int,
    y_min: int,
    width: int | None = None,
    height: int | None = None,
    x_max: int | None = None,
    y_max: int | None = None,
    clip: bool = True,
) -> tuple[int, int, int, int]:
    if x_max is None:
        if width is None:
            raise ValueError("Provide either x_max or width.")
        x_max = x_min + width

    if y_max is None:
        if height is None:
            raise ValueError("Provide either y_max or height.")
        y_max = y_min + height

    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)

    if clip:
        x_min = max(0, min(width_total, x_min))
        x_max = max(0, min(width_total, x_max))
        y_min = max(0, min(height_total, y_min))
        y_max = max(0, min(height_total, y_max))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            f"Invalid patch window: x=[{x_min}, {x_max}), y=[{y_min}, {y_max})"
        )

    return x_min, y_min, x_max, y_max


def get_patch(
    event,
    x_min: int,
    y_min: int,
    width: int | None = None,
    height: int | None = None,
    x_max: int | None = None,
    y_max: int | None = None,
    bands: Any = "all",
    band_axis: int = 0,
    dtype: Any | None = None,
    squeeze: bool = True,
    clip: bool = True,
    copy: bool = True,
) -> np.ndarray:
    """
    Extract a spatial patch from selected event bands.

    Args:
        x_min, y_min: top-left pixel coordinates.
        width, height: patch size. Alternative to x_max/y_max.
        x_max, y_max: exclusive bottom-right coordinates.
        bands: "all", one selector, or a sequence.
        band_axis: output band axis for multi-band patches.
        squeeze: if True and only one band is selected, return a 2D array.
        clip: clip window to image bounds.
        copy: return a copy.

    Returns:
        2D array for one band if squeeze=True, otherwise a 3D cube.
    """
    selectors = resolve_band_selectors(event, bands)

    first = np.asarray(event.get_band(selectors[0]))
    h_total, w_total = first.shape[:2]

    x_min, y_min, x_max, y_max = _resolve_window(
        height_total=h_total,
        width_total=w_total,
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height,
        x_max=x_max,
        y_max=y_max,
        clip=clip,
    )

    arrays = []
    for selector in selectors:
        band = np.asarray(event.get_band(selector))
        arrays.append(band[y_min:y_max, x_min:x_max])

    if len(arrays) == 1 and squeeze:
        patch = arrays[0]
    else:
        patch = np.stack(arrays, axis=0)

        if band_axis not in (0, -1, 2):
            patch = np.moveaxis(patch, 0, band_axis)
        elif band_axis in (-1, 2):
            patch = np.moveaxis(patch, 0, -1)

    if dtype is not None:
        patch = patch.astype(dtype, copy=False)

    if copy:
        patch = patch.copy()

    return patch


def _finite_values(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(x)

    if mask is not None:
        valid &= mask

    return x[valid]


def normalize_array(
    array: np.ndarray,
    method: str = "percentile",
    percentiles: tuple[float, float] = (1, 99),
    per_band: bool = True,
    band_axis: int = 0,
    mask: np.ndarray | None = None,
    out_range: tuple[float, float] = (0.0, 1.0),
    dtype: Any = np.float32,
) -> np.ndarray:
    """
    Normalize a 2D band or 3D cube.

    This is intended for display and ML preprocessing experiments.
    It is not physical radiometric calibration.

    Args:
        array: 2D or 3D array.
        method: "percentile", "minmax", or "none".
        percentiles: percentile clipping range for method="percentile".
        per_band: for 3D arrays, normalize each band independently.
        band_axis: band/channel axis for 3D arrays.
        mask: optional valid mask used to compute normalization stats.
        out_range: output range.
        dtype: output dtype.

    Returns:
        Normalized array.
    """
    arr = np.asarray(array).astype(np.float32, copy=False)

    if method == "none":
        return arr.astype(dtype, copy=True)

    lo_out, hi_out = out_range

    def normalize_one(x: np.ndarray) -> np.ndarray:
        vals = _finite_values(x, mask=mask)

        if vals.size == 0:
            return np.zeros_like(x, dtype=np.float32)

        if method == "percentile":
            lo, hi = np.percentile(vals, percentiles)
        elif method == "minmax":
            lo, hi = np.nanmin(vals), np.nanmax(vals)
        else:
            raise ValueError(f"Unknown normalization method: {method!r}")

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype=np.float32)

        y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        return y * (hi_out - lo_out) + lo_out

    if arr.ndim == 2:
        out = normalize_one(arr)
        return out.astype(dtype, copy=False)

    if arr.ndim != 3:
        raise ValueError(f"Expected a 2D band or 3D cube, got shape {arr.shape}")

    if not per_band:
        vals = _finite_values(arr, mask=mask)

        if vals.size == 0:
            return np.zeros_like(arr, dtype=dtype)

        if method == "percentile":
            lo, hi = np.percentile(vals, percentiles)
        elif method == "minmax":
            lo, hi = np.nanmin(vals), np.nanmax(vals)
        else:
            raise ValueError(f"Unknown normalization method: {method!r}")

        if hi <= lo:
            return np.zeros_like(arr, dtype=dtype)

        out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        out = out * (hi_out - lo_out) + lo_out
        return out.astype(dtype, copy=False)

    moved = np.moveaxis(arr, band_axis, 0)
    out = np.empty_like(moved, dtype=np.float32)

    for idx in range(moved.shape[0]):
        out[idx] = normalize_one(moved[idx])

    out = np.moveaxis(out, 0, band_axis)
    return out.astype(dtype, copy=False)


def show_patch(
    event,
    x_min: int,
    y_min: int,
    width: int | None = None,
    height: int | None = None,
    x_max: int | None = None,
    y_max: int | None = None,
    bands: Any = ("RED", "GREEN", "BLUE"),
    registered: bool = False,
    registration_master: Any = "NIR",
    normalization: str = "percentile",
    percentiles: tuple[float, float] = (1, 99),
    per_band: bool = True,
    cmap: str = "gray",
    figsize: tuple[float, float] = (7, 7),
    title: str | None = None,
    show: bool = True,
):
    """
    Display a spatial patch.

    Supports:
    - one band: grayscale;
    - three bands: RGB composite;
    - more bands: grid of normalized bands.

    If registered=True and the event supports display registration, bands are
    registered to registration_master before extracting the patch.
    """
    import matplotlib.pyplot as plt

    ev = event

    if registered:
        if not hasattr(event, "_registered_for_display"):
            raise AttributeError("This event does not support display registration.")
        ev = event._registered_for_display(master_band=registration_master)

    selectors = resolve_band_selectors(ev, bands)

    patch = get_patch(
        ev,
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height,
        x_max=x_max,
        y_max=y_max,
        bands=selectors,
        band_axis=0,
        squeeze=False,
        clip=True,
    )

    n_bands = patch.shape[0]

    if n_bands == 1:
        img = normalize_array(
            patch[0],
            method=normalization,
            percentiles=percentiles,
            per_band=False,
        )

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title or f"Patch {selectors[0]}")
        ax.axis("off")

    elif n_bands == 3:
        img = np.moveaxis(patch, 0, -1)
        img = normalize_array(
            img,
            method=normalization,
            percentiles=percentiles,
            per_band=per_band,
            band_axis=-1,
        )

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img)
        ax.set_title(title or f"Patch {tuple(selectors)}")
        ax.axis("off")

    else:
        ncols = min(4, n_bands)
        nrows = int(np.ceil(n_bands / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize[0] * ncols / 2, figsize[1] * nrows / 2),
        )
        axes = np.array(axes).reshape(-1)

        for ax in axes:
            ax.axis("off")

        for idx, selector in enumerate(selectors):
            img = normalize_array(
                patch[idx],
                method=normalization,
                percentiles=percentiles,
                per_band=False,
            )

            axes[idx].imshow(img, cmap=cmap)
            axes[idx].set_title(str(selector))
            axes[idx].axis("off")

        fig.suptitle(title or "Patch bands")
        plt.tight_layout()

    if show:
        plt.show()

    return {
        "figure": fig,
        "patch": patch,
        "bands": selectors,
        "window": {
            "x_min": x_min,
            "y_min": y_min,
            "width": width,
            "height": height,
            "x_max": x_max,
            "y_max": y_max,
        },
        "registered": registered,
        "registration_master": registration_master if registered else None,
    }
