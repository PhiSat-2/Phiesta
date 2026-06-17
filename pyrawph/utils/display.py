from __future__ import annotations

from typing import Any, Sequence
import math

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_RGB_BANDS = ("RED", "GREEN", "BLUE")

PHISAT2_BAND_INFO = {
    0: {"name": "PAN / B0", "wavelength_nm": 625, "bandwidth_nm": None},
    0: {"name": "PAN", "wavelength_nm": 625, "bandwidth_nm": 250},
    1: {"name": "BLUE / MS1", "wavelength_nm": 490, "bandwidth_nm": 65},
    2: {"name": "GREEN / MS2", "wavelength_nm": 560, "bandwidth_nm": 35},
    3: {"name": "RED / MS3", "wavelength_nm": 665, "bandwidth_nm": 30},
    4: {"name": "RE1 / MS4", "wavelength_nm": 705, "bandwidth_nm": 15},
    5: {"name": "RE2 / MS5", "wavelength_nm": 740, "bandwidth_nm": 15},
    6: {"name": "RE3 / MS6", "wavelength_nm": 783, "bandwidth_nm": 20},
    7: {"name": "NIR / MS7", "wavelength_nm": 842, "bandwidth_nm": 115},
}

PHISAT2_ALIAS_TO_INDEX = {
    "PAN": 0,
    "PANCHROMATIC": 0,
    "B0": 0,
    "625NM": 0,
    "PAN": 0,
    "BLUE": 1,
    "GREEN": 2,
    "RED": 3,
    "RE1": 4,
    "RE2": 5,
    "RE3": 6,
    "NIR": 7,
}

def _resolve_event_band_index(event, band) -> int | None:
    if isinstance(band, int):
        return int(band)

    if isinstance(band, str):
        b = band.upper()

        if b in PHISAT2_ALIAS_TO_INDEX:
            return PHISAT2_ALIAS_TO_INDEX[b]

        if b.startswith("B") and b[1:].isdigit():
            return int(b[1:])

        if b.startswith("BAND_") and b[5:].isdigit():
            return int(b[5:])

        if b.isdigit():
            return int(b)

    return None


def _event_wavelengths(event) -> list | None:
    meta = event.get_meta() if hasattr(event, "get_meta") else {}

    for key in ["wavelengths_nm", "band_wavelength_nm", "band_wavelengths_nm"]:
        if key in meta and meta[key] is not None:
            return list(meta[key])

    return None


def event_band_label(event, band, compact: bool = False) -> str:
    idx = _resolve_event_band_index(event, band)

    if idx is None:
        return str(band)

    info = PHISAT2_BAND_INFO.get(idx, {})
    name = info.get("name", f"Band {idx}")

    wavelengths = _event_wavelengths(event)
    if wavelengths is not None and 0 <= idx < len(wavelengths):
        wavelength = wavelengths[idx]
    else:
        wavelength = info.get("wavelength_nm")

    bandwidth = info.get("bandwidth_nm")

    if compact:
        if wavelength is not None:
            return f"{name} ({wavelength} nm)"
        return f"{name}"

    parts = [f"Band {idx}", name]

    if wavelength is not None:
        parts.append(f"{wavelength} nm")

    if bandwidth is not None:
        parts.append(f"BW {bandwidth} nm")

    return " — ".join(parts)

def normalize_for_display(
    x: np.ndarray,
    normalization: str = "percentile",
    percentiles: tuple[float, float] = (2, 98),
) -> np.ndarray:
    """
    Normalize one 2D image for visualization.
    """
    x = np.asarray(x, dtype=np.float32)

    if normalization in (None, "none", "raw"):
        return x

    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)

    xv = x[valid]

    if normalization == "percentile":
        lo, hi = np.percentile(xv, percentiles)
    elif normalization == "minmax":
        lo, hi = float(np.nanmin(xv)), float(np.nanmax(xv))
    elif normalization == "zscore":
        mu = float(np.nanmean(xv))
        sigma = float(np.nanstd(xv)) + 1e-6
        z = (x - mu) / sigma
        return np.clip((z + 3.0) / 6.0, 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(
            f"Unsupported normalization={normalization!r}. "
            "Use 'percentile', 'minmax', 'zscore', or 'none'."
        )

    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _as_band_list(bands: Any, count: int) -> list[Any]:
    if bands is None:
        return list(DEFAULT_RGB_BANDS)

    if isinstance(bands, str):
        if bands.lower() == "all":
            return list(range(count))
        return [bands]

    if isinstance(bands, int):
        return [bands]

    return list(bands)


def prepare_event_display_image(
    event,
    bands: Any = DEFAULT_RGB_BANDS,
    normalize: bool = True,
    normalization: str = "percentile",
    percentiles: tuple[float, float] = (2, 98),
    per_band: bool = True,
):
    """
    Prepare an event image for display.

    Rules:
    - one band -> grayscale image
    - three bands -> RGB image
    - all / several non-RGB bands -> grid mode
    """
    arr = event.as_numpy()
    band_list = _as_band_list(bands, count=arr.shape[0])
    labels = [event_band_label(event, b, compact=False) for b in band_list]
    compact_labels = [event_band_label(event, b, compact=True) for b in band_list]

    images = [event.get_band(b).astype(np.float32) for b in band_list]

    if len(images) == 1:
        img = images[0]
        if normalize:
            img = normalize_for_display(
                img,
                normalization=normalization,
                percentiles=percentiles,
            )
        return {
        "mode": "gray",
        "image": img,
        "bands": band_list,
        "labels": labels,
        "compact_labels": compact_labels,
    }

    if len(images) == 3:
        stack = np.stack(images, axis=-1)

        if normalize:
            if per_band:
                stack = np.stack(
                    [
                        normalize_for_display(
                            stack[..., i],
                            normalization=normalization,
                            percentiles=percentiles,
                        )
                        for i in range(3)
                    ],
                    axis=-1,
                )
            else:
                stack = normalize_for_display(
                    stack,
                    normalization=normalization,
                    percentiles=percentiles,
                )

        return {
        "mode": "rgb",
        "image": stack.astype(np.float32),
        "bands": band_list,
        "labels": labels,
        "compact_labels": compact_labels,
    }

    # Grid mode for "all" or arbitrary number of bands.
    grid_images = []
    for img in images:
        if normalize:
            img = normalize_for_display(
                img,
                normalization=normalization,
                percentiles=percentiles,
            )
        grid_images.append(img)

    return {
        "mode": "grid",
        "images": grid_images,
        "bands": band_list,
        "labels": labels,
        "compact_labels": compact_labels,
    }


def show_prepared_display(
    prepared: dict,
    figsize=(8, 8),
    title: str | None = None,
    interpolation: str = "nearest",
    out_png: str | None = None,
):
    """
    Show an image prepared by prepare_event_display_image(...).
    """
    mode = prepared["mode"]

    if mode == "gray":
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(prepared["image"], cmap="gray", interpolation=interpolation)
        ax.set_title(title or prepared["labels"][0])
        ax.axis("off")

    elif mode == "rgb":
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(prepared["image"], interpolation=interpolation)
        rgb_title = "RGB " + " / ".join(prepared["compact_labels"])
        ax.set_title(title or rgb_title)
        ax.axis("off")

    elif mode == "grid":
        images = prepared["images"]
        bands = prepared["bands"]
        n = len(images)
        ncols = min(4, n)
        nrows = int(math.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize[0] * ncols / 2.0, figsize[1] * nrows / 2.0),
            squeeze=False,
        )

        for i, ax in enumerate(axes.flat):
            if i >= n:
                ax.axis("off")
                continue

            ax.imshow(images[i], cmap="gray", interpolation=interpolation)
            ax.set_title(prepared["labels"][i], fontsize=9)
            ax.axis("off")

        plt.suptitle(title or "Bands")
    else:
        raise ValueError(f"Unsupported display mode: {mode}")

    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=180, bbox_inches="tight")

    plt.show()
    return prepared

def _format_percentiles(percentiles) -> str:
    if percentiles is None:
        return ""
    if isinstance(percentiles, (list, tuple)) and len(percentiles) == 2:
        return f"({percentiles[0]:g}, {percentiles[1]:g})"
    return str(percentiles)


def format_event_display_title(
    level: str,
    prepared: dict,
    registered: bool = False,
    normalize: bool = True,
    normalization: str = "percentile",
    percentiles=(2, 98),
    per_band: bool = True,
    registration_master=None,
) -> str:
    """
    Build a compact, informative title for event visualization.
    """
    mode = prepared.get("mode", "image")
    compact_labels = prepared.get("compact_labels", prepared.get("bands", []))

    if mode == "rgb":
        band_part = "RGB " + " / ".join(str(x) for x in compact_labels)

    elif mode == "gray":
        band_part = str(compact_labels[0]) if compact_labels else "Band"

    elif mode == "grid":
        bands = prepared.get("bands", [])
        if len(bands) == 0:
            band_part = "Bands"
        elif len(bands) == 1:
            band_part = str(compact_labels[0])
        else:
            band_part = f"{len(bands)} bands"

    else:
        band_part = str(mode)

    if registered:
        if registration_master is not None:
            reg_part = f"registered to {registration_master}"
        else:
            reg_part = "registered"
    else:
        reg_part = "raw"

    if (not normalize) or normalization in (None, "none", "raw"):
        norm_part = "norm none"
    elif normalization == "percentile":
        norm_part = f"norm percentile {_format_percentiles(percentiles)}"
    else:
        norm_part = f"norm {normalization}"

    if normalize and per_band and mode == "rgb":
        norm_part += ", per-band"

    return f"{level} — {band_part} — {reg_part} — {norm_part}"