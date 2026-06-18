from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .display import event_band_label


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def _downsample_for_display(img: np.ndarray, max_side: int | None = 1800) -> np.ndarray:
    """
    Lightweight display downsampling by integer stride.

    This avoids making notebooks painfully slow when displaying 4096 x 4096 images.
    It is only for visualization; no analysis is performed on the downsampled image.
    """
    if max_side is None:
        return img

    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img

    step = int(np.ceil(m / max_side))
    if img.ndim == 2:
        return img[::step, ::step]
    return img[::step, ::step, :]


def _finite_values(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x)
    valid = np.isfinite(x)

    if mask is not None:
        valid &= mask

    vals = x[valid]
    return vals.astype(np.float32, copy=False)


def normalize_band(
    x: np.ndarray,
    percentiles: tuple[float, float] = (1, 99),
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normalize one band to [0, 1] using percentile clipping.

    This is meant for display, not physical calibration.
    """
    x = np.asarray(x, dtype=np.float32)
    vals = _finite_values(x, mask=mask)

    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.percentile(vals, percentiles)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def make_display_rgb(
    event,
    bands: Sequence[Any] = ("RED", "GREEN", "BLUE"),
    percentiles: tuple[float, float] = (1, 99),
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a display RGB composite from an event.

    Each channel is normalized independently with percentile clipping.
    This is the correct default for raw ΦSat-2 DN visualization.
    """
    if len(bands) != 3:
        raise ValueError(f"Expected exactly 3 bands for RGB, got {bands!r}")

    channels = [
        normalize_band(event.get_band(b), percentiles=percentiles, mask=mask)
        for b in bands
    ]

    return np.dstack(channels)


def _simple_bright_mask(
    event,
    bands: Sequence[Any] = ("BLUE", "GREEN", "RED"),
    high_percentile: float = 98.5,
) -> np.ndarray:
    """
    Very simple bright-pixel mask.

    This is not a cloud detector. It is only useful to understand whether bright
    clouds dominate display percentiles.
    """
    arr = np.dstack([event.get_band(b).astype(np.float32) for b in bands])
    brightness = np.nanmean(arr, axis=-1)
    vals = brightness[np.isfinite(brightness)]

    if vals.size == 0:
        return np.zeros(brightness.shape, dtype=bool)

    threshold = np.percentile(vals, high_percentile)
    return brightness > threshold


def _raw_ndvi_like(event, nir="NIR", red="RED") -> np.ndarray:
    """
    Raw NDVI-like index computed on raw DN values.

    This is not a physically calibrated NDVI unless bands have already been
    converted to comparable reflectance-like units.
    """
    n = event.get_band(nir).astype(np.float32)
    r = event.get_band(red).astype(np.float32)
    return (n - r) / (n + r + 1e-6)


def plot_display_diagnostics(
    event,
    product_id: str | None = None,
    percentiles: tuple[float, float] = (1, 99),
    registered: bool = True,
    registration_master: Any = "NIR",
    max_shifts: tuple[int, int] = (80, 80),
    force_registration: bool = False,
    natural_bands: Sequence[Any] = ("RED", "GREEN", "BLUE"),
    false_color_bands: Sequence[Any] = ("NIR", "RED", "GREEN"),
    vegetation_bands: Sequence[Any] = ("NIR", "RE1", "RED"),
    gray_band: Any = "NIR",
    include_raw_ndvi: bool = True,
    include_stats: bool = True,
    stats_bands: Sequence[Any] = ("PAN", "BLUE", "GREEN", "RED", "RE1", "RE2", "RE3", "NIR"),
    stats_percentiles: Sequence[float] = (1, 2, 5, 50, 95, 98, 99),
    stats_sample: int | None = 1_000_000,
    max_side: int | None = 1800,
    figsize: tuple[float, float] = (18, 12),
    out_png: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """
    Plot standard display diagnostics for a ΦSat-2 L1 event.

    This function is intended for visual scene review, not physical analysis.

    It shows:
    - natural RGB;
    - false color NIR/RED/GREEN;
    - vegetation composite NIR/RE1/RED;
    - one normalized grayscale band, usually NIR;
    - raw NDVI-like map;
    - raw NDVI-like distribution.

    Important:
    The NDVI-like panel is computed from raw DN values. It should not be used as
    a physical vegetation index unless data are calibrated into comparable
    reflectance-like units.
    """
    ev = event

    if registered:
        if not hasattr(event, "_registered_for_display"):
            raise AttributeError("This event does not support display registration.")
        ev = event._registered_for_display(
            master_band=registration_master,
            max_shifts=max_shifts,
            force=force_registration,
        )

    suffix = f" — {product_id}" if product_id is not None else ""
    reg_text = f"registered to {registration_master}" if registered else "raw"

    natural = make_display_rgb(ev, natural_bands, percentiles=percentiles)
    false_color = make_display_rgb(ev, false_color_bands, percentiles=percentiles)
    vegetation = make_display_rgb(ev, vegetation_bands, percentiles=percentiles)
    gray = normalize_band(ev.get_band(gray_band), percentiles=percentiles)

    raw_ndvi = None
    if include_raw_ndvi:
        try:
            raw_ndvi = _raw_ndvi_like(ev, nir="NIR", red="RED")
        except Exception:
            raw_ndvi = None

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    axes[0, 0].imshow(_downsample_for_display(natural, max_side=max_side))
    axes[0, 0].set_title(f"Natural RGB {tuple(natural_bands)}{suffix}\n{reg_text}, p={percentiles}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_downsample_for_display(false_color, max_side=max_side))
    axes[0, 1].set_title(f"False color {tuple(false_color_bands)}\n{reg_text}, p={percentiles}")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(_downsample_for_display(vegetation, max_side=max_side))
    axes[0, 2].set_title(f"Vegetation {tuple(vegetation_bands)}\n{reg_text}, p={percentiles}")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(_downsample_for_display(gray, max_side=max_side), cmap="gray")
    axes[1, 0].set_title(f"{event_band_label(ev, gray_band, compact=True)}\npercentile normalized")
    axes[1, 0].axis("off")

    if raw_ndvi is not None:
        ndvi_display = _downsample_for_display(raw_ndvi, max_side=max_side)
        im = axes[1, 1].imshow(ndvi_display, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[1, 1].set_title("Raw NDVI-like index\nnot physically calibrated")
        axes[1, 1].axis("off")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

        vals = raw_ndvi[np.isfinite(raw_ndvi)]
        if vals.size > 2_000_000:
            rng = np.random.default_rng(0)
            vals = vals[rng.choice(vals.size, size=2_000_000, replace=False)]
        axes[1, 2].hist(vals.ravel(), bins=200, range=(-1, 1), log=True)
        axes[1, 2].set_title("Raw NDVI-like distribution")
        axes[1, 2].set_xlabel("value")
        axes[1, 2].set_ylabel("pixel count, log scale")
    else:
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")

    plt.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    stats = None
    if include_stats:
        stats = event.band_stats(
            bands=stats_bands,
            percentiles=stats_percentiles,
            sample=stats_sample,
        )

    return {
        "figure": fig,
        "axes": axes,
        "stats": stats,
        "event_used_for_display": ev,
        "registered": registered,
        "registration_master": registration_master,
        "percentiles": percentiles,
        "out_png": str(out_png) if out_png is not None else None,
    }


def compare_display_stretches(
    event,
    bands: Sequence[Any] = ("NIR", "RED", "GREEN"),
    stretches: Sequence[tuple[float, float]] = ((0.5, 99.5), (1, 99), (2, 98), (5, 95)),
    registered: bool = True,
    registration_master: Any = "NIR",
    max_shifts: tuple[int, int] = (80, 80),
    force_registration: bool = False,
    max_side: int | None = 1400,
    figsize: tuple[float, float] | None = None,
    out_png: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """
    Compare several percentile stretches for one RGB composite.
    """
    ev = event

    if registered:
        if not hasattr(event, "_registered_for_display"):
            raise AttributeError("This event does not support display registration.")
        ev = event._registered_for_display(
            master_band=registration_master,
            max_shifts=max_shifts,
            force=force_registration,
        )

    if figsize is None:
        figsize = (5 * len(stretches), 5)

    fig, axes = plt.subplots(1, len(stretches), figsize=figsize)

    if len(stretches) == 1:
        axes = [axes]

    for ax, p in zip(axes, stretches):
        rgb = make_display_rgb(ev, bands=bands, percentiles=p)
        ax.imshow(_downsample_for_display(rgb, max_side=max_side))
        ax.set_title(f"{tuple(bands)}\npercentile {p}")
        ax.axis("off")

    plt.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    return {
        "figure": fig,
        "axes": axes,
        "event_used_for_display": ev,
        "bands": tuple(bands),
        "stretches": list(stretches),
        "registered": registered,
        "registration_master": registration_master,
        "out_png": str(out_png) if out_png is not None else None,
    }
