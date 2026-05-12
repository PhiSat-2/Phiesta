from __future__ import annotations

from typing import Any, Sequence
import math

import numpy as np
import matplotlib.pyplot as plt

from .display import event_band_label


def _as_band_list(event, bands: Any) -> list[Any]:
    arr = event.as_numpy()
    count = arr.shape[0]

    if bands is None or bands == "all":
        return list(range(count))

    if isinstance(bands, (int, str)):
        return [bands]

    return list(bands)


def _finite_sample(
    x: np.ndarray,
    sample: int | None = 500_000,
    random_state: int = 0,
) -> np.ndarray:
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]

    if sample is not None and x.size > sample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(x.size, size=sample, replace=False)
        x = x[idx]

    return x.astype(np.float32, copy=False)


def compute_band_stats(
    event,
    bands: Any = "all",
    percentiles: Sequence[float] = (0, 1, 2, 5, 50, 95, 98, 99, 100),
    sample: int | None = 500_000,
    random_state: int = 0,
) -> dict[str, dict[str, float]]:
    """
    Compute raw-value statistics for selected event bands.
    """
    band_list = _as_band_list(event, bands)
    stats = {}

    for band in band_list:
        label = event_band_label(event, band, compact=True)
        values = _finite_sample(
            event.get_band(band),
            sample=sample,
            random_state=random_state,
        )

        if values.size == 0:
            stats[label] = {"count": 0}
            continue

        band_stats = {
            "count": int(values.size),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

        qs = np.percentile(values, percentiles)
        for p, q in zip(percentiles, qs):
            key = f"p{str(p).replace('.', '_')}"
            band_stats[key] = float(q)

        stats[label] = band_stats

    return stats


def plot_value_distribution(
    event,
    bands: Any = "all",
    bins: int = 256,
    sample: int | None = 500_000,
    log_y: bool = True,
    percentiles: Sequence[float] = (1, 2, 50, 98, 99),
    hist_range_percentiles: tuple[float, float] | None = (0.1, 99.9),
    random_state: int = 0,
    figsize=(12, 7),
    title: str | None = None,
    out_png: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Plot raw-value histograms for selected event bands.

    The histogram range can be clipped for readability, but the returned stats
    are computed from the sampled raw values.
    """
    band_list = _as_band_list(event, bands)
    n = len(band_list)

    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize[0] * ncols / 2.0, figsize[1] * nrows / 2.0),
        squeeze=False,
    )

    stats = {}

    for i, band in enumerate(band_list):
        ax = axes.flat[i]
        label = event_band_label(event, band, compact=True)

        values = _finite_sample(
            event.get_band(band),
            sample=sample,
            random_state=random_state,
        )

        if values.size == 0:
            ax.set_title(f"{label} — no finite values")
            ax.axis("off")
            stats[label] = {"count": 0}
            continue

        if hist_range_percentiles is not None:
            lo, hi = np.percentile(values, hist_range_percentiles)
            hist_range = (float(lo), float(hi)) if hi > lo else None
        else:
            hist_range = None

        ax.hist(values, bins=bins, range=hist_range)

        qs = np.percentile(values, percentiles)
        for p, q in zip(percentiles, qs):
            ax.axvline(q, linestyle="--", linewidth=1)
            ax.text(q, ax.get_ylim()[1] * 0.9, f"p{p:g}", rotation=90, va="top", fontsize=8)

        if log_y:
            ax.set_yscale("log")

        ax.set_title(label)
        ax.set_xlabel("Raw value")
        ax.set_ylabel("Count")

        band_stats = {
            "count": int(values.size),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

        for p, q in zip(percentiles, qs):
            band_stats[f"p{str(p).replace('.', '_')}"] = float(q)

        stats[label] = band_stats

    for j in range(n, nrows * ncols):
        axes.flat[j].axis("off")

    plt.suptitle(title or "Band value distributions")
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=180, bbox_inches="tight")

    plt.show()
    return stats