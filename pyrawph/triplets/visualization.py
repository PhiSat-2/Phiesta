from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import matplotlib.pyplot as plt


def _get_paths(triplet: dict[str, Any]) -> dict[str, str]:
    if "paths" in triplet:
        return triplet["paths"]
    return triplet


def _stretch_channel(x: np.ndarray, percentiles=(2, 98)) -> np.ndarray:
    x = x.astype(np.float32)
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.percentile(vals, percentiles)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def _stretch_rgb(rgb: np.ndarray, percentiles=(2, 98)) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        out[..., c] = _stretch_channel(rgb[..., c], percentiles=percentiles)
    return out


def _read_rgb(path: str | Path, kind: str, percentiles=(2, 98)) -> np.ndarray:
    """
    Read RGB from one triplet component.

    kind:
        - "real": real PhiSat-2 saved as [PAN, BLUE, GREEN, RED, RE1, RE2, RE3, NIR]
        - "sentinel": Sentinel saved as [BLUE, GREEN, RED, NIR, RE1, RE2, RE3]
        - "simulated": simulated PhiSat-2 saved as [BLUE, GREEN, RED, PAN, NIR, RE1, RE2, RE3]
    """
    with rasterio.open(path) as src:
        data = src.read()

    if kind == "real":
        rgb = np.transpose(data[[3, 2, 1]], (1, 2, 0))
    elif kind in ("sentinel", "simulated"):
        rgb = np.transpose(data[[2, 1, 0]], (1, 2, 0))
    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    return _stretch_rgb(rgb, percentiles=percentiles)


def _read_pan(path: str | Path, kind: str, percentiles=(2, 98)) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read()

    if kind == "real":
        pan = data[0]
    elif kind == "simulated":
        pan = data[3]
    else:
        raise ValueError("PAN is currently defined only for real and simulated PhiSat-2.")

    return _stretch_channel(pan, percentiles=percentiles)


def valid_fraction(path: str | Path, band: int = 1) -> float:
    """
    Fraction of non-zero finite pixels in a raster band.

    band is 1-indexed, as in rasterio.
    """
    with rasterio.open(path) as src:
        x = src.read(band)
    return float(np.mean(np.isfinite(x) & (x != 0)))


def inspect_full_triplet(triplet: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """
    Return basic shape and validity information for a full triplet.
    """
    paths = _get_paths(triplet)

    out: dict[str, Any] = {}

    for key in ["real", "sentinel", "simulated"]:
        path = paths[key]
        with rasterio.open(path) as src:
            out[key] = {
                "path": str(path),
                "count": int(src.count),
                "width": int(src.width),
                "height": int(src.height),
                "crs": str(src.crs),
                "descriptions": tuple(src.descriptions),
                "valid_fraction_band1": valid_fraction(path, band=1),
            }

    if verbose:
        for key, info in out.items():
            print(f"{key}:")
            print(f"  path: {info['path']}")
            print(f"  count: {info['count']}")
            print(f"  size: {info['width']} x {info['height']}")
            print(f"  crs: {info['crs']}")
            print(f"  valid_fraction_band1: {info['valid_fraction_band1']:.3f}")
            print(f"  descriptions: {info['descriptions']}")
            print()

    return out


def show_full_triplet(
    triplet: dict[str, Any],
    rgb_percentiles=(2, 98),
    pan_percentiles=(2, 98),
    figsize_rgb=(18, 6),
    figsize_overlay=(8, 8),
    show_overlay: bool = True,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Display a full triplet:
        real PhiSat-2 RGB,
        Sentinel-2 warped RGB,
        simulated PhiSat-2 warped RGB,
        optional PAN overlay real=red / simulated=green.

    Returns paths to saved figures if save_dir is provided.
    """
    paths = _get_paths(triplet)

    real_rgb = _read_rgb(paths["real"], kind="real", percentiles=rgb_percentiles)
    sentinel_rgb = _read_rgb(paths["sentinel"], kind="sentinel", percentiles=rgb_percentiles)
    simulated_rgb = _read_rgb(paths["simulated"], kind="simulated", percentiles=rgb_percentiles)

    saved: dict[str, Any] = {}

    fig, axes = plt.subplots(1, 3, figsize=figsize_rgb)

    axes[0].imshow(real_rgb)
    axes[0].set_title("Real ΦSat-2 RGB")
    axes[0].axis("off")

    axes[1].imshow(sentinel_rgb)
    axes[1].set_title("Sentinel-2 warped to ΦSat-2 grid")
    axes[1].axis("off")

    axes[2].imshow(simulated_rgb)
    axes[2].set_title("Simulated ΦSat-2 warped to real grid")
    axes[2].axis("off")

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = save_dir / "triplet_rgb_comparison.png"
        fig.savefig(rgb_path, dpi=180, bbox_inches="tight")
        saved["rgb_comparison"] = str(rgb_path)

    plt.show()

    if show_overlay:
        real_pan = _read_pan(paths["real"], kind="real", percentiles=pan_percentiles)
        simulated_pan = _read_pan(paths["simulated"], kind="simulated", percentiles=pan_percentiles)

        overlay = np.zeros((*real_pan.shape, 3), dtype=np.float32)
        overlay[..., 0] = real_pan
        overlay[..., 1] = simulated_pan

        fig, ax = plt.subplots(1, 1, figsize=figsize_overlay)
        ax.imshow(overlay)
        ax.set_title("PAN overlay: real=red, simulated=green")
        ax.axis("off")
        plt.tight_layout()

        if save_dir is not None:
            overlay_path = save_dir / "triplet_pan_overlay.png"
            fig.savefig(overlay_path, dpi=180, bbox_inches="tight")
            saved["pan_overlay"] = str(overlay_path)

        plt.show()

    return {
        "status": "SUCCESS",
        "saved": saved,
        "paths": paths,
        "inspection": inspect_full_triplet(triplet, verbose=False),
    }
