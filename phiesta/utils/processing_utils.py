from __future__ import annotations

from typing import Tuple

import numpy as np


def percentile_stretch(
    x: np.ndarray,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalize an array to the [0, 1] range using robust percentiles.

    The input is converted to `float32`, lower and upper percentiles are computed
    with `np.nanpercentile`, and the result is linearly rescaled and clipped to
    `[0, 1]`.

    Args:
        x: Input array to normalize.
        p_lo: Lower percentile used for the contrast stretch.
        p_hi: Upper percentile used for the contrast stretch.
        eps: Small positive constant added to the denominator for numerical
            stability.

    Returns:
        A `float32` array with the same shape as the input, normalized to
        the `[0, 1]` range.
    """
    xf = x.astype(np.float32, copy=False)
    lo, hi = np.nanpercentile(xf, (p_lo, p_hi))
    y = (xf - lo) / (hi - lo + eps)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute a normalized difference between two arrays.

    The returned value is:

        `(a - b) / (a + b + eps)`

    Both inputs are converted to `float32` before the computation.

    Args:
        a: First input array.
        b: Second input array. Must be broadcast-compatible with `a`.
        eps: Small positive constant added to the denominator for numerical
            stability.

    Returns:
        A `float32` array containing the normalized difference.
    """
    af = a.astype(np.float32, copy=False)
    bf = b.astype(np.float32, copy=False)
    return (af - bf) / (af + bf + eps)


def make_rgb(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    stretch: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    Build an RGB image from three 2D bands.

    Each input band is independently normalized with :func:`percentile_stretch`,
    then stacked along the last axis to produce an image with shape `(H, W, 3)`.

    Args:
        r: Red channel as a 2D array.
        g: Green channel as a 2D array.
        b: Blue channel as a 2D array.
        stretch: Two-element tuple `(p_lo, p_hi)` defining the percentiles used
            for contrast stretching.

    Returns:
        A `float32` RGB image with shape `(H, W, 3)` and values in `[0, 1]`.
    """
    p_lo, p_hi = float(stretch[0]), float(stretch[1])
    rr = percentile_stretch(r, p_lo=p_lo, p_hi=p_hi)
    gg = percentile_stretch(g, p_lo=p_lo, p_hi=p_hi)
    bb = percentile_stretch(b, p_lo=p_lo, p_hi=p_hi)
    return np.stack([rr, gg, bb], axis=-1)