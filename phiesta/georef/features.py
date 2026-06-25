from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = np.nanpercentile(x, 2)
    hi = np.nanpercentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def resize_to_gsd(img_hwc: np.ndarray, src_gsd_m: float, dst_gsd_m: float) -> np.ndarray:
    if src_gsd_m <= 0 or dst_gsd_m <= 0:
        raise ValueError("src_gsd_m and dst_gsd_m must be > 0")
    scale = float(src_gsd_m) / float(dst_gsd_m)
    if np.isclose(scale, 1.0):
        return img_hwc.astype(np.float32)
    h, w = img_hwc.shape[:2]
    nw = max(32, int(round(w * scale)))
    nh = max(32, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img_hwc.astype(np.float32), (nw, nh), interpolation=interp)


def build_match_feature_from_4band(img_4band_hwc: np.ndarray) -> np.ndarray:
    """
    Default robust feature for coarse matching.

    Input band order is expected to be [BLUE, GREEN, RED, NIR].
    The output is a single-channel structural feature designed to remain useful
    across moderate radiometric differences.
    """
    img = np.asarray(img_4band_hwc, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"Expected HWC 4-band image, got {img.shape}")

    blue = norm01(img[..., 0])
    green = norm01(img[..., 1])
    red = norm01(img[..., 2])
    nir = norm01(img[..., 3])

    pseudo_pan = 0.15 * blue + 0.30 * green + 0.30 * red + 0.25 * nir
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = norm01(ndvi)

    sobelx_pan = cv2.Sobel(pseudo_pan, cv2.CV_32F, 1, 0, ksize=3)
    sobely_pan = cv2.Sobel(pseudo_pan, cv2.CV_32F, 0, 1, ksize=3)
    grad_pan = cv2.magnitude(sobelx_pan, sobely_pan)

    sobelx_ndvi = cv2.Sobel(ndvi, cv2.CV_32F, 1, 0, ksize=3)
    sobely_ndvi = cv2.Sobel(ndvi, cv2.CV_32F, 0, 1, ksize=3)
    grad_ndvi = cv2.magnitude(sobelx_ndvi, sobely_ndvi)

    feature = 0.55 * norm01(grad_pan) + 0.25 * norm01(np.abs(ndvi - 0.5)) + 0.20 * norm01(grad_ndvi)
    return norm01(feature).astype(np.float32)


def prepare_phi_for_matching(l1_event, dst_gsd_m: float = 10.0, src_gsd_m: float = 4.75) -> Tuple[np.ndarray, np.ndarray]:
    phi_hwc = np.stack(
        [
            l1_event.get_band("BLUE").astype(np.float32),
            l1_event.get_band("GREEN").astype(np.float32),
            l1_event.get_band("RED").astype(np.float32),
            l1_event.get_band("NIR").astype(np.float32),
        ],
        axis=-1,
    )
    phi_hwc = resize_to_gsd(phi_hwc, src_gsd_m=src_gsd_m, dst_gsd_m=dst_gsd_m)
    phi_feat = build_match_feature_from_4band(phi_hwc)
    return phi_hwc, phi_feat


def prepare_s2_for_matching(s2_hwc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s2_hwc = np.asarray(s2_hwc, dtype=np.float32)
    if s2_hwc.ndim != 3 or s2_hwc.shape[2] != 4:
        raise ValueError(f"Expected HWC 4-band image, got {s2_hwc.shape}")
    s2_feat = build_match_feature_from_4band(s2_hwc)
    return s2_hwc, s2_feat
