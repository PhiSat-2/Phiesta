from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio


def _event_shape(event: Any) -> tuple[int, int]:
    arr = getattr(event, "_arr", None)
    if arr is None:
        raise AttributeError("Event has no _arr attribute.")
    return int(arr.shape[1]), int(arr.shape[2])  # H, W


def _perspective_transform_points(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out


def estimate_final_sentinel_window_from_proxy(
    event: Any,
    sentinel_big_crop_path: str | Path,
    proxy_simulated_path: str | Path,
    rectification_homography: list | np.ndarray,
    H_rectified_to_real: list | np.ndarray,
    margin_pct: float = 0.15,
    min_size_px: int = 64,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Estimate the native Sentinel-2 window to crop for final full-resolution simulation.

    Geometry chain:
        proxy simulated pixels
            -- rectification_homography -->
        rectified proxy pixels, 4096x4096 frame
            -- H_rectified_to_real -->
        real PhiSat-2 pixels

    We invert the chain:
        real PhiSat-2 corners -> proxy simulated pixels -> native big Sentinel pixels

    The final returned window is expressed in pixel coordinates of sentinel_big_crop_path.
    """
    sentinel_big_crop_path = Path(sentinel_big_crop_path)
    proxy_simulated_path = Path(proxy_simulated_path)

    if not sentinel_big_crop_path.exists():
        raise FileNotFoundError(f"Sentinel big crop not found: {sentinel_big_crop_path}")

    if not proxy_simulated_path.exists():
        raise FileNotFoundError(f"Proxy simulated path not found: {proxy_simulated_path}")

    real_h, real_w = _event_shape(event)

    with rasterio.open(sentinel_big_crop_path) as src_big:
        big_w = int(src_big.width)
        big_h = int(src_big.height)
        big_crs = str(src_big.crs)
        big_transform = src_big.transform

    with rasterio.open(proxy_simulated_path) as src_proxy:
        proxy_w = int(src_proxy.width)
        proxy_h = int(src_proxy.height)
        proxy_crs = str(src_proxy.crs)

    H_rect = np.asarray(rectification_homography, dtype=np.float64)
    H_lg = np.asarray(H_rectified_to_real, dtype=np.float64)

    # proxy simulated -> real PhiSat-2
    H_proxy_to_real = H_lg @ H_rect
    H_proxy_to_real = H_proxy_to_real / H_proxy_to_real[2, 2]

    # real PhiSat-2 -> proxy simulated
    H_real_to_proxy = np.linalg.inv(H_proxy_to_real)
    H_real_to_proxy = H_real_to_proxy / H_real_to_proxy[2, 2]

    real_corners = np.array(
        [
            [0, 0],
            [real_w - 1, 0],
            [real_w - 1, real_h - 1],
            [0, real_h - 1],
        ],
        dtype=np.float32,
    )

    proxy_corners = _perspective_transform_points(real_corners, H_real_to_proxy)

    # The proxy simulated raster represents the same spatial extent as the big Sentinel crop,
    # but at simulated PhiSat-2 pixel spacing. Convert proxy pixels to big Sentinel pixels.
    scale_x_proxy_to_big = big_w / proxy_w
    scale_y_proxy_to_big = big_h / proxy_h

    big_corners = proxy_corners.copy()
    big_corners[:, 0] *= scale_x_proxy_to_big
    big_corners[:, 1] *= scale_y_proxy_to_big

    x_min = float(np.min(big_corners[:, 0]))
    y_min = float(np.min(big_corners[:, 1]))
    x_max = float(np.max(big_corners[:, 0]))
    y_max = float(np.max(big_corners[:, 1]))

    raw_w = x_max - x_min
    raw_h = y_max - y_min

    if raw_w < min_size_px or raw_h < min_size_px:
        raise RuntimeError(
            f"Estimated window too small: width={raw_w:.1f}, height={raw_h:.1f}"
        )

    mx = raw_w * margin_pct
    my = raw_h * margin_pct

    x0 = int(np.floor(x_min - mx))
    y0 = int(np.floor(y_min - my))
    x1 = int(np.ceil(x_max + mx))
    y1 = int(np.ceil(y_max + my))

    # Clamp to big crop bounds.
    x0_clamped = max(0, x0)
    y0_clamped = max(0, y0)
    x1_clamped = min(big_w, x1)
    y1_clamped = min(big_h, y1)

    width = int(x1_clamped - x0_clamped)
    height = int(y1_clamped - y0_clamped)

    if width < min_size_px or height < min_size_px:
        raise RuntimeError(
            f"Clamped window too small: width={width}, height={height}"
        )

    clipped = bool(
        x0_clamped != x0
        or y0_clamped != y0
        or x1_clamped != x1
        or y1_clamped != y1
    )

    window_native = {
        "x_min": int(x0_clamped),
        "y_min": int(y0_clamped),
        "width": width,
        "height": height,
        "x_max": int(x1_clamped),
        "y_max": int(y1_clamped),
    }

    result = {
        "status": "SUCCESS",
        "window_native": window_native,
        "margin_pct": float(margin_pct),
        "clipped_to_big_crop": clipped,
        "big_crop_shape": {"height": big_h, "width": big_w},
        "proxy_shape": {"height": proxy_h, "width": proxy_w},
        "real_shape": {"height": real_h, "width": real_w},
        "scale_x_proxy_to_big": float(scale_x_proxy_to_big),
        "scale_y_proxy_to_big": float(scale_y_proxy_to_big),
        "real_corners_px": real_corners.tolist(),
        "proxy_corners_px": proxy_corners.tolist(),
        "big_corners_px": big_corners.tolist(),
        "H_proxy_to_real": H_proxy_to_real.tolist(),
        "H_real_to_proxy": H_real_to_proxy.tolist(),
        "sentinel_big_crop_path": str(sentinel_big_crop_path),
        "proxy_simulated_path": str(proxy_simulated_path),
        "big_crs": big_crs,
        "proxy_crs": proxy_crs,
        "big_transform": tuple(big_transform)[:6],
    }

    if verbose:
        print("[Phiesta] Estimated final Sentinel window from proxy alignment")
        print(f"[Phiesta] big crop shape: {(big_h, big_w)}")
        print(f"[Phiesta] proxy shape: {(proxy_h, proxy_w)}")
        print(f"[Phiesta] real shape: {(real_h, real_w)}")
        print(f"[Phiesta] window_native: {window_native}")
        print(f"[Phiesta] clipped_to_big_crop: {clipped}")
        print(f"[Phiesta] big_corners_px:\n{big_corners}")

    return result
