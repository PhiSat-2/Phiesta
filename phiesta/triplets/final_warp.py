from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine


REAL_BAND_DESCRIPTIONS = (
    "PAN",
    "BLUE",
    "GREEN",
    "RED",
    "RED_EDGE_1",
    "RED_EDGE_2",
    "RED_EDGE_3",
    "NIR",
)


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _get_event_array(event: Any) -> np.ndarray:
    arr = getattr(event, "_arr", None)
    if arr is None:
        arr = getattr(event, "arr", None)
    if arr is None:
        raise AttributeError("Could not find array on event (_arr / arr).")
    return arr


def _get_event_meta(event: Any) -> dict:
    meta = getattr(event, "meta", None)
    if meta is None:
        meta = getattr(event, "_meta", None)
    if meta is None:
        raise AttributeError("Could not find metadata on event (meta / _meta).")
    return meta


def _as_h3x3(H: Any) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if H.shape == (2, 3):
        H = np.vstack([H, [0.0, 0.0, 1.0]])
    if H.shape != (3, 3):
        raise ValueError(f"Expected homography shape (3,3) or (2,3), got {H.shape}")
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def _ensure_affine(transform: Any) -> Affine | None:
    if transform is None:
        return None
    if isinstance(transform, Affine):
        return transform
    if isinstance(transform, (tuple, list)):
        if len(transform) >= 6:
            return Affine(*transform[:6])
    return None


def _warp_stack(
    data: np.ndarray,
    H: np.ndarray,
    out_w: int,
    out_h: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    out = np.zeros((data.shape[0], out_h, out_w), dtype=data.dtype)
    for i in range(data.shape[0]):
        out[i] = cv2.warpPerspective(
            data[i],
            H,
            (out_w, out_h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return out


def _save_tiff(
    path: Path,
    data: np.ndarray,
    crs: Any = None,
    transform: Affine | None = None,
    descriptions: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": int(data.shape[1]),
        "width": int(data.shape[2]),
        "count": int(data.shape[0]),
        "dtype": str(data.dtype),
        "compress": "deflate",
    }
    if crs is not None:
        profile["crs"] = crs
    if transform is not None:
        profile["transform"] = transform

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if descriptions is not None:
            dst.descriptions = descriptions


def warp_final_triplet_to_real_grid(
    event: Any,
    final_sentinel_crop_path: str | Path,
    final_simulated_path: str | Path,
    window_info: dict,
    output_dir: str | Path,
    overwrite: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Warp the final Sentinel crop and the final simulated PhiSat-2 product
    onto the real PhiSat-2 pixel grid (typically 4096x4096).

    Inputs:
    - event: real PhiSat-2 L1_event
    - final_sentinel_crop_path: 7-band final S2 crop at 10m
    - final_simulated_path: 8-band final simulated PhiSat-2 crop at 4.75m
    - window_info: dict returned by estimate_final_sentinel_window_from_proxy(...)
    - output_dir: where to save final warped triplet
    """

    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_sentinel_crop_path = _as_path(final_sentinel_crop_path)
    final_simulated_path = _as_path(final_simulated_path)

    real_arr = _get_event_array(event)
    real_meta = _get_event_meta(event)

    real_h = int(real_arr.shape[1])
    real_w = int(real_arr.shape[2])

    real_crs = real_meta.get("crs")
    real_transform = _ensure_affine(real_meta.get("transform"))

    if verbose:
        print("[Phiesta] Warping final triplet to real PhiSat-2 grid")
        print(f"[Phiesta] real shape: ({real_h}, {real_w})")
        print(f"[Phiesta] final_sentinel_crop: {final_sentinel_crop_path}")
        print(f"[Phiesta] final_simulated: {final_simulated_path}")

    with rasterio.open(final_sentinel_crop_path) as src_s2:
        s2_data = src_s2.read()
        s2_desc = src_s2.descriptions
        s2_h, s2_w = src_s2.height, src_s2.width

    with rasterio.open(final_simulated_path) as src_sim:
        sim_data = src_sim.read()
        sim_desc = src_sim.descriptions
        sim_h, sim_w = src_sim.height, src_sim.width

    H_real_to_proxy = _as_h3x3(window_info["H_real_to_proxy"])

    scale_x_proxy_to_big = float(window_info["scale_x_proxy_to_big"])
    scale_y_proxy_to_big = float(window_info["scale_y_proxy_to_big"])

    win = window_info["window_native"]
    x_min = float(win["x_min"])
    y_min = float(win["y_min"])
 
    real_corners = np.array(
        [
            [0.0, 0.0],
            [real_w - 1.0, 0.0],
            [real_w - 1.0, real_h - 1.0],
            [0.0, real_h - 1.0],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    # real -> proxy
    proxy_corners = cv2.perspectiveTransform(real_corners, H_real_to_proxy).reshape(-1, 2)

    # proxy -> big crop
    big_corners = proxy_corners.copy()
    big_corners[:, 0] *= scale_x_proxy_to_big
    big_corners[:, 1] *= scale_y_proxy_to_big

    # big crop -> final S2 crop local coordinates
    s2_local_corners = big_corners.copy()
    s2_local_corners[:, 0] -= x_min
    s2_local_corners[:, 1] -= y_min

    # final S2 crop -> final simulated local coordinates
    scale_x_s2_to_sim = sim_w / float(s2_w)
    scale_y_s2_to_sim = sim_h / float(s2_h)

    sim_local_corners = s2_local_corners.copy()
    sim_local_corners[:, 0] *= scale_x_s2_to_sim
    sim_local_corners[:, 1] *= scale_y_s2_to_sim

    dst_corners = np.array(
        [
            [0.0, 0.0],
            [real_w - 1.0, 0.0],
            [real_w - 1.0, real_h - 1.0],
            [0.0, real_h - 1.0],
        ],
        dtype=np.float32,
    )

    H_s2_to_real = cv2.getPerspectiveTransform(
        s2_local_corners.astype(np.float32),
        dst_corners.astype(np.float32),
    )

    H_sim_to_real = cv2.getPerspectiveTransform(
        sim_local_corners.astype(np.float32),
        dst_corners.astype(np.float32),
    )

    if verbose:
        print("[Phiesta] s2_local_corners:")
        print(s2_local_corners)
        print("[Phiesta] sim_local_corners:")
        print(sim_local_corners)
        print("[Phiesta] H_s2_to_real:")
        print(H_s2_to_real)
        print("[Phiesta] H_sim_to_real:")
        print(H_sim_to_real)

    s2_warped = _warp_stack(s2_data, H_s2_to_real, real_w, real_h, interpolation=cv2.INTER_LINEAR)
    sim_warped = _warp_stack(sim_data, H_sim_to_real, real_w, real_h, interpolation=cv2.INTER_LINEAR)

    real_out = output_dir / "phisat2_real_4096.tif"
    s2_out = output_dir / "sentinel_final_warped_to_real_4096.tif"
    sim_out = output_dir / "simulated_final_warped_to_real_4096.tif"
    meta_out = output_dir / "final_triplet_metadata.json"

    if overwrite or not real_out.exists():
        _save_tiff(
            real_out,
            real_arr,
            crs=real_crs,
            transform=real_transform,
            descriptions=REAL_BAND_DESCRIPTIONS,
        )

    _save_tiff(
        s2_out,
        s2_warped,
        crs=real_crs,
        transform=real_transform,
        descriptions=s2_desc,
    )

    _save_tiff(
        sim_out,
        sim_warped,
        crs=real_crs,
        transform=real_transform,
        descriptions=sim_desc,
    )

    result = {
        "status": "SUCCESS",
        "real_path": str(real_out),
        "sentinel_warped_path": str(s2_out),
        "simulated_warped_path": str(sim_out),
        "metadata_path": str(meta_out),
        "real_shape": {"height": real_h, "width": real_w},
        "sentinel_shape": {"height": s2_h, "width": s2_w},
        "simulated_shape": {"height": sim_h, "width": sim_w},
        "s2_local_corners": s2_local_corners.tolist(),
        "sim_local_corners": sim_local_corners.tolist(),
        "H_s2_to_real": H_s2_to_real.tolist(),
        "H_sim_to_real": H_sim_to_real.tolist(),
        "window_native": win,
    }

    meta_out.write_text(json.dumps(result, indent=2))

    if verbose:
        print("[Phiesta] Final triplet saved")
        print(f"[Phiesta] real:      {real_out}")
        print(f"[Phiesta] sentinel:  {s2_out}")
        print(f"[Phiesta] simulated: {sim_out}")
        print(f"[Phiesta] metadata:  {meta_out}")

    return result