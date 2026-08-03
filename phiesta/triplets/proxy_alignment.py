from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import time
import warnings

import cv2
import numpy as np
import rasterio

from .simulation import simulate_phisat2_from_sentinel_crop
from .rectification import rectify_simulated_catalog_crop


def _normalize_u8(x: np.ndarray, percentiles=(2, 98)) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x)

    if not valid.any():
        return np.zeros(x.shape, dtype=np.uint8)

    lo, hi = np.percentile(x[valid], percentiles)
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)

    y = np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return (255.0 * y).astype(np.uint8)


def _resize_max(img: np.ndarray, max_side: int = 1800) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(float(max_side) / float(max(h, w)), 1.0)

    if scale >= 1.0:
        return img, 1.0

    resized = cv2.resize(
        img,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _to_lightglue_tensor(
    img_u8: np.ndarray,
    device: str,
    features: str,
) -> torch.Tensor:
    import torch

    img = img_u8.astype(np.float32) / 255.0

    if features.lower() == "sift":
        # LightGlue SIFT extractor follows the common examples with shape [B,H,W]
        return torch.from_numpy(img)[None].to(device)

    # SuperPoint expects [B,C,H,W]
    return torch.from_numpy(img)[None, None].to(device)


def _load_extractor_and_matcher(
    features: Literal["superpoint", "sift"],
    max_keypoints: int,
    device: str,
):
    from lightglue import LightGlue, SuperPoint, SIFT

    if features == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)
    elif features == "sift":
        extractor = SIFT(max_num_keypoints=max_keypoints).eval().to(device)
        matcher = LightGlue(features="sift").eval().to(device)
    else:
        raise ValueError(f"Unsupported LightGlue features: {features!r}")

    return extractor, matcher


def _estimate_lightglue_homography(
    real_img: np.ndarray,
    sim_img: np.ndarray,
    features: Literal["superpoint", "sift"] = "superpoint",
    max_keypoints: int = 8000,
    matching_max_side: int = 1800,
    ransac_thresh: float = 5.0,
    min_inliers: int = 30,
    min_inlier_ratio: float = 0.25,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Estimate H mapping simulated rectified image pixels -> real image pixels.
    """
    import torch

    real_u8 = _normalize_u8(real_img)
    sim_u8 = _normalize_u8(sim_img)

    real_small, real_scale = _resize_max(real_u8, max_side=matching_max_side)
    sim_small, sim_scale = _resize_max(sim_u8, max_side=matching_max_side)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    extractor, matcher = _load_extractor_and_matcher(
        features=features,
        max_keypoints=max_keypoints,
        device=device,
    )

    from lightglue.utils import rbd

    real_tensor = _to_lightglue_tensor(real_small, device=device, features=features)
    sim_tensor = _to_lightglue_tensor(sim_small, device=device, features=features)

    with torch.no_grad():
        feats_sim = extractor.extract(sim_tensor)
        feats_real = extractor.extract(real_tensor)

        matches01 = matcher(
            {
                "image0": feats_sim,
                "image1": feats_real,
            }
        )

    feats_sim, feats_real, matches01 = [
        rbd(x) for x in [feats_sim, feats_real, matches01]
    ]

    kpts_sim = feats_sim["keypoints"]
    kpts_real = feats_real["keypoints"]
    matches = matches01["matches"]

    if matches is None or len(matches) < 4:
        raise RuntimeError(f"Not enough LightGlue matches: {0 if matches is None else len(matches)}")

    pts_sim = kpts_sim[matches[:, 0]].detach().cpu().numpy()
    pts_real = kpts_real[matches[:, 1]].detach().cpu().numpy()

    H_small, mask = cv2.findHomography(
        pts_sim.reshape(-1, 1, 2),
        pts_real.reshape(-1, 1, 2),
        cv2.RANSAC,
        ransac_thresh,
    )

    if H_small is None or mask is None:
        raise RuntimeError("cv2.findHomography failed.")

    mask = mask.ravel().astype(bool)
    n_matches = int(len(matches))
    n_inliers = int(mask.sum())
    inlier_ratio = float(n_inliers / max(n_matches, 1))

    if n_inliers < min_inliers or inlier_ratio < min_inlier_ratio:
        raise RuntimeError(
            "Unreliable LightGlue homography: "
            f"matches={n_matches}, inliers={n_inliers}, ratio={inlier_ratio:.3f}"
        )

    # Convert homography from resized coordinates to full-resolution coordinates.
    # small = S * full
    S_sim = np.array(
        [[sim_scale, 0.0, 0.0], [0.0, sim_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    S_real = np.array(
        [[real_scale, 0.0, 0.0], [0.0, real_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    H_full = np.linalg.inv(S_real) @ H_small @ S_sim
    H_full = H_full / H_full[2, 2]

    if verbose:
        print("[Phiesta] LightGlue proxy alignment")
        print(f"[Phiesta] device={device}, features={features}")
        print(f"[Phiesta] keypoints sim={len(kpts_sim)}, real={len(kpts_real)}")
        print(f"[Phiesta] matches={n_matches}, inliers={n_inliers}, ratio={inlier_ratio:.3f}")
        print(f"[Phiesta] H_full:\n{H_full}")

    return {
        "status": "SUCCESS",
        "features": features,
        "device": device,
        "matching_max_side": int(matching_max_side),
        "max_keypoints": int(max_keypoints),
        "ransac_thresh": float(ransac_thresh),
        "num_keypoints_sim": int(len(kpts_sim)),
        "num_keypoints_real": int(len(kpts_real)),
        "matches": n_matches,
        "inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
        "real_scale": float(real_scale),
        "sim_scale": float(sim_scale),
        "H_small": H_small.tolist(),
        "H_rectified_to_real": H_full.tolist(),
    }


def _warp_rectified_proxy_preview(
    rectified_path: str | Path,
    H_rectified_to_real: np.ndarray,
    output_path: str | Path,
    real_shape: tuple[int, int],
    verbose: bool = True,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_h, out_w = real_shape

    with rasterio.open(rectified_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        descriptions = src.descriptions

    warped = np.zeros((data.shape[0], out_h, out_w), dtype=np.float32)

    for b in range(data.shape[0]):
        warped[b] = cv2.warpPerspective(
            data[b],
            H_rectified_to_real,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    profile.update(
        height=out_h,
        width=out_w,
        count=warped.shape[0],
        dtype="float32",
        compress="deflate",
        bigtiff="if_safer",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(warped)
        if descriptions:
            dst.descriptions = descriptions

    if verbose:
        print(f"[Phiesta] Proxy aligned preview saved: {output_path}")

    return str(output_path)


def run_proxy_alignment(
    event: Any,
    triplet: Any,
    snr_psf_method: str = None,
    proxy_target_size: tuple[int, int] = (1024, 1024),
    matching_max_side: int = 1800,
    features: Literal["superpoint", "sift"] = "superpoint",
    max_keypoints: int = 8000,
    real_band: str = "PAN",
    simulated_band_index: int = 4,
    flip_horizontal: bool = True,
    overwrite: bool = True,
    save_aligned_preview: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run low-resolution proxy simulation + catalog rectification + LightGlue alignment.

    This is for localization/R&D, not final triplet production.
    """
    t0 = time.time()

    if triplet.sentinel_crop is None:
        raise ValueError("triplet.sentinel_crop is None. Build the big Sentinel crop first.")

    base_sim_dir = Path(triplet.paths.simulated_dir)
    base_aligned_dir = Path(triplet.paths.aligned_dir)

    proxy_sim_dir = base_sim_dir / "proxy"
    proxy_rect_dir = base_aligned_dir / "proxy_rectified"
    proxy_preview_dir = base_aligned_dir / "proxy_preview"

    if verbose:
        print("[Phiesta] Running proxy simulation")

    proxy_sim = simulate_phisat2_from_sentinel_crop(
        crop=triplet.sentinel_crop,
        output_dir=proxy_sim_dir,
        snr_psf_method=snr_psf_method,
        processing_level="L1C",
        workers=1,
        overwrite=overwrite,
        target_size=proxy_target_size,
        verbose=verbose,
    )

    if verbose:
        print("[Phiesta] Rectifying proxy simulation to catalog footprint")

    rect = rectify_simulated_catalog_crop(
        event=event,
        simulated_path=proxy_sim.simulated_path,
        output_dir=proxy_rect_dir,
        flip_horizontal=flip_horizontal,
        overwrite=overwrite,
        verbose=verbose,
    )

    real_img = event.get_band(real_band).astype(np.float32)

    with rasterio.open(rect["rectified_path"]) as src:
        sim_img = src.read(simulated_band_index).astype(np.float32)

    lg = _estimate_lightglue_homography(
        real_img=real_img,
        sim_img=sim_img,
        features=features,
        max_keypoints=max_keypoints,
        matching_max_side=matching_max_side,
        verbose=verbose,
    )

    H_full = np.asarray(lg["H_rectified_to_real"], dtype=np.float64)

    preview_path = None
    if save_aligned_preview:
        real_shape = real_img.shape
        preview_path = _warp_rectified_proxy_preview(
            rectified_path=rect["rectified_path"],
            H_rectified_to_real=H_full,
            output_path=proxy_preview_dir / f"{Path(rect['rectified_path']).stem}_lightglue_preview.tif",
            real_shape=real_shape,
            verbose=verbose,
        )

    elapsed = time.time() - t0

    return {
        "status": "SUCCESS",
        "proxy_target_size": tuple(proxy_target_size),
        "proxy_simulated_path": proxy_sim.simulated_path,
        "rectified_proxy_path": rect["rectified_path"],
        "aligned_proxy_preview_path": preview_path,
        "rectification_homography": rect["homography"],
        "rectification_src_quad_px": rect["src_quad_px"],
        "H_rectified_to_real": lg["H_rectified_to_real"],
        "matches": lg["matches"],
        "inliers": lg["inliers"],
        "inlier_ratio": lg["inlier_ratio"],
        "features": lg["features"],
        "matching_max_side": lg["matching_max_side"],
        "elapsed_s": elapsed,
    }
