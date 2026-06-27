from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd
import rasterio

from .proxy_alignment import (
    _normalize_u8,
    _resize_max,
    _to_lightglue_tensor,
    _load_extractor_and_matcher,
)


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _as_h3x3(H: Any) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if H.shape == (2, 3):
        H = np.vstack([H, [0.0, 0.0, 1.0]])
    if H.shape != (3, 3):
        raise ValueError(f"Expected homography shape (3,3), got {H.shape}")
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_triplet_paths(triplet: dict | str | Path) -> dict:
    """
    Resolve final triplet paths from either:
    - dict returned by build_full_sentinel_triplet
    - path to final_triplet_metadata.json
    - path to full_triplet_report.json
    """
    if isinstance(triplet, (str, Path)):
        path = Path(triplet)

        if path.name == "full_triplet_report.json":
            report = _load_json(path)
            paths = report.get("paths", report)
            metadata_path = paths.get("metadata")
        else:
            report = {}
            paths = {}
            metadata_path = str(path)

    elif isinstance(triplet, dict):
        report = triplet
        paths = triplet.get("paths", triplet)
        metadata_path = (
            triplet.get("metadata")
            or triplet.get("metadata_path")
            or paths.get("metadata")
        )
    else:
        raise TypeError(f"Unsupported triplet type: {type(triplet)}")

    if metadata_path is None:
        raise ValueError("Could not resolve final_triplet_metadata.json path.")

    metadata_path = Path(metadata_path)
    metadata = _load_json(metadata_path)

    final_dir = metadata_path.parent

    real_path = (
        paths.get("real")
        or metadata.get("real_path")
        or final_dir / "phisat2_real_4096.tif"
    )

    sentinel_path = (
        paths.get("sentinel")
        or paths.get("sentinel_warped_path")
        or metadata.get("sentinel_warped_path")
        or final_dir / "sentinel_final_warped_to_real_4096.tif"
    )

    simulated_path = (
        paths.get("simulated")
        or paths.get("simulated_warped_path")
        or metadata.get("simulated_warped_path")
        or final_dir / "simulated_final_warped_to_real_4096.tif"
    )

    out = {
        "metadata_path": metadata_path,
        "metadata": metadata,
        "report": report,
        "real_path": Path(real_path),
        "sentinel_warped_path": Path(sentinel_path),
        "simulated_warped_path": Path(simulated_path),
    }

    for key in ["real_path", "sentinel_warped_path", "simulated_warped_path"]:
        if not out[key].exists():
            raise FileNotFoundError(f"Missing {key}: {out[key]}")

    return out


def _read_stack(path: str | Path) -> tuple[np.ndarray, dict, tuple[str | None, ...]]:
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()
        descriptions = src.descriptions
    return data, profile, descriptions


def _find_band_index(descriptions: tuple[str | None, ...], candidates: list[str]) -> int | None:
    desc = [(d or "").upper() for d in descriptions]
    candidates = [c.upper() for c in candidates]

    for cand in candidates:
        for i, d in enumerate(desc):
            if d == cand:
                return i

    for cand in candidates:
        for i, d in enumerate(desc):
            if cand in d:
                return i

    return None


def _build_match_image(
    data: np.ndarray,
    descriptions: tuple[str | None, ...],
    preferred: str = "PAN",
) -> np.ndarray:
    """
    Build a grayscale image for feature matching.
    """
    preferred = preferred.upper()

    if preferred == "PAN":
        idx = _find_band_index(descriptions, ["PAN"])
        if idx is not None:
            return data[idx].astype(np.float32)

    if preferred == "NIR":
        idx = _find_band_index(descriptions, ["NIR", "B08_NIR", "NIR_BROAD"])
        if idx is not None:
            return data[idx].astype(np.float32)

    if preferred == "RED":
        idx = _find_band_index(descriptions, ["RED", "B04_RED"])
        if idx is not None:
            return data[idx].astype(np.float32)

    # Robust fallback: average RGB-like bands if available.
    idxs = []
    for names in [["RED", "B04_RED"], ["GREEN", "B03_GREEN"], ["BLUE", "B02_BLUE"]]:
        idx = _find_band_index(descriptions, names)
        if idx is not None:
            idxs.append(idx)

    if idxs:
        return np.mean(data[idxs].astype(np.float32), axis=0)

    return data[0].astype(np.float32)


def _get_band_or_none(
    data: np.ndarray,
    descriptions: tuple[str | None, ...],
    candidates: list[str],
) -> np.ndarray | None:
    idx = _find_band_index(descriptions, candidates)
    if idx is None:
        return None
    return data[idx].astype(np.float32)


def _quality_mask(
    img: np.ndarray,
    data: np.ndarray,
    descriptions: tuple[str | None, ...],
    *,
    mask_clouds: bool = True,
    mask_water: bool = True,
    mask_low_texture: bool = True,
    min_texture_percentile: float = 20.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Build a conservative match mask.

    This is heuristic. It removes nodata, very bright cloud-like regions,
    very low-texture regions, and some water-like areas when NIR is available.
    """
    img = np.asarray(img, dtype=np.float32)
    valid = np.isfinite(img) & (img > 0)

    if not valid.any():
        return np.zeros(img.shape, dtype=bool)

    mask = valid.copy()

    img_u8 = _normalize_u8(img)

    if mask_clouds:
        vals = img[valid]
        bright_thr = np.percentile(vals, 98.5)
        very_bright = img >= bright_thr

        # Clouds are often bright and locally smooth.
        blur = cv2.GaussianBlur(img_u8, (0, 0), 3)
        smooth_residual = cv2.absdiff(img_u8, blur)
        smooth_thr = np.percentile(smooth_residual[valid], 35)

        cloud_like = very_bright & (smooth_residual <= smooth_thr)
        mask &= ~cloud_like

    if mask_water:
        nir = _get_band_or_none(descriptions=descriptions, data=data, candidates=["NIR", "B08_NIR", "NIR_BROAD"])
        red = _get_band_or_none(descriptions=descriptions, data=data, candidates=["RED", "B04_RED"])

        if nir is not None:
            nir_valid = np.isfinite(nir) & (nir > 0)
            if nir_valid.any():
                nir_thr = np.percentile(nir[nir_valid], 15)
                water_like = nir <= nir_thr

                if red is not None:
                    red_valid = np.isfinite(red) & (red > 0)
                    if red_valid.any():
                        red_thr = np.percentile(red[red_valid], 40)
                        water_like &= red <= red_thr

                mask &= ~water_like

    if mask_low_texture:
        lap = cv2.Laplacian(img_u8, cv2.CV_32F, ksize=3)
        texture = np.abs(lap)

        tex_valid = texture[valid]
        if tex_valid.size > 0:
            tex_thr = np.percentile(tex_valid, min_texture_percentile)
            mask &= texture >= tex_thr

    # Clean small islands and holes.
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    mask = mask_u8 > 0

    if verbose:
        print(
            "[StrictGeoref] mask valid fraction:",
            float(mask.mean()),
            f"(clouds={mask_clouds}, water={mask_water}, low_texture={mask_low_texture})",
        )

    return mask


def _resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    small = cv2.resize(
        mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    return small.astype(bool)


def _sample_mask(mask: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    x = np.round(pts_xy[:, 0]).astype(int)
    y = np.round(pts_xy[:, 1]).astype(int)

    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    ok = np.zeros(len(pts_xy), dtype=bool)
    ok[inside] = mask[y[inside], x[inside]]
    return ok


def _reprojection_errors(src_pts: np.ndarray, dst_pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    src = np.concatenate([src_pts, np.ones((len(src_pts), 1), dtype=np.float64)], axis=1)
    pred = (H @ src.T).T
    pred = pred[:, :2] / pred[:, 2:3]
    return np.linalg.norm(pred - dst_pts, axis=1)


def _estimate_strict_homography(
    source_img: np.ndarray,
    real_img: np.ndarray,
    source_mask: np.ndarray | None = None,
    real_mask: np.ndarray | None = None,
    *,
    features: Literal["superpoint", "sift"] = "superpoint",
    max_keypoints: int = 12000,
    matching_max_side: int = 2600,
    ransac_thresh: float = 3.0,
    min_matches: int = 80,
    min_inliers: int = 150,
    min_inlier_ratio: float = 0.12,
    trim_quantile: float = 0.90,
    refinement_rounds: int = 2,
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Estimate residual H mapping source warped grid -> real grid.
    """
    source_u8 = _normalize_u8(source_img)
    import torch

    real_u8 = _normalize_u8(real_img)

    source_small, source_scale = _resize_max(source_u8, max_side=matching_max_side)
    real_small, real_scale = _resize_max(real_u8, max_side=matching_max_side)

    if source_mask is not None:
        source_mask_small = _resize_mask(source_mask, source_small.shape[:2])
        source_small = np.where(source_mask_small, source_small, 0).astype(np.uint8)
    else:
        source_mask_small = None

    if real_mask is not None:
        real_mask_small = _resize_mask(real_mask, real_small.shape[:2])
        real_small = np.where(real_mask_small, real_small, 0).astype(np.uint8)
    else:
        real_mask_small = None

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    extractor, matcher = _load_extractor_and_matcher(
        features=features,
        max_keypoints=max_keypoints,
        device=device,
    )

    from lightglue.utils import rbd

    source_tensor = _to_lightglue_tensor(source_small, device=device, features=features)
    real_tensor = _to_lightglue_tensor(real_small, device=device, features=features)

    with torch.no_grad():
        feats_source = extractor.extract(source_tensor)
        feats_real = extractor.extract(real_tensor)

        matches01 = matcher(
            {
                "image0": feats_source,
                "image1": feats_real,
            }
        )

    feats_source, feats_real, matches01 = [
        rbd(x) for x in [feats_source, feats_real, matches01]
    ]

    kpts_source = feats_source["keypoints"]
    kpts_real = feats_real["keypoints"]
    matches = matches01["matches"]

    if matches is None or len(matches) < min_matches:
        raise RuntimeError(
            f"Not enough LightGlue matches: {0 if matches is None else len(matches)} < {min_matches}"
        )

    pts_source_small = kpts_source[matches[:, 0]].detach().cpu().numpy()
    pts_real_small = kpts_real[matches[:, 1]].detach().cpu().numpy()

    keep = np.ones(len(pts_source_small), dtype=bool)

    if source_mask_small is not None:
        keep &= _sample_mask(source_mask_small, pts_source_small)

    if real_mask_small is not None:
        keep &= _sample_mask(real_mask_small, pts_real_small)

    pts_source_small = pts_source_small[keep]
    pts_real_small = pts_real_small[keep]

    if len(pts_source_small) < min_matches:
        raise RuntimeError(
            f"Not enough matches after quality masks: {len(pts_source_small)} < {min_matches}"
        )

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)

    H_small, inlier_mask = cv2.findHomography(
        pts_source_small.reshape(-1, 1, 2),
        pts_real_small.reshape(-1, 1, 2),
        method,
        ransac_thresh,
    )

    if H_small is None or inlier_mask is None:
        raise RuntimeError("cv2.findHomography failed.")

    inlier_mask = inlier_mask.ravel().astype(bool)

    for _ in range(int(refinement_rounds)):
        if inlier_mask.sum() < 4:
            break

        errors = _reprojection_errors(
            pts_source_small,
            pts_real_small,
            H_small,
        )

        inlier_errors = errors[inlier_mask]
        if len(inlier_errors) < 4:
            break

        trim_thr = np.quantile(inlier_errors, trim_quantile)
        refined_mask = inlier_mask & (errors <= max(trim_thr, ransac_thresh))

        if refined_mask.sum() < 4 or refined_mask.sum() == inlier_mask.sum():
            break

        H_refined, refined_cv_mask = cv2.findHomography(
            pts_source_small[refined_mask].reshape(-1, 1, 2),
            pts_real_small[refined_mask].reshape(-1, 1, 2),
            method,
            ransac_thresh,
        )

        if H_refined is None or refined_cv_mask is None:
            break

        H_small = H_refined
        new_mask = np.zeros_like(inlier_mask)
        refined_indices = np.where(refined_mask)[0]
        new_mask[refined_indices[refined_cv_mask.ravel().astype(bool)]] = True
        inlier_mask = new_mask

    errors_small = _reprojection_errors(pts_source_small, pts_real_small, H_small)

    n_matches_raw = int(len(matches))
    n_matches_filtered = int(len(pts_source_small))
    n_inliers = int(inlier_mask.sum())
    inlier_ratio = float(n_inliers / max(n_matches_filtered, 1))

    if n_inliers < min_inliers or inlier_ratio < min_inlier_ratio:
        raise RuntimeError(
            "Unreliable strict homography: "
            f"raw_matches={n_matches_raw}, filtered={n_matches_filtered}, "
            f"inliers={n_inliers}, ratio={inlier_ratio:.3f}"
        )

    S_source = np.array(
        [[source_scale, 0.0, 0.0], [0.0, source_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    S_real = np.array(
        [[real_scale, 0.0, 0.0], [0.0, real_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    H_full = np.linalg.inv(S_real) @ H_small @ S_source
    H_full = H_full / H_full[2, 2]

    pts_source_full = pts_source_small / source_scale
    pts_real_full = pts_real_small / real_scale
    errors_full = _reprojection_errors(pts_source_full, pts_real_full, H_full)

    inlier_errors = errors_full[inlier_mask]

    metrics = {
        "raw_matches": n_matches_raw,
        "filtered_matches": n_matches_filtered,
        "inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
        "error_median_px": float(np.median(inlier_errors)),
        "error_mean_px": float(np.mean(inlier_errors)),
        "error_p90_px": float(np.percentile(inlier_errors, 90)),
        "error_p95_px": float(np.percentile(inlier_errors, 95)),
        "error_max_px": float(np.max(inlier_errors)),
    }

    matches_df = pd.DataFrame({
        "source_x": pts_source_full[:, 0],
        "source_y": pts_source_full[:, 1],
        "real_x": pts_real_full[:, 0],
        "real_y": pts_real_full[:, 1],
        "is_inlier": inlier_mask,
        "reprojection_error_px": errors_full,
    })

    if verbose:
        print("[StrictGeoref] LightGlue strict residual alignment")
        print(f"[StrictGeoref] device={device}, features={features}")
        print(f"[StrictGeoref] raw matches={n_matches_raw}")
        print(f"[StrictGeoref] filtered matches={n_matches_filtered}")
        print(f"[StrictGeoref] inliers={n_inliers}, ratio={inlier_ratio:.3f}")
        print(
            "[StrictGeoref] reprojection error px: "
            f"median={metrics['error_median_px']:.3f}, "
            f"p90={metrics['error_p90_px']:.3f}, "
            f"p95={metrics['error_p95_px']:.3f}"
        )
        print(f"[StrictGeoref] H_residual:\n{H_full}")

    return {
        "status": "SUCCESS",
        "H_residual": H_full,
        "H_small": H_small,
        "metrics": metrics,
        "matches": matches_df,
        "features": features,
        "device": device,
        "matching_max_side": int(matching_max_side),
        "max_keypoints": int(max_keypoints),
        "ransac_thresh": float(ransac_thresh),
        "source_scale": float(source_scale),
        "real_scale": float(real_scale),
    }


def _warp_stack_with_profile(
    src_path: Path,
    H: np.ndarray,
    out_path: Path,
    *,
    overwrite: bool = True,
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return str(out_path)

    with rasterio.open(src_path) as src:
        data = src.read()
        profile = src.profile.copy()
        descriptions = src.descriptions

    out_h = int(profile["height"])
    out_w = int(profile["width"])

    out = np.zeros_like(data)

    for b in range(data.shape[0]):
        out[b] = cv2.warpPerspective(
            data[b],
            H,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out)
        if descriptions:
            dst.descriptions = descriptions

    return str(out_path)


def _save_overlay_preview(
    real_img: np.ndarray,
    source_img_before: np.ndarray,
    source_img_after: np.ndarray,
    out_png: Path,
) -> str:
    import matplotlib.pyplot as plt

    real = _normalize_u8(real_img)
    before = _normalize_u8(source_img_before)
    after = _normalize_u8(source_img_after)

    before_rgb = np.dstack([before, real, real])
    after_rgb = np.dstack([after, real, real])

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(before_rgb)
    axes[0].set_title("Before strict residual correction")
    axes[0].axis("off")

    axes[1].imshow(after_rgb)
    axes[1].set_title("After strict residual correction")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return str(out_png)


def refine_triplet_georeference_strict(
    triplet: dict | str | Path,
    *,
    source: Literal["simulated", "sentinel"] = "simulated",
    real_match_band: str = "PAN",
    source_match_band: str = "PAN",
    features: Literal["superpoint", "sift"] = "superpoint",
    max_keypoints: int = 12000,
    matching_max_side: int = 2600,
    ransac_thresh: float = 3.0,
    min_matches: int = 80,
    min_inliers: int = 150,
    min_inlier_ratio: float = 0.12,
    mask_clouds: bool = True,
    mask_water: bool = True,
    mask_low_texture: bool = True,
    min_texture_percentile: float = 20.0,
    refinement_rounds: int = 2,
    out_dir: str | Path | None = None,
    overwrite: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Refine final triplet georeference using whole-acquisition LightGlue matching.

    This estimates a residual homography from already-warped Sentinel/simulated
    triplet grid to the real PhiSat-2 grid.

    The default source is "simulated", because the simulated product comes from
    Sentinel-2 but is spectrally closer to PhiSat-2.
    """
    paths = _resolve_triplet_paths(triplet)
    metadata = paths["metadata"]

    final_dir = paths["metadata_path"].parent

    if out_dir is None:
        out_dir = final_dir.parent / "strict_georef"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_data, real_profile, real_desc = _read_stack(paths["real_path"])

    if source == "simulated":
        source_path = paths["simulated_warped_path"]
    elif source == "sentinel":
        source_path = paths["sentinel_warped_path"]
    else:
        raise ValueError("source must be 'simulated' or 'sentinel'.")

    source_data, source_profile, source_desc = _read_stack(source_path)

    real_img = _build_match_image(real_data, real_desc, preferred=real_match_band)
    source_img = _build_match_image(source_data, source_desc, preferred=source_match_band)

    real_mask = _quality_mask(
        real_img,
        real_data,
        real_desc,
        mask_clouds=mask_clouds,
        mask_water=mask_water,
        mask_low_texture=mask_low_texture,
        min_texture_percentile=min_texture_percentile,
        verbose=verbose,
    )

    source_mask = _quality_mask(
        source_img,
        source_data,
        source_desc,
        mask_clouds=mask_clouds,
        mask_water=mask_water,
        mask_low_texture=mask_low_texture,
        min_texture_percentile=min_texture_percentile,
        verbose=verbose,
    )

    estimated = _estimate_strict_homography(
        source_img=source_img,
        real_img=real_img,
        source_mask=source_mask,
        real_mask=real_mask,
        features=features,
        max_keypoints=max_keypoints,
        matching_max_side=matching_max_side,
        ransac_thresh=ransac_thresh,
        min_matches=min_matches,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
        refinement_rounds=refinement_rounds,
        device=None,
        verbose=verbose,
    )

    H_residual = _as_h3x3(estimated["H_residual"])

    strict_sentinel_path = out_dir / "sentinel_warped_to_real_strict.tif"
    strict_simulated_path = out_dir / "simulated_warped_to_real_strict.tif"

    _warp_stack_with_profile(
        paths["sentinel_warped_path"],
        H_residual,
        strict_sentinel_path,
        overwrite=overwrite,
    )
    _warp_stack_with_profile(
        paths["simulated_warped_path"],
        H_residual,
        strict_simulated_path,
        overwrite=overwrite,
    )

    H_s2_to_real = metadata.get("H_s2_to_real")
    H_sim_to_real = metadata.get("H_sim_to_real")

    H_s2_to_real_strict = None
    H_sim_to_real_strict = None

    if H_s2_to_real is not None:
        H_s2_to_real_strict = H_residual @ _as_h3x3(H_s2_to_real)
        H_s2_to_real_strict = H_s2_to_real_strict / H_s2_to_real_strict[2, 2]

    if H_sim_to_real is not None:
        H_sim_to_real_strict = H_residual @ _as_h3x3(H_sim_to_real)
        H_sim_to_real_strict = H_sim_to_real_strict / H_sim_to_real_strict[2, 2]

    # Preview uses the selected source.
    warped_source_after = cv2.warpPerspective(
        source_img,
        H_residual,
        (source_img.shape[1], source_img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    preview_path = out_dir / f"strict_overlay_{source}.png"
    _save_overlay_preview(real_img, source_img, warped_source_after, preview_path)

    matches_path = out_dir / f"matches_{source}_strict.csv"
    inliers_path = out_dir / f"matches_{source}_strict_inliers.csv"

    matches_df = estimated["matches"]
    matches_df.to_csv(matches_path, index=False)
    matches_df[matches_df["is_inlier"]].to_csv(inliers_path, index=False)

    report = {
        "status": "SUCCESS",
        "mode": "strict_georef_residual",
        "source": source,
        "features": features,
        "matching_max_side": int(matching_max_side),
        "max_keypoints": int(max_keypoints),
        "ransac_thresh": float(ransac_thresh),
        "quality_masks": {
            "mask_clouds": bool(mask_clouds),
            "mask_water": bool(mask_water),
            "mask_low_texture": bool(mask_low_texture),
            "min_texture_percentile": float(min_texture_percentile),
            "real_valid_fraction": float(real_mask.mean()),
            "source_valid_fraction": float(source_mask.mean()),
        },
        "metrics": estimated["metrics"],
        "paths": {
            "real": str(paths["real_path"]),
            "sentinel_warped_input": str(paths["sentinel_warped_path"]),
            "simulated_warped_input": str(paths["simulated_warped_path"]),
            "sentinel_strict": str(strict_sentinel_path),
            "simulated_strict": str(strict_simulated_path),
            "matches": str(matches_path),
            "inliers": str(inliers_path),
            "preview": str(preview_path),
        },
        "H_residual_source_to_real": H_residual.tolist(),
        "H_s2_to_real_strict": H_s2_to_real_strict.tolist() if H_s2_to_real_strict is not None else None,
        "H_sim_to_real_strict": H_sim_to_real_strict.tolist() if H_sim_to_real_strict is not None else None,
    }

    report_path = out_dir / f"strict_georef_report_{source}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print("[StrictGeoref] saved report:", report_path)
        print("[StrictGeoref] saved preview:", preview_path)
        print("[StrictGeoref] strict sentinel:", strict_sentinel_path)
        print("[StrictGeoref] strict simulated:", strict_simulated_path)

    return {
        "status": "SUCCESS",
        "report": report,
        "report_path": str(report_path),
        "paths": report["paths"],
        "metrics": estimated["metrics"],
        "H_residual_source_to_real": H_residual.tolist(),
        "H_s2_to_real_strict": report["H_s2_to_real_strict"],
        "H_sim_to_real_strict": report["H_sim_to_real_strict"],
    }


def inspect_strict_georef(
    strict_or_report,
    *,
    print_report: bool = True,
) -> dict:
    """
    Inspect a strict georeference refinement report.

    Args:
        strict_or_report: either:
            - dict returned by refine_triplet_georeference_strict(...)
            - path to strict_georef_report_*.json
        print_report: print a compact human-readable summary.

    Returns:
        Compact summary dictionary.
    """
    if isinstance(strict_or_report, (str, Path)):
        report_path = Path(strict_or_report)
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    elif isinstance(strict_or_report, dict):
        report = strict_or_report.get("report", strict_or_report)
        report_path = strict_or_report.get("report_path")
    else:
        raise TypeError(f"Unsupported input type: {type(strict_or_report)}")

    metrics = report.get("metrics", {})
    H = np.asarray(report.get("H_residual_source_to_real"), dtype=np.float64)

    if H.shape != (3, 3):
        raise ValueError("Report does not contain a valid H_residual_source_to_real.")

    tx = float(H[0, 2])
    ty = float(H[1, 2])
    sx = float(np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2))
    sy = float(np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2))
    shear_like = float(abs(H[0, 1]) + abs(H[1, 0]))
    perspective_like = float(abs(H[2, 0]) + abs(H[2, 1]))

    inliers = int(metrics.get("inliers", 0))
    median = float(metrics.get("error_median_px", np.inf))
    p90 = float(metrics.get("error_p90_px", np.inf))
    ratio = float(metrics.get("inlier_ratio", 0.0))

    if inliers >= 200 and median <= 4.0 and p90 <= 6.0:
        quality = "GOOD"
    elif inliers >= 100 and median <= 5.0 and p90 <= 8.0:
        quality = "OK"
    else:
        quality = "RISKY"

    summary = {
        "quality": quality,
        "source": report.get("source"),
        "features": report.get("features"),
        "inliers": inliers,
        "inlier_ratio": ratio,
        "error_median_px": median,
        "error_p90_px": p90,
        "error_p95_px": float(metrics.get("error_p95_px", np.inf)),
        "translation_px": {"x": tx, "y": ty},
        "scale_approx": {"x": sx, "y": sy},
        "shear_like": shear_like,
        "perspective_like": perspective_like,
        "preview": report.get("paths", {}).get("preview"),
        "sentinel_strict": report.get("paths", {}).get("sentinel_strict"),
        "simulated_strict": report.get("paths", {}).get("simulated_strict"),
    }

    if print_report:
        print("\n" + "=" * 80)
        print("Strict georeference inspection")
        print("=" * 80)
        print(f"quality              : {quality}")
        print(f"source               : {summary['source']}")
        print(f"features             : {summary['features']}")
        print(f"inliers              : {inliers}")
        print(f"inlier_ratio          : {ratio:.3f}")
        print(f"median error          : {median:.3f} px")
        print(f"p90 error             : {p90:.3f} px")
        print(f"p95 error             : {summary['error_p95_px']:.3f} px")
        print(f"translation           : x={tx:.2f}px, y={ty:.2f}px")
        print(f"scale approx          : x={sx:.5f}, y={sy:.5f}")
        print(f"shear-like            : {shear_like:.6f}")
        print(f"perspective-like      : {perspective_like:.8f}")
        print(f"preview               : {summary['preview']}")
        print(f"sentinel strict       : {summary['sentinel_strict']}")
        print(f"simulated strict      : {summary['simulated_strict']}")

    return summary


def _resolve_final_sentinel_crop_for_georef(triplet, metadata_path=None, metadata=None) -> Path:
    """
    Resolve the final Sentinel crop used before warping to the real grid.
    """
    metadata = metadata or {}
    paths = triplet.get("paths", triplet) if isinstance(triplet, dict) else {}

    candidates = [
        paths.get("final_sentinel_crop"),
        triplet.get("final_sentinel_crop") if isinstance(triplet, dict) else None,
        metadata.get("final_sentinel_crop"),
        metadata.get("final_sentinel_crop_path"),
        metadata.get("paths", {}).get("final_sentinel_crop") if isinstance(metadata.get("paths"), dict) else None,
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        p = Path(str(candidate))
        if p.exists():
            return p

    if metadata_path is not None:
        root = Path(metadata_path).parent.parent
        matches = sorted((root / "sentinel_final").glob("*_s2b_final_crop_7bands.tif"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Could not resolve final Sentinel crop. Pass the triplet dict returned by "
        "build_full_sentinel_triplet(), not only the strict report."
    )


def georef_from_strict_result(
    triplet,
    strict_result,
    *,
    print_report: bool = False,
) -> dict:
    """
    Convert a strict georef result into a directly usable georeference object.

    Returns corners/polygon in lon-lat, homographies, quality metrics and output paths.
    """
    from pyproj import Transformer

    paths = _resolve_triplet_paths(triplet)
    metadata = paths["metadata"]
    metadata_path = paths["metadata_path"]

    report = strict_result.get("report", strict_result)

    H_s2_to_real = report.get("H_s2_to_real_strict")
    if H_s2_to_real is None:
        raise ValueError("Strict result does not contain H_s2_to_real_strict.")

    H_s2_to_real = _as_h3x3(H_s2_to_real)
    H_real_to_s2 = np.linalg.inv(H_s2_to_real)
    H_real_to_s2 = H_real_to_s2 / H_real_to_s2[2, 2]

    final_sentinel_crop = _resolve_final_sentinel_crop_for_georef(
        triplet,
        metadata_path=metadata_path,
        metadata=metadata,
    )

    real_path = Path(report.get("paths", {}).get("real") or paths["real_path"])

    with rasterio.open(real_path) as real_src:
        real_w = real_src.width
        real_h = real_src.height

    real_corners = np.array(
        [
            [0.0, 0.0],
            [real_w - 1.0, 0.0],
            [real_w - 1.0, real_h - 1.0],
            [0.0, real_h - 1.0],
        ],
        dtype=np.float64,
    )

    center_px = np.array(
        [[(real_w - 1.0) / 2.0, (real_h - 1.0) / 2.0]],
        dtype=np.float64,
    )

    def project_real_to_s2(points_xy):
        pts = np.concatenate(
            [points_xy, np.ones((len(points_xy), 1), dtype=np.float64)],
            axis=1,
        )
        out = (H_real_to_s2 @ pts.T).T
        out = out[:, :2] / out[:, 2:3]
        return out

    s2_corners = project_real_to_s2(real_corners)
    s2_center = project_real_to_s2(center_px)

    with rasterio.open(final_sentinel_crop) as s2_src:
        transform = s2_src.transform
        crs = s2_src.crs

        map_x, map_y = transform * (s2_corners[:, 0], s2_corners[:, 1])
        center_x, center_y = transform * (s2_center[:, 0], s2_center[:, 1])

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(map_x, map_y)
        center_lon, center_lat = transformer.transform(center_x, center_y)

    corners_lonlat = [[float(x), float(y)] for x, y in zip(lon, lat)]
    polygon_lonlat = corners_lonlat + [corners_lonlat[0]]

    inspection = inspect_strict_georef(strict_result, print_report=print_report)

    product_id = (
        metadata.get("product_id")
        or report.get("product_id")
        or Path(real_path).parts[-3] if len(Path(real_path).parts) >= 3 else None
    )

    georef = {
        "status": "SUCCESS",
        "method": "sentinel_strict",
        "product_id": str(product_id) if product_id is not None else None,
        "quality": inspection["quality"],
        "corners_lonlat": corners_lonlat,
        "corners_latlon": [[lat, lon] for lon, lat in corners_lonlat],
        "center_lonlat": [float(center_lon[0]), float(center_lat[0])],
        "center_latlon": [float(center_lat[0]), float(center_lon[0])],
        "polygon_geojson": {
            "type": "Polygon",
            "coordinates": [polygon_lonlat],
        },
        "real_shape": {
            "width": int(real_w),
            "height": int(real_h),
        },
        "H_s2_to_real": H_s2_to_real.tolist(),
        "H_real_to_s2": H_real_to_s2.tolist(),
        "metrics": report.get("metrics", {}),
        "strict_report": report,
        "paths": {
            "real": str(real_path),
            "final_sentinel_crop": str(final_sentinel_crop),
            "preview": report.get("paths", {}).get("preview"),
            "sentinel_strict": report.get("paths", {}).get("sentinel_strict"),
            "simulated_strict": report.get("paths", {}).get("simulated_strict"),
            "matches": report.get("paths", {}).get("matches"),
            "inliers": report.get("paths", {}).get("inliers"),
        },
    }

    return georef


def get_strict_georef_from_triplet(
    triplet,
    *,
    source: str = "simulated",
    print_report: bool = False,
    **strict_kwargs,
) -> dict:
    """
    Run strict georeference refinement and return a directly usable georef object.
    """
    strict = refine_triplet_georeference_strict(
        triplet,
        source=source,
        **strict_kwargs,
    )

    return georef_from_strict_result(
        triplet,
        strict,
        print_report=print_report,
    )
