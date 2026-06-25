from __future__ import annotations

from typing import Dict, Any, Tuple, List

import numpy as np
try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


def _require_torch() -> None:
    if torch is None or F is None:
        raise ImportError(
            "Torch is required for L0/L1 registration utilities. "
            "Install it separately, e.g. `pip install torch`."
        )
    
def prep_for_phase_corr(img: np.ndarray) -> torch.Tensor:
    """
    Prepare one 2D image for phase-correlation-based registration.

    The image is converted to `float32`, standardized to zero mean and unit
    variance, and reshaped to `(1, 1, H, W)` for use with the registration
    functions.

    Args:
        img: Input 2D image.

    Returns:
        A torch tensor with shape `(1, 1, H, W)`.
    """
    _require_torch()
    x = torch.tensor(img, dtype=torch.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


def phase_correlation_shift(src, tgt, eps=1e-8, max_shifts=None):
    """
    Estimate the translation that aligns a target image to a reference image using
    phase correlation.

    Args:
        src: Reference tensor with shape `(B, 1, H, W)`.
        tgt: Target tensor with shape `(B, 1, H, W)` to be shifted toward `src`.
        eps: Small positive constant for numerical stability in Fourier-domain
            normalization.
        max_shifts: Optional `(max_dy, max_dx)` clamp applied to the estimated
            shifts.

    Returns:
        A tensor of shape `(B, 2)` containing shifts `(dy, dx)` to apply to `tgt`
        so that it aligns with `src`.
    """
    _require_torch()
    B, C, H, W = src.shape

    fs = torch.fft.rfft2(src)
    ft = torch.fft.rfft2(tgt)

    R = fs * torch.conj(ft)
    R = R / (torch.abs(R) + eps)
    r = torch.fft.irfft2(R, s=(H, W))

    maxidx = r.view(B, -1).argmax(dim=1).long()
    py = (maxidx // W).float()
    px = (maxidx % W).float()

    dy = py.clone()
    dx = px.clone()

    dy[dy > H // 2] -= H
    dx[dx > W // 2] -= W

    if max_shifts is not None:
        dy = torch.clamp(dy, min=-max_shifts[0], max=max_shifts[0])
        dx = torch.clamp(dx, min=-max_shifts[1], max=max_shifts[1])

    return torch.stack([dy, dx], dim=1)


def warp_by_shift(img, shift):
    """
    Warp a batch of images by pixel translations using bilinear sampling.

    Args:
        img: Tensor with shape `(B, C, H, W)`.
        shift: Tensor with shape `(B, 2)` containing shifts `(dy, dx)` in pixels.

    Returns:
        A tensor with the same shape as `img`, shifted according to `shift`.
    """
    _require_torch()
    B, C, H, W = img.shape
    dy = shift[:, 0]
    dx = shift[:, 1]

    yy = torch.linspace(-1, 1, H, device=img.device)
    xx = torch.linspace(-1, 1, W, device=img.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    grid = torch.stack([grid_x, grid_y], dim=-1)[None].repeat(B, 1, 1, 1).clone()

    grid[..., 0] = grid[..., 0] - (dx / ((W - 1) / 2))[:, None, None]
    grid[..., 1] = grid[..., 1] - (dy / ((H - 1) / 2))[:, None, None]

    out = F.grid_sample(
        img,
        grid,
        align_corners=True,
        mode="bilinear",
        padding_mode="border",
    )
    return out


def warp_np_by_shift(img: np.ndarray, shift_dy_dx: np.ndarray) -> np.ndarray:
    """
    Warp one 2D NumPy image by a pixel translation.

    Args:
        img: Input 2D image.
        shift_dy_dx: Shift `(dy, dx)` in pixels.

    Returns:
        The shifted image as a 2D NumPy array.
    """
    _require_torch()
    t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    shift_dy_dx = np.asarray(shift_dy_dx, dtype=np.float32)
    s = torch.from_numpy(shift_dy_dx[None, :])
    out = warp_by_shift(t, s).squeeze(0).squeeze(0).cpu().numpy()
    return out


def top_crop_to_match_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """
    Crop the top part of an image so that its height matches a target value.

    This helper is used for L0/L1 pairing in cases where the raw L0 image contains
    extra rows below the nominal L1 footprint.

    Args:
        img: Input 2D image.
        target_h: Desired output height.

    Returns:
        The top-cropped image.

    Raises:
        ValueError: If the input image is smaller than the requested height.
    """
    H, W = img.shape
    if H < target_h:
        raise ValueError(f"Image height {H} is smaller than target height {target_h}")
    return img[:target_h, :]


def register_l0_to_l1_space(
    l0_event,
    l1_event,
    master_band: int = 7,
    max_shifts: Tuple[int, int] = (300, 300),
):
    """
    Register one L0 event into the reference space of an L1 event.

    The current pipeline follows two steps:
    1. align the L0 master band to the corresponding L1 master band,
    2. align every L0 band directly to the already-warped master reference.

    If the L0 image is taller than the L1 image, a top crop is applied before
    registration so that both master bands share the same height.

    Args:
        l0_event: Source L0 event in native raw space.
        l1_event: Target L1 event defining the reference space.
        master_band: Band used as the master reference on both sides.
        max_shifts: Optional `(max_dy, max_dx)` clamp applied to all estimated
            translations.

    Returns:
        A new :class:`L0_event` whose array is expressed in the L1 reference space.
        The returned metadata contains a `registration_info` dictionary describing
        the applied transformations.
    """
    l0_master_full = l0_event.get_band(master_band)
    l1_master = l1_event.get_band(master_band)

    top_crop_rows = 0

    # Some L0 products contain extra rows below the nominal L1 footprint.
    # In the current pipeline, these cases are handled by keeping the top part only.

    if l0_master_full.shape[1] != l1_master.shape[1]:
        raise ValueError(
            f"L0/L1 widths differ: {l0_master_full.shape} vs {l1_master.shape}"
        )

    if l0_master_full.shape[0] != l1_master.shape[0]:
        if l0_master_full.shape[0] > l1_master.shape[0]:
            top_crop_rows = l0_master_full.shape[0] - l1_master.shape[0]
            l0_master = top_crop_to_match_height(l0_master_full, l1_master.shape[0])
        else:
            raise ValueError(
                f"L0 height smaller than L1 height: {l0_master_full.shape} vs {l1_master.shape}"
            )
    else:
        l0_master = l0_master_full

    src = prep_for_phase_corr(l1_master)
    tgt = prep_for_phase_corr(l0_master)
    master_shift = phase_correlation_shift(src, tgt, max_shifts=max_shifts)[0].cpu().numpy().astype(np.float32)

    l0_master_warped = warp_np_by_shift(l0_master, master_shift)

    aligned_bands: List[np.ndarray] = []
    band_shifts_to_ref: Dict[str, List[float]] = {}

    ref_np = l0_master_warped

    # Each L0 band is aligned directly to the already-warped master reference,
    # so the final cube is expressed in the common L1 reference space.

    for i in range(l0_event.as_numpy().shape[0]):
        band_np = l0_event.get_band(i)
        if top_crop_rows > 0:
            band_np = band_np[:l1_master.shape[0], :]

        if band_np.shape[0] != ref_np.shape[0]:
            band_np = top_crop_to_match_height(band_np, ref_np.shape[0])

        if i == master_band:
            warped_np = ref_np
            rel_shift = np.array([0.0, 0.0], dtype=np.float32)
        else:
            src_i = prep_for_phase_corr(ref_np)
            tgt_i = prep_for_phase_corr(band_np)
            rel_shift = phase_correlation_shift(src_i, tgt_i, max_shifts=max_shifts)[0].cpu().numpy().astype(np.float32)
            warped_np = warp_np_by_shift(band_np, rel_shift)

        aligned_bands.append(warped_np)
        band_shifts_to_ref[str(i)] = rel_shift.tolist()

    aligned_cube = np.stack(aligned_bands, axis=0)

    reg_info = {
        "method": "phase_correlation_master_then_bands",
        "master_band": int(master_band),
        "reference_space": f"L1_band_{master_band}",
        "master_shift_l0_to_l1_dy_dx": master_shift.tolist(),
        "band_shifts_to_ref_dy_dx": band_shifts_to_ref,
        "source_l1_path": l1_event.get_meta().get("path", None),
        "l0_original_shape": list(l0_event.as_numpy().shape),
        "l0_registered_shape": list(aligned_cube.shape),
        "l1_shape": list(l1_event.as_numpy().shape),
        "l0_pre_crop_strategy": "top_crop" if top_crop_rows > 0 else "none",
        "l0_top_crop_removed_rows": int(top_crop_rows),
    }

    meta_updates = {
        "count": int(aligned_cube.shape[0]),
        "height": int(aligned_cube.shape[1]),
        "width": int(aligned_cube.shape[2]),
        "dtype": str(aligned_cube.dtype),
        "native_space": "L0_in_L1_space",
        "parent_native_space": l0_event.meta.get("native_space", "L0_native"),
        "registration_info": reg_info,
        "crs": l1_event.get_meta().get("crs"),
        "transform": l1_event.get_meta().get("transform"),
        "bounds": l1_event.get_meta().get("bounds"),
    }

    l0_registered = l0_event.with_array(aligned_cube.astype(np.float32), meta_updates=meta_updates)
    return l0_registered


def invert_shift(shift_dy_dx):
    """
    Return the inverse of a pixel translation.

    Args:
        shift_dy_dx: Shift `(dy, dx)`.

    Returns:
        The inverse shift `(-dy, -dx)` as a list.
    """
    shift_dy_dx = np.asarray(shift_dy_dx, dtype=np.float32)
    return (-shift_dy_dx).tolist()


def get_total_shift_to_l1_space(registration_info: Dict[str, Any], band_idx: int) -> np.ndarray:
    """
    Return the total shift that maps one L0 native band into the common L1
    reference space.

    For the current registration pipeline:
    - the master band uses `master_shift_l0_to_l1_dy_dx`,
    - every other band uses its direct shift toward the already-warped reference.

    Args:
        registration_info: Registration metadata dictionary stored in the event.
        band_idx: Target L0 native band index.

    Returns:
        A NumPy array containing the total shift `(dy, dx)` toward the L1 reference
        space.
    """
    master_band = int(registration_info["master_band"])
    if band_idx == master_band:
        return np.asarray(registration_info["master_shift_l0_to_l1_dy_dx"], dtype=np.float32)

    return np.asarray(registration_info["band_shifts_to_ref_dy_dx"][str(band_idx)], dtype=np.float32)


def shift_points(points_yx: np.ndarray, shift_dy_dx) -> np.ndarray:
    """
    Apply a translation to a set of points expressed as `(y, x)` coordinates.

    Args:
        points_yx: Array of shape `(N, 2)` containing points `(y, x)`.
        shift_dy_dx: Shift `(dy, dx)` to apply.

    Returns:
        The shifted points as a NumPy array of shape `(N, 2)`.
    """
    pts = np.asarray(points_yx, dtype=np.float32).copy()
    shift = np.asarray(shift_dy_dx, dtype=np.float32)
    pts[:, 0] += shift[0]
    pts[:, 1] += shift[1]
    return pts


def project_points_l1_to_l0_native(points_yx: np.ndarray, registration_info: Dict[str, Any], target_band: int = 7) -> np.ndarray:
    """
    Project points from the common L1 reference space back to one L0 native band
    space.

    Args:
        points_yx: Array of shape `(N, 2)` containing points `(y, x)` in the common
            L1 reference space.
        registration_info: Registration metadata dictionary.
        target_band: L0 native band into which the points should be projected.

    Returns:
        The projected points as an array of shape `(N, 2)` in the target L0 native
        band space.
    """
    total_shift = get_total_shift_to_l1_space(registration_info, target_band)
    inv_shift = -total_shift
    return shift_points(points_yx, inv_shift)


def project_bbox_l1_to_l0_native(bbox_yxyx, registration_info: Dict[str, Any], target_band: int = 7):
    """
    Project one bounding box from the common L1 reference space back to one L0
    native band space.

    Args:
        bbox_yxyx: Bounding box in `[y0, x0, y1, x1]` format, expressed in the
            common L1 reference space.
        registration_info: Registration metadata dictionary.
        target_band: L0 native band into which the box should be projected.

    Returns:
        A bounding box in `[y0, x0, y1, x1]` format expressed in the target L0
        native band space.
    """
    y0, x0, y1, x1 = bbox_yxyx
    pts = np.array([[y0, x0], [y1, x1]], dtype=np.float32)
    pts_back = project_points_l1_to_l0_native(pts, registration_info, target_band=target_band)
    return [float(pts_back[0, 0]), float(pts_back[0, 1]), float(pts_back[1, 0]), float(pts_back[1, 1])]

def register_event_bands_to_master(
    event,
    master_band=3,
    max_shifts: Tuple[int, int] = (80, 80),
):
    """
    Register all bands of one event to one master band.

    This is an intra-event band-to-band registration helper intended for
    visualization and quick analysis. It does not modify the original event.

    Args:
        event: Event-like object exposing get_band(...), as_numpy(), get_meta(),
            and with_array(...).
        master_band: Master/reference band selector. Usually "RED" / 3 for L1
            visualization.
        max_shifts: Optional maximum shift `(max_dy, max_dx)`.

    Returns:
        A new event-like object with bands shifted into the master-band space.
    """
    _require_torch()

    # Resolve aliases if event supports them through get_band.
    master = event.get_band(master_band)
    arr = event.as_numpy()

    aligned_bands: List[np.ndarray] = []
    band_shifts_to_master: Dict[str, List[float]] = {}

    ref_np = master.astype(np.float32)

    for i in range(arr.shape[0]):
        band_np = event.get_band(i).astype(np.float32)

        if i == master_band or str(master_band).upper() in {str(i), f"B{i}", f"BAND_{i}"}:
            warped_np = ref_np
            shift = np.array([0.0, 0.0], dtype=np.float32)
        else:
            src_i = prep_for_phase_corr(ref_np)
            tgt_i = prep_for_phase_corr(band_np)
            shift = phase_correlation_shift(
                src_i,
                tgt_i,
                max_shifts=max_shifts,
            )[0].cpu().numpy().astype(np.float32)
            warped_np = warp_np_by_shift(band_np, shift)

        aligned_bands.append(warped_np)
        band_shifts_to_master[str(i)] = shift.tolist()

    aligned_cube = np.stack(aligned_bands, axis=0).astype(np.float32)

    meta_updates = {
        "dtype": str(aligned_cube.dtype),
        "band_registration_info": {
            "method": "phase_correlation_to_master_band",
            "master_band": master_band,
            "max_shifts": list(max_shifts),
            "band_shifts_to_master_dy_dx": band_shifts_to_master,
            "source_shape": list(arr.shape),
            "registered_shape": list(aligned_cube.shape),
        },
    }

    return event.with_array(aligned_cube, meta_updates=meta_updates)