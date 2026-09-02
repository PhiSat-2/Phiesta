from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from affine import Affine


S2_RESOLUTION_M = 10.0
PHISAT2_RESOLUTION_M = 4.75
S2_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07"]
S2_PAN_BANDS = ["B02", "B03", "B04", "PAN", "B08", "B05", "B06", "B07"]
PAN_WEIGHTS = np.asarray(
    [0.21594369, 0.28731533, 0.25719303, 0.0, 0.12275664, 0.11679131, 0.0],
    dtype=np.float32,
)

SIMULATED_OUTPUT_BAND_ORDER_WITH_PAN = [
    "B02_BLUE",
    "B03_GREEN",
    "B04_RED",
    "PAN",
    "B08_NIR",
    "B05_RED_EDGE_1",
    "B06_RED_EDGE_2",
    "B07_RED_EDGE_3",
]


def _resize_stack(stack_chw: np.ndarray, size_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    h, w = map(int, size_hw)
    return np.stack(
        [cv2.resize(band, (w, h), interpolation=interpolation) for band in stack_chw],
        axis=0,
    ).astype(np.float32, copy=False)


def _resize_2d(image: np.ndarray, size_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    h, w = map(int, size_hw)
    return cv2.resize(np.asarray(image, dtype=np.float32), (w, h), interpolation=interpolation)


def _prepare_sun_zenith(metadata: dict[str, Any], size_hw: tuple[int, int]) -> np.ndarray:
    raw = metadata.get("sun_zenith_angles")
    if raw is None:
        raise ValueError(
            "Sentinel metadata does not contain sun_zenith_angles. "
            "Rebuild the Sentinel crop so Phiesta can extract simulation metadata."
        )

    zenith = np.asarray(raw, dtype=np.float32)
    if zenith.ndim == 3:
        zenith = zenith[..., 0]
    if zenith.ndim != 2 or zenith.size == 0:
        raise ValueError(f"Unexpected sun_zenith_angles shape: {zenith.shape}")

    return _resize_2d(zenith, size_hw, cv2.INTER_LINEAR)


def _s2_solar_irradiances(metadata: dict[str, Any]) -> np.ndarray:
    values = metadata.get("solar_irradiances") or {}
    missing = [band for band in S2_BANDS if band not in values]
    if missing:
        raise ValueError(
            "Sentinel metadata is missing solar irradiance values for "
            f"{missing}. Rebuild the Sentinel crop from a complete L1C SAFE product."
        )
    return np.asarray([float(values[band]) for band in S2_BANDS], dtype=np.float32)


def _solar_irradiances_with_pan(metadata: dict[str, Any]) -> np.ndarray:
    s2 = _s2_solar_irradiances(metadata)
    pan = float(np.sum(s2 * PAN_WEIGHTS))
    return np.insert(s2, 3, pan).astype(np.float32)


def _to_radiance(reflectance_hwc: np.ndarray, sun_zenith_hw: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    irradiance = _s2_solar_irradiances(metadata)
    # Prefer the explicit Sentinel-2 reflectance-conversion factor U. The
    # legacy ``earth_sun_dist`` key is kept only for backward compatibility.
    sun_earth_u = float(
        metadata.get("reflectance_conversion_U", metadata.get("earth_sun_dist", 1.0))
    )
    factor = np.cos(np.deg2rad(sun_zenith_hw)).astype(np.float32)
    factor *= sun_earth_u / np.pi
    return reflectance_hwc * factor[..., None] * irradiance[None, None, :]


def _to_reflectance(radiance_hwc: np.ndarray, sun_zenith_hw: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    irradiance = _solar_irradiances_with_pan(metadata)
    # Prefer the explicit Sentinel-2 reflectance-conversion factor U. The
    # legacy ``earth_sun_dist`` key is kept only for backward compatibility.
    sun_earth_u = float(
        metadata.get("reflectance_conversion_U", metadata.get("earth_sun_dist", 1.0))
    )
    factor = np.cos(np.deg2rad(sun_zenith_hw)).astype(np.float32)
    factor *= sun_earth_u / np.pi
    denom = factor[..., None] * irradiance[None, None, :]
    return np.divide(
        radiance_hwc,
        denom,
        out=np.zeros_like(radiance_hwc, dtype=np.float32),
        where=np.abs(denom) > 1e-12,
    )


def _add_pan(stack_hwc: np.ndarray) -> np.ndarray:
    pan = np.sum(stack_hwc * PAN_WEIGHTS[None, None, :], axis=-1)
    return np.insert(stack_hwc, 3, pan, axis=-1).astype(np.float32, copy=False)


def _resample_to_phisat(stack_hwc: np.ndarray, sun_zenith_hw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = stack_hwc.shape[:2]
    scale = S2_RESOLUTION_M / PHISAT2_RESOLUTION_M
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    out = np.stack(
        [
            cv2.resize(stack_hwc[..., i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            for i in range(stack_hwc.shape[-1])
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    sun = cv2.resize(sun_zenith_hw, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return out, sun.astype(np.float32, copy=False)


def _apply_band_misalignment(
    stack_hwc: np.ndarray,
    *,
    processing_level: str,
    rng: np.random.Generator,
    std_pixels: float = 1.0,
) -> np.ndarray:
    """Apply the small stochastic inter-band shifts used by the legacy simulator workflow."""
    if processing_level.upper() == "L1A":
        relative = np.asarray([0.0, 1.105, 1.046, 0.943, 0.837, 0.717, 0.55, 0.44])
        amplitude = rng.normal(0.0, 0.4, size=8) + relative
        angle = rng.uniform(0.0, 2.0 * np.pi, size=8)
        shifts = np.column_stack((amplitude * np.cos(angle), amplitude * np.sin(angle)))
        shifts[0] = 0.0
        shifts = np.flip(np.cumsum(shifts, axis=0), axis=0)
    else:
        amplitude = rng.normal(0.0, float(std_pixels), size=8)
        angle = rng.uniform(0.0, 2.0 * np.pi, size=8)
        shifts = np.column_stack((amplitude * np.cos(angle), amplitude * np.sin(angle)))
        shifts[2] = 0.0

    h, w = stack_hwc.shape[:2]
    bands = []
    for idx in range(stack_hwc.shape[-1]):
        matrix = np.array(
            [[1.0, 0.0, shifts[idx, 0]], [0.0, 1.0, shifts[idx, 1]]],
            dtype=np.float32,
        )
        bands.append(
            cv2.warpAffine(
                stack_hwc[..., idx],
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        )
    return np.stack(bands, axis=-1).astype(np.float32, copy=False)


def _run_binary(executable: str | Path, calculation: str, array_thwc: np.ndarray) -> np.ndarray:
    executable = str(Path(executable))
    with tempfile.TemporaryDirectory(prefix="phiesta_phisat2_") as tmp:
        input_npy = Path(tmp) / "input.npy"
        output_npy = Path(tmp) / "output.npy"
        np.save(input_npy, np.asarray(array_thwc, dtype=np.float32))
        subprocess.run(
            [executable, calculation, str(input_npy), str(output_npy)],
            check=True,
        )
        if not output_npy.exists():
            raise FileNotFoundError(
                f"PhiSat-2 executable completed without writing {output_npy}."
            )
        return np.load(output_npy).astype(np.float32, copy=False)


def simulate_single_file_native(
    *,
    s2_tiff_path: str | Path,
    output_tiff_path: str | Path,
    metadata: dict[str, Any],
    phisat2_exec_path: str | Path,
    processing_level: str = "L1C",
    random_seed: int | None = 0,
) -> None:
    """Simulate a PhiSat-2-like raster without requiring OrbitalAI Python helper code.

    The stochastic inter-band perturbation is deterministic by default so proxy
    and final simulations use the same sensor perturbation. Pass
    ``random_seed=None`` for non-deterministic sampling.
    """
    s2_tiff_path = Path(s2_tiff_path)
    output_tiff_path = Path(output_tiff_path)
    output_tiff_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(s2_tiff_path) as src:
        stack_chw = src.read().astype(np.float32)
        profile = src.profile.copy()
        src_height = int(src.height)
        src_width = int(src.width)

    if stack_chw.ndim != 3 or stack_chw.shape[0] != 7:
        raise ValueError(
            "Phiesta native simulation expects a 7-band Sentinel crop in order "
            f"{S2_BANDS}; got shape {stack_chw.shape}."
        )

    target_size = metadata.get("target_size")
    if target_size is not None:
        target_size = tuple(int(v) for v in target_size)
        if tuple(stack_chw.shape[1:]) != target_size:
            stack_chw = _resize_stack(stack_chw, target_size, cv2.INTER_LINEAR)

    work_h, work_w = stack_chw.shape[1:]
    sun_zenith = _prepare_sun_zenith(metadata, (work_h, work_w))

    reflectance = np.transpose(stack_chw, (1, 2, 0))
    radiance = _to_radiance(reflectance, sun_zenith, metadata)
    radiance = _add_pan(radiance)
    radiance, sun_zenith = _resample_to_phisat(radiance, sun_zenith)
    rng = np.random.default_rng(random_seed)
    radiance = _apply_band_misalignment(
        radiance,
        processing_level=processing_level,
        rng=rng,
        std_pixels=1.0,
    )

    simulated = radiance[None, ...]
    simulated = _run_binary(phisat2_exec_path, "SNR", simulated)
    simulated = _run_binary(phisat2_exec_path, "PSF", simulated)

    if simulated.ndim != 4 or simulated.shape[0] != 1 or simulated.shape[-1] != 8:
        raise ValueError(
            "Unexpected PhiSat-2 executable output shape: "
            f"{simulated.shape}; expected (1,H,W,8)."
        )

    output_hwc = simulated[0]
    if processing_level.upper() == "L1C":
        output_hwc = _to_reflectance(output_hwc, sun_zenith, metadata)

    output_chw = np.transpose(output_hwc, (2, 0, 1)).astype(np.float32, copy=False)
    out_height, out_width = output_chw.shape[1:]

    scale_x = src_width / float(out_width)
    scale_y = src_height / float(out_height)
    profile.update(
        driver="GTiff",
        height=out_height,
        width=out_width,
        count=8,
        dtype="float32",
        transform=profile["transform"] * Affine.scale(scale_x, scale_y),
        compress="deflate",
        bigtiff="if_safer",
    )

    with rasterio.open(output_tiff_path, "w", **profile) as dst:
        dst.write(output_chw)
        dst.descriptions = tuple(SIMULATED_OUTPUT_BAND_ORDER_WITH_PAN)
        dst.update_tags(
            PHIESTA_SIMULATION_BACKEND="native",
            PHIESTA_SIMULATION_PROCESSING_LEVEL=str(processing_level),
            PHIESTA_SIMULATION_RANDOM_SEED=(
                "none" if random_seed is None else str(int(random_seed))
            ),
        )
