from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .models import SentinelCropResult, SimulationResult
from .phisat2_executable import resolve_phisat2_executable


SIMULATED_BAND_ORDER = [
    "BLUE",
    "GREEN",
    "RED",
    "PAN",
    "NIR",
    "RED_EDGE_1",
    "RED_EDGE_2",
    "RED_EDGE_3",
]



def _ensure_orbitalai_vendor_importable() -> None:
    """
    Make the vendored OrbitalAI simulation package importable.

    This is needed when running scripts from examples/, where Python's sys.path
    may not include the repository root even if pyrawph itself is installed in
    editable mode.
    """
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _ensure_simulation_metadata_alias(metadata_path: str | Path) -> Path:
    """
    Ensure compatibility with the vendored OrbitalAI simulation code.

    Some older scripts expect:
        <product_id>_S2B_metadata.json

    PyRawPh writes:
        <product_id>_s2b_metadata.json

    This helper creates the expected alias if needed.
    """
    metadata_path = Path(metadata_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    product_id = str(meta.get("product_id", metadata_path.stem.split("_")[0]))
    expected = metadata_path.parent / f"{product_id}_S2B_metadata.json"

    if expected.resolve() != metadata_path.resolve():
        shutil.copy2(metadata_path, expected)

    return expected


def simulate_phisat2_from_sentinel_crop(
    crop: SentinelCropResult,
    output_dir: str | Path,
    phisat2_exec_path: str | Path | None = None,
    processing_level: str = "L1C",
    workers: int = 1,
    overwrite: bool = False,
    verbose: bool = True,
    target_size: tuple[int, int] | None = None,
    allow_large_native: bool = False,
) -> SimulationResult:
    """
    Simulate a PhiSat-2-like product from a Sentinel-2 crop.

    This wraps the vendored OrbitalAI / PhiSat-2 simulation workflow.

    Args:
        crop: Result returned by create_sentinel_crop(...).
        output_dir: Directory where the simulated GeoTIFF will be written.
        phisat2_exec_path: Optional path to the PhiSat-2 executable. If omitted,
            the bundled executable for the current OS is used.
        processing_level: Simulation processing level, usually "L1C".
        workers: Kept for API compatibility. Current wrapper runs one file.
        overwrite: If False and an existing simulation output is found, reuse it.
        verbose: Print progress messages.
        target_size: Optional debug resize `(height, width)` before simulation.
            Keep `None` for native crop size.

    Returns:
        SimulationResult with the simulated GeoTIFF path and metadata.
    """
    if crop.crop_path is None:
        raise ValueError("crop.crop_path is None. Cannot run simulation.")

    if crop.metadata_path is None:
        raise ValueError("crop.metadata_path is None. Cannot run simulation.")

    crop_path = Path(crop.crop_path)
    metadata_path = Path(crop.metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not crop_path.exists():
        raise FileNotFoundError(f"Sentinel crop not found: {crop_path}")
    
    import rasterio

    with rasterio.open(crop_path) as src:
        input_pixels = src.width * src.height
        input_shape = (src.height, src.width)

    if target_size is None and not allow_large_native and input_pixels > 4_000_000:
        raise MemoryError(
            "Native full-size simulation is likely too large for memory. "
            f"Input crop shape is {input_shape}, i.e. {input_pixels:,} pixels before "
            "PhiSat-2 resampling. Use target_size=(512,512) or target_size=(1024,1024) "
            "for testing, or implement tiled simulation for full-resolution output. "
            "Pass allow_large_native=True only if you are sure the machine has enough RAM."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(f"Sentinel metadata not found: {metadata_path}")

    exec_path = resolve_phisat2_executable(phisat2_exec_path)

    if target_size is None:
        size_tag = "native"
    else:
        size_tag = f"{int(target_size[0])}x{int(target_size[1])}"

    expected_output = output_dir / f"simulated_{processing_level}_{size_tag}_{crop_path.name}"

    if expected_output.exists() and not overwrite:
        if verbose:
            print(f"[PyRawPh] Reusing simulated PhiSat-2 product: {expected_output}")

        return SimulationResult(
            simulated_path=str(expected_output),
            phisat2_exec_path=str(exec_path),
            processing_level=processing_level,
            backend="executable",
            band_order=list(SIMULATED_BAND_ORDER),
            metadata={
                "status": "ALREADY_EXISTS",
                "source_crop": str(crop_path),
                "metadata_path": str(metadata_path),
                "workers_ignored": workers,
            },
        )

    expected_metadata = _ensure_simulation_metadata_alias(metadata_path)

    if verbose:
        print(f"[PyRawPh] Simulating PhiSat-2 from Sentinel crop: {crop_path}")
        print(f"[PyRawPh] Simulation metadata: {expected_metadata}")
        print(f"[PyRawPh] PhiSat-2 executable: {exec_path}")

    _ensure_orbitalai_vendor_importable()

    from third_party.orbitalai_phisat2_sim.simulation_config import (
        SimulationConfig,
        SimulationSteps,
    )
    from third_party.orbitalai_phisat2_sim.simulation_pipeline import SimulationPipeline

    steps = SimulationSteps(
        radiance=True,
        add_panchromatic=True,
        band_misalignment=True,
        snr_simulation=True,
        psf_filtering=True,
        reflectance_conversion=True,
    )

    config = SimulationConfig(
        steps=steps,
        s2_source_dir=str(crop_path.parent),
        output_dir=str(output_dir),
        phisat2_exec_path=str(exec_path),
        snr_psf_method="executable",
        processing_level=str(processing_level),
    )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata["target_size"] = target_size

    pipeline = SimulationPipeline(config=config)

    ok = pipeline.simulate_single_file(
        s2_tiff_path=crop_path,
        output_tiff_path=expected_output,
        metadata=metadata,
    )

    if not ok:
        raise RuntimeError("PhiSat-2 simulation failed.")

    if not expected_output.exists():
        raise FileNotFoundError(
            f"Simulation returned success but output file was not found: {expected_output}"
        )

    if verbose:
        print(f"[PyRawPh] Simulated PhiSat-2 product saved: {expected_output}")

    return SimulationResult(
        simulated_path=str(expected_output),
        phisat2_exec_path=str(exec_path),
        processing_level=processing_level,
        backend="executable",
        band_order=list(SIMULATED_BAND_ORDER),
        metadata={
            "status": "SUCCESS",
            "source_crop": str(crop_path),
            "metadata_path": str(metadata_path),
            "expected_metadata_alias": str(expected_metadata),
            "target_size": target_size,
            "workers_ignored": workers,
        },
    )