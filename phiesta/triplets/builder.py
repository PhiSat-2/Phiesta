from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import (
    TripletPaths,
    TripletResult,
    SimulationResult,
)
from .phisat2_executable import resolve_phisat2_executable

from ..remote.catalog_geometry import (
    get_catalog_identifier,
    extract_phisat_acquisition_id,
)

from .sentinel_source import find_best_sentinel_source

from .sentinel_crop import create_sentinel_crop

from .simulation import simulate_phisat2_from_sentinel_crop

def _event_meta(event: Any) -> dict:
    if hasattr(event, "get_meta"):
        return event.get_meta()
    if hasattr(event, "meta"):
        return event.meta
    if hasattr(event, "_meta"):
        return event._meta
    return {}




def _infer_product_id(event: Any, product_id: str | None = None) -> str:
    if product_id is not None:
        extracted = extract_phisat_acquisition_id(product_id)
        return extracted if extracted is not None else str(product_id)

    # Best case: event already has catalog_geo metadata.
    catalog_id = get_catalog_identifier(event)
    if catalog_id is not None:
        return str(catalog_id)

    meta = _event_meta(event)

    for key in (
        "acquisition_id",
        "phisat_acquisition_id",
        "product_id",
        "insula_product_identifier",
        "productIdentifier",
        "insula_filename",
        "filename",
        "resolved_product_folder",
        "product_folder",
        "path",
    ):
        extracted = extract_phisat_acquisition_id(meta.get(key))
        if extracted is not None:
            return extracted

    for attr in ("product_folder", "_product_folder"):
        extracted = extract_phisat_acquisition_id(getattr(event, attr, None))
        if extracted is not None:
            return extracted

    raise ValueError(
        "Could not infer PhiSat-2 acquisition id from event metadata. "
        "Pass product_id=... explicitly."
    )


def build_sentinel_triplet(
    event: Any,
    product_id: str | None = None,
    output_dir: str | Path = "data/triplets",
    satellite: str = "S2B",
    window_days: int = 60,
    max_cloud_cover: float = 40.0,
    buffer_km: float = 10.0,
    match_band: str = "PAN",
    transform_model: str = "homography",
    phisat2_exec_path: str | Path | None = None,
    save: bool = True,
    verbose: bool = True,
    run_sentinel_source: bool = True,
    min_coverage: float = 0.85,
    w_time: float = 0.05,
    w_cloud: float = 1.0,
    max_candidates_to_verify: int = 20,
    run_sentinel_crop: bool = True,
    overwrite_crop: bool = False,
    sentinel_backend: str = "auto",
    sentinel_cache_dir: str | Path = "cache/sentinel2",
    cdse_username: str | None = None,
    cdse_password: str | None = None,
    cdse_access_token: str | None = None,
    overwrite_sentinel_download: bool = False,
    run_simulation: bool = True,
    overwrite_simulation: bool = False,
    simulation_workers: int = 1,
    simulation_target_size: tuple[int, int] | None = (1024, 1024),
    simulation_seed: int | None = 0,
) -> TripletResult:
    """
    Initialize a Sentinel-2 / simulated PhiSat-2 / real PhiSat-2 triplet build.

    This function currently creates the standard output structure, resolves the
    PhiSat-2 simulator executable, and writes an initial QC file. The actual
    Sentinel sourcing, crop, simulation, and LightGlue alignment are added in
    the next pipeline stages.
    """
    pid = _infer_product_id(event, product_id=product_id)

    root = Path(output_dir) / str(pid)
    paths = TripletPaths.from_root(root)

    if save:
        paths.make_dirs()

    exec_path = resolve_phisat2_executable(phisat2_exec_path)

    triplet = TripletResult(
        product_id=str(pid),
        status="INITIALIZED",
        paths=paths,
        simulation=SimulationResult(
            simulated_path=None,
            phisat2_exec_path=str(exec_path),
            processing_level="L1C",
            backend="executable",
            band_order=[],
            metadata={
                "simulator": "PhiSat-2 executable",
            },
        ),
        qc={
            "product_id": str(pid),
            "status": "INITIALIZED",
            "satellite": satellite,
            "window_days": int(window_days),
            "max_cloud_cover": float(max_cloud_cover),
            "min_coverage": float(min_coverage),
            "source_w_time": float(w_time),
            "source_w_cloud": float(w_cloud),
            "max_candidates_to_verify": int(max_candidates_to_verify),
            "buffer_km": float(buffer_km),
            "match_band": str(match_band),
            "transform_model": str(transform_model),
            "phisat2_exec_path": str(exec_path),
            "pipeline_stage": "init",
        },
    )

    if verbose:
        print(f"[Phiesta] Initialized triplet workspace: {paths.root_dir}")
        print(f"[Phiesta] PhiSat-2 executable: {exec_path}")

    if run_sentinel_source:
        if verbose:
            print(
                f"[Phiesta] Searching {satellite} source "
                f"(horizon=±{window_days}d, cloud<={max_cloud_cover}%, "
                f"coverage>={min_coverage:.2f}, weak_time_prior={w_time})"
            )

        source = find_best_sentinel_source(
            event,
            product_id=pid,
            satellite=satellite,
            buffer_km=buffer_km,
            window_days=window_days,
            max_cloud_cover=max_cloud_cover,
            min_coverage=min_coverage,
            w_time=w_time,
            w_cloud=w_cloud,
            max_candidates_to_verify=max_candidates_to_verify,
        )

        triplet.sentinel_source = source
        triplet.status = "SENTINEL_SOURCE_FOUND"
        triplet.qc.update(
            {
                "status": triplet.status,
                "pipeline_stage": "sentinel_source",
                "s2_datetime": source.s2_datetime,
                "delta_days": source.delta_days,
                "s2_cloud_cover": source.cloud_cover,
                "s2_coverage": source.coverage,
                "l1c_paths": source.l1c_paths,
                "l2a_paths": source.l2a_paths,
            }
        )

        if verbose:
            print(
                "[Phiesta] Sentinel source found: "
                f"{source.satellite}, delta={source.delta_days}d, "
                f"cloud={source.cloud_cover}%, coverage={source.coverage}%"
            )

    if run_sentinel_crop:
        if triplet.sentinel_source is None:
            raise ValueError("Cannot run Sentinel crop before Sentinel source selection.")

        if verbose:
            print("[Phiesta] Building Sentinel-2B crop")

        crop = create_sentinel_crop(
            event=event,
            source=triplet.sentinel_source,
            output_dir=paths.sentinel_dir,
            buffer_km=buffer_km,
            overwrite=overwrite_crop,
            verbose=verbose,
            sentinel_backend=sentinel_backend,
            sentinel_cache_dir=sentinel_cache_dir,
            cdse_username=cdse_username,
            cdse_password=cdse_password,
            cdse_access_token=cdse_access_token,
            overwrite_sentinel_download=overwrite_sentinel_download,
        )

        triplet.sentinel_crop = crop
        triplet.status = "SENTINEL_CROP_CREATED"
        triplet.qc.update(
            {
                "status": triplet.status,
                "pipeline_stage": "sentinel_crop",
                "s2_crop_path": crop.crop_path,
                "s2_metadata_path": crop.metadata_path,
                "s2_crop_bands": crop.bands,
            }
        )
    
    if run_simulation:
        if triplet.sentinel_crop is None:
            raise ValueError("Cannot run simulation before Sentinel crop creation.")

        if verbose:
            print("[Phiesta] Running PhiSat-2 simulation")

        simulation = simulate_phisat2_from_sentinel_crop(
            crop=triplet.sentinel_crop,
            output_dir=paths.simulated_dir,
            phisat2_exec_path=exec_path,
            processing_level="L1C",
            workers=simulation_workers,
            overwrite=overwrite_simulation,
            target_size=simulation_target_size,
            verbose=verbose,
            random_seed=simulation_seed,
        )

        triplet.simulation = simulation
        triplet.status = "SIMULATION_CREATED"
        triplet.qc.update(
            {
                "status": triplet.status,
                "pipeline_stage": "simulation",
                "simulated_path": simulation.simulated_path,
                "simulation_backend": simulation.backend,
                "simulation_processing_level": simulation.processing_level,
                "simulation_band_order": simulation.band_order,
                "simulation_target_size": simulation_target_size,
                "simulation_seed": simulation_seed,
            }
        )

    if save:
        triplet.save_qc()

    return triplet