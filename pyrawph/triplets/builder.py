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
    window_days: int = 15,
    max_cloud_cover: float = 20.0,
    buffer_km: float = 10.0,
    match_band: str = "PAN",
    transform_model: str = "homography",
    phisat2_exec_path: str | Path | None = None,
    save: bool = True,
    verbose: bool = True,
    run_sentinel_source: bool = True,
    min_coverage: float = 0.85,
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
                "simulator": "OrbitalAI PhiSat-2 executable",
            },
        ),
        qc={
            "product_id": str(pid),
            "status": "INITIALIZED",
            "satellite": satellite,
            "window_days": int(window_days),
            "max_cloud_cover": float(max_cloud_cover),
            "buffer_km": float(buffer_km),
            "match_band": str(match_band),
            "transform_model": str(transform_model),
            "phisat2_exec_path": str(exec_path),
            "pipeline_stage": "init",
        },
    )

    if verbose:
        print(f"[PyRawPh] Initialized triplet workspace: {paths.root_dir}")
        print(f"[PyRawPh] PhiSat-2 executable: {exec_path}")

    if run_sentinel_source:
        if verbose:
            print(
                f"[PyRawPh] Searching {satellite} source "
                f"(±{window_days}d, cloud<={max_cloud_cover}%, buffer={buffer_km} km)"
            )

        source = find_best_sentinel_source(
            event,
            product_id=pid,
            satellite=satellite,
            buffer_km=buffer_km,
            window_days=window_days,
            max_cloud_cover=max_cloud_cover,
            min_coverage=min_coverage,
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
                "[PyRawPh] Sentinel source found: "
                f"{source.satellite}, delta={source.delta_days}d, "
                f"cloud={source.cloud_cover}%, coverage={source.coverage}%"
            )

    if save:
        triplet.save_qc()

    return triplet