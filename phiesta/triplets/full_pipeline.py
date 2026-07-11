from __future__ import annotations

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import rasterio

from .proxy_alignment import run_proxy_alignment
from .window_from_proxy import estimate_final_sentinel_window_from_proxy
from .sentinel_window_crop import crop_sentinel_window
from .simulation import simulate_phisat2_from_sentinel_crop
from .final_warp import warp_final_triplet_to_real_grid


def _infer_product_id(event: Any, product_id: str | int | None = None) -> str:
    if product_id is not None:
        s = str(product_id)
        return str(int(s)) if s.isdigit() else s

    meta = getattr(event, "meta", getattr(event, "_meta", {}))

    candidates = [
        meta.get("insula_product_identifier"),
        meta.get("insula_filename"),
        meta.get("path"),
        meta.get("resolved_product_folder"),
    ]

    for value in candidates:
        if value is None:
            continue
        text = str(value)
        m = re.search(r"PHISAT-2_L[01]_0*(\d+)_", text)
        if m:
            return str(int(m.group(1)))

    raise ValueError("Could not infer product_id. Pass product_id=... explicitly.")


def _valid_fraction(path: str | Path, band: int = 1) -> float:
    with rasterio.open(path) as src:
        x = src.read(band)
    return float(np.mean(np.isfinite(x) & (x != 0)))


def build_full_sentinel_triplet(
    event: Any,
    product_id: str | int | None = None,
    output_root: str | Path = "data/triplets",
    window_days: int = 60,
    max_cloud_cover: float = 40.0,
    min_coverage: float = 0.85,
    source_w_time: float = 0.05,
    source_w_cloud: float = 1.0,
    max_candidates_to_verify: int = 20,
    buffer_km: float = 20.0,
    satellite: str = "S2B",
    proxy_target_size: tuple[int, int] = (1024, 1024),
    matching_max_side: int = 1800,
    features: str = "superpoint",
    max_keypoints: int = 8000,
    final_margin_pct: float = 0.15,
    sentinel_backend: str = "auto",
    sentinel_cache_dir: str | Path = "cache/sentinel2",
    cdse_username: str | None = None,
    cdse_password: str | None = None,
    cdse_access_token: str | None = None,
    overwrite_sentinel_download: bool = False,
    final_simulation_target_size: tuple[int, int] | None = None,
    allow_large_native: bool = True,
    overwrite_big_crop: bool = False,
    overwrite_proxy: bool = False,
    overwrite_final_crop: bool = True,
    overwrite_final_simulation: bool = True,
    overwrite_final_warp: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Build a full pixel-aligned triplet:

        real PhiSat-2 L1
        Sentinel-2B warped to the real PhiSat-2 grid
        simulated PhiSat-2 from Sentinel-2B warped to the real PhiSat-2 grid

    This is the high-level researcher-facing wrapper.

    Final outputs are written to:
        data/triplets/<product_id>/final_triplet/

    Default strategy:
        1. Search Sentinel-2B candidates within a configurable temporal horizon
        2. Build a large Sentinel crop with buffer_km
        3. Run low-resolution proxy simulation
        4. Align proxy to real PhiSat-2 using LightGlue
        5. Estimate final native Sentinel crop window
        6. Simulate final crop at native resolution
        7. Warp Sentinel and simulation to the real PhiSat-2 4096x4096 grid
    """
    t0 = time.time()

    pid = _infer_product_id(event, product_id)
    output_root = Path(output_root)
    product_dir = output_root / pid

    if verbose:
        print("[Phiesta] Building full Sentinel/PhiSat-2 triplet")
        print(f"[Phiesta] product_id={pid}")
        print(f"[Phiesta] output_dir={product_dir}")
        print(f"[Phiesta] buffer_km={buffer_km}")
        print(f"[Phiesta] proxy_target_size={proxy_target_size}")
        print(f"[Phiesta] final_simulation_target_size={final_simulation_target_size}")

    # 1. Big Sentinel crop
    triplet = event.build_sentinel_triplet(
        product_id=pid,
        output_dir=output_root,
        satellite=satellite,
        window_days=window_days,
        max_cloud_cover=max_cloud_cover,
        min_coverage=min_coverage,
        w_time=source_w_time,
        w_cloud=source_w_cloud,
        max_candidates_to_verify=max_candidates_to_verify,
        buffer_km=buffer_km,
        run_sentinel_source=True,
        run_sentinel_crop=True,
        run_simulation=False,
        overwrite_crop=overwrite_big_crop,
            sentinel_backend=sentinel_backend,
        sentinel_cache_dir=sentinel_cache_dir,
        cdse_username=cdse_username,
        cdse_password=cdse_password,
        cdse_access_token=cdse_access_token,
        overwrite_sentinel_download=overwrite_sentinel_download,
)

    # 2. Proxy simulation + rectification + LightGlue
    proxy = run_proxy_alignment(
        event=event,
        triplet=triplet,
        proxy_target_size=proxy_target_size,
        matching_max_side=matching_max_side,
        features=features,
        max_keypoints=max_keypoints,
        overwrite=overwrite_proxy,
        save_aligned_preview=True,
        verbose=verbose,
    )

    # 3. Estimate final Sentinel native window
    window = estimate_final_sentinel_window_from_proxy(
        event=event,
        sentinel_big_crop_path=triplet.sentinel_crop.crop_path,
        proxy_simulated_path=proxy["proxy_simulated_path"],
        rectification_homography=proxy["rectification_homography"],
        H_rectified_to_real=proxy["H_rectified_to_real"],
        margin_pct=final_margin_pct,
        verbose=verbose,
    )

    # 4. Crop final Sentinel window
    final_crop = crop_sentinel_window(
        sentinel_big_crop_path=triplet.sentinel_crop.crop_path,
        metadata_path=triplet.sentinel_crop.metadata_path,
        window_native=window["window_native"],
        output_dir=product_dir / "sentinel_final",
        product_id=pid,
        overwrite=overwrite_final_crop,
        verbose=verbose,
    )

    final_crop_obj = SimpleNamespace(
        crop_path=final_crop["crop_path"],
        metadata_path=final_crop["metadata_path"],
    )

    # 5. Final simulation
    final_sim = simulate_phisat2_from_sentinel_crop(
        crop=final_crop_obj,
        output_dir=product_dir / "simulated_final_native",
        overwrite=overwrite_final_simulation,
        target_size=final_simulation_target_size,
        allow_large_native=allow_large_native,
        verbose=verbose,
    )

    # 6. Final warp to real PhiSat-2 grid
    final_triplet = warp_final_triplet_to_real_grid(
        event=event,
        final_sentinel_crop_path=final_crop["crop_path"],
        final_simulated_path=final_sim.simulated_path,
        window_info=window,
        output_dir=product_dir / "final_triplet",
        overwrite=overwrite_final_warp,
        verbose=verbose,
    )

    # 7. Metrics
    vf_sentinel = _valid_fraction(final_triplet["sentinel_warped_path"], band=1)
    vf_simulated = _valid_fraction(final_triplet["simulated_warped_path"], band=1)

    elapsed_s = time.time() - t0

    result = {
        "status": "SUCCESS",
        "product_id": pid,
        "elapsed_s": elapsed_s,
        "config": {
            "window_days": window_days,
            "max_cloud_cover": max_cloud_cover,
            "min_coverage": min_coverage,
            "source_w_time": source_w_time,
            "source_w_cloud": source_w_cloud,
            "max_candidates_to_verify": max_candidates_to_verify,
            "buffer_km": buffer_km,
            "satellite": satellite,
            "proxy_target_size": proxy_target_size,
            "matching_max_side": matching_max_side,
            "features": features,
            "max_keypoints": max_keypoints,
            "final_margin_pct": final_margin_pct,
            "final_simulation_target_size": final_simulation_target_size,
        },
        "source": {
            "satellite": triplet.sentinel_source.satellite if triplet.sentinel_source else None,
            "delta_days": triplet.sentinel_source.delta_days if triplet.sentinel_source else None,
            "cloud_cover": triplet.sentinel_source.cloud_cover if triplet.sentinel_source else None,
            "coverage": triplet.sentinel_source.coverage if triplet.sentinel_source else None,
        },
        "metrics": {
            "matches": proxy["matches"],
            "inliers": proxy["inliers"],
            "inlier_ratio": proxy["inlier_ratio"],
            "valid_fraction_sentinel": vf_sentinel,
            "valid_fraction_simulated": vf_simulated,
        },
        "paths": {
            "real": final_triplet["real_path"],
            "sentinel": final_triplet["sentinel_warped_path"],
            "simulated": final_triplet["simulated_warped_path"],
            "metadata": final_triplet["metadata_path"],
            "final_sentinel_crop": final_crop["crop_path"],
            "final_simulated_native": final_sim.simulated_path,
            "proxy_simulated": proxy["proxy_simulated_path"],
            "proxy_preview": proxy["aligned_proxy_preview_path"],
        },
        "intermediate": {
            "triplet_summary": triplet.summary(),
            "proxy": proxy,
            "window": window,
            "final_crop": final_crop,
            "final_simulation": final_sim.__dict__ if hasattr(final_sim, "__dict__") else str(final_sim),
            "final_triplet": final_triplet,
        },
    }

    report_path = product_dir / "full_triplet_report.json"
    report_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["paths"]["report"] = str(report_path)

    if verbose:
        print("[Phiesta] Full triplet build complete")
        print(f"[Phiesta] elapsed_s={elapsed_s:.1f}")
        print(f"[Phiesta] valid_fraction_sentinel={vf_sentinel:.3f}")
        print(f"[Phiesta] valid_fraction_simulated={vf_simulated:.3f}")
        print(f"[Phiesta] report={report_path}")

    return result
