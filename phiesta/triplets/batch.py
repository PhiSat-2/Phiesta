from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path
from .metrics import evaluate_real_sim_alignment
from typing import Any, Iterable

from .full_pipeline import build_full_sentinel_triplet


def _safe_get(d: dict[str, Any], path: list[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _summary_row(product_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": product_id,
        "status": result.get("status"),
        "elapsed_s": result.get("elapsed_s"),
        "satellite": _safe_get(result, ["source", "satellite"]),
        "delta_days": _safe_get(result, ["source", "delta_days"]),
        "cloud_cover": _safe_get(result, ["source", "cloud_cover"]),
        "coverage": _safe_get(result, ["source", "coverage"]),
        "matches": _safe_get(result, ["metrics", "matches"]),
        "inliers": _safe_get(result, ["metrics", "inliers"]),
        "inlier_ratio": _safe_get(result, ["metrics", "inlier_ratio"]),
        "valid_fraction_sentinel": _safe_get(result, ["metrics", "valid_fraction_sentinel"]),
        "valid_fraction_simulated": _safe_get(result, ["metrics", "valid_fraction_simulated"]),
        "real_path": _safe_get(result, ["paths", "real"]),
        "sentinel_path": _safe_get(result, ["paths", "sentinel"]),
        "simulated_path": _safe_get(result, ["paths", "simulated"]),
        "report_path": _safe_get(result, ["paths", "report"]),
        "error": "",
    }


def _error_row(product_id: str, elapsed_s: float, exc: BaseException) -> dict[str, Any]:
    return {
        "product_id": product_id,
        "status": "FAILED",
        "elapsed_s": elapsed_s,
        "satellite": "",
        "delta_days": "",
        "cloud_cover": "",
        "coverage": "",
        "matches": "",
        "inliers": "",
        "inlier_ratio": "",
        "valid_fraction_sentinel": "",
        "valid_fraction_simulated": "",
        "real_path": "",
        "sentinel_path": "",
        "simulated_path": "",
        "report_path": "",
        "error": f"{type(exc).__name__}: {exc}",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_full_sentinel_triplets_batch(
    client: Any,
    product_ids: Iterable[str | int],
    output_root: str | Path = "data/triplets",
    batch_output_dir: str | Path | None = None,
    continue_on_error: bool = True,
    save_full_results_json: bool = True,
    verbose: bool = True,
    **triplet_kwargs,
) -> dict[str, Any]:
    """
    Build full Sentinel-2 / simulated PhiSat-2 / real PhiSat-2 triplets for several acquisitions.

    Parameters
    ----------
    client:
        Phiesta Insula client, typically returned by connect_insula().
    product_ids:
        Acquisition identifiers, e.g. ["5359", "5095"].
    output_root:
        Root folder where individual triplets are written.
    batch_output_dir:
        Folder where batch reports are written. Defaults to output_root / "_batch_reports".
    continue_on_error:
        If True, failed acquisitions are logged and the batch continues.
    **triplet_kwargs:
        Forwarded to build_full_sentinel_triplet(...), e.g.
        buffer_km=20.0, proxy_target_size=(1024, 1024), final_margin_pct=0.15.

    Returns
    -------
    dict with:
        - rows: CSV-friendly summary rows
        - results: full successful results / error traces
        - summary_csv
        - summary_json
    """
    t_batch = time.time()

    output_root = Path(output_root)
    if batch_output_dir is None:
        batch_output_dir = output_root / "_batch_reports"
    batch_output_dir = Path(batch_output_dir)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    product_ids = [str(pid) for pid in product_ids]

    rows: list[dict[str, Any]] = []
    full_results: dict[str, Any] = {}

    if verbose:
        print("[Phiesta] Starting full triplet batch")
        print(f"[Phiesta] product_ids={product_ids}")
        print(f"[Phiesta] output_root={output_root}")
        print(f"[Phiesta] batch_output_dir={batch_output_dir}")

    for i, pid in enumerate(product_ids, start=1):
        t0 = time.time()

        if verbose:
            print()
            print("=" * 80)
            print(f"[Phiesta] [{i}/{len(product_ids)}] Building product_id={pid}")
            print("=" * 80)

        try:
            event = client.load_l1(pid)

            result = build_full_sentinel_triplet(
                event=event,
                product_id=pid,
                output_root=output_root,
                verbose=verbose,
                **triplet_kwargs,
            )

            row = _summary_row(pid, result)
            rows.append(row)
            full_results[pid] = result

            if verbose:
                print(f"[Phiesta] SUCCESS product_id={pid}")

        except Exception as exc:
            elapsed_s = time.time() - t0
            row = _error_row(pid, elapsed_s, exc)
            rows.append(row)

            full_results[pid] = {
                "status": "FAILED",
                "product_id": pid,
                "elapsed_s": elapsed_s,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

            if verbose:
                print(f"[Phiesta] FAILED product_id={pid}")
                print(f"[Phiesta] error={type(exc).__name__}: {exc}")

            if not continue_on_error:
                raise

    elapsed_batch_s = time.time() - t_batch

    summary = {
        "status": "SUCCESS",
        "num_products": len(product_ids),
        "num_success": sum(1 for r in rows if r["status"] == "SUCCESS"),
        "num_failed": sum(1 for r in rows if r["status"] != "SUCCESS"),
        "elapsed_batch_s": elapsed_batch_s,
        "product_ids": product_ids,
        "rows": rows,
    }

    summary_csv = batch_output_dir / "full_triplet_batch_summary.csv"
    summary_json = batch_output_dir / "full_triplet_batch_summary.json"
    full_json = batch_output_dir / "full_triplet_batch_results.json"

    _write_csv(summary_csv, rows)
    summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if save_full_results_json:
        full_json.write_text(json.dumps(full_results, indent=2, default=str), encoding="utf-8")

    if verbose:
        print()
        print("=" * 80)
        print("[Phiesta] Batch complete")
        print(f"[Phiesta] success={summary['num_success']} failed={summary['num_failed']}")
        print(f"[Phiesta] elapsed_batch_s={elapsed_batch_s:.1f}")
        print(f"[Phiesta] summary_csv={summary_csv}")
        print(f"[Phiesta] summary_json={summary_json}")
        if save_full_results_json:
            print(f"[Phiesta] full_results_json={full_json}")

    return {
        **summary,
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "full_results_json": str(full_json) if save_full_results_json else None,
        "results": full_results,
    }


def _deep_find(obj, keys):
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
        for v in obj.values():
            found = _deep_find(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _deep_find(v, keys)
            if found is not None:
                return found
    return None


def build_georeferenced_triplets_batch(
    client,
    product_ids,
    *,
    output_root: str | Path = "data/triplets",
    batch_output_dir: str | Path = "outputs/georef_batch_review/batch_reports",
    source: str = "simulated",
    continue_on_error: bool = True,
    save_full_results_json: bool = True,
    verbose: bool = True,
    georef_kwargs: dict[str, Any] | None = None,
    triplet_kwargs: dict[str, Any] | None = None,
    strict_kwargs: dict[str, Any] | None = None,
):
    """
    Recommended batch georeferencing pipeline.

    For each product:
      1. load ΦSat-2 L1 through Insula;
      2. build the full Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplet;
      3. run strict final LightGlue/RANSAC refinement through event.get_georef(...);
      4. save a compact CSV/JSON summary with source, proxy, strict and real/sim metrics.

    This is the batch equivalent of the recommended high-level event.get_georef(...) API.
    """
    georef_kwargs = dict(georef_kwargs or {})
    triplet_kwargs = dict(triplet_kwargs or {})
    strict_kwargs = dict(strict_kwargs or {})

    output_root = Path(output_root)
    batch_output_dir = Path(batch_output_dir)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    full_results = {}

    t_batch = time.time()

    if verbose:
        print("[Phiesta] Starting georeferenced triplet batch")
        print(f"[Phiesta] product_ids={list(product_ids)}")
        print(f"[Phiesta] output_root={output_root}")
        print(f"[Phiesta] batch_output_dir={batch_output_dir}")

    for i, pid in enumerate(product_ids, start=1):
        t0 = time.time()

        if verbose:
            print()
            print("=" * 80)
            print(f"[Phiesta] [{i}/{len(product_ids)}] Georeferencing product_id={pid}")
            print("=" * 80)

        try:
            event = client.load_l1(str(pid))

            tk = {
                "product_id": str(pid),
                "output_root": output_root,
                **triplet_kwargs,
            }

            georef = event.get_georef(
                source=source,
                verbose=verbose,
                triplet_kwargs=tk,
                strict_kwargs=strict_kwargs,
                **georef_kwargs,
            )

            elapsed_s = time.time() - t0

            triplet_report_path = output_root / str(pid) / "full_triplet_report.json"
            triplet_report = {}
            if triplet_report_path.exists():
                triplet_report = json.loads(triplet_report_path.read_text(encoding="utf-8"))

            georef_out = batch_output_dir / f"{pid}_georef.json"
            georef_out.write_text(json.dumps(georef, indent=2, default=str), encoding="utf-8")

            src = triplet_report.get("source", {})
            proxy = triplet_report.get("metrics", {})
            ae = triplet_report.get("alignment_eval", {}).get("summary", {})

            # Backward-compatible fallback: older reports may not contain alignment_eval.
            if not ae:
                real_path = _deep_find(triplet_report, ["real_path", "real"])
                simulated_path = _deep_find(triplet_report, ["simulated_path", "simulated"])

                if real_path and simulated_path:
                    try:
                        ae_full = evaluate_real_sim_alignment(real_path, simulated_path)
                        ae = ae_full.get("summary", {})
                        triplet_report["alignment_eval"] = ae_full
                        triplet_report_path.write_text(
                            json.dumps(triplet_report, indent=2, default=str),
                            encoding="utf-8",
                        )
                    except Exception as exc:
                        ae = {"error": f"{type(exc).__name__}: {exc}"}

            strict_metrics = georef.get("metrics", {})

            row = {
                "product_id": str(pid),
                "status": "SUCCESS",
                "elapsed_s": elapsed_s,

                # Sentinel source selection
                "satellite": src.get("satellite"),
                "delta_days": src.get("delta_days"),
                "cloud_cover": src.get("cloud_cover"),
                "coverage": src.get("coverage"),
                "source_score": src.get("selection_score"),

                # Proxy LightGlue/RANSAC used to crop the final Sentinel window
                "proxy_matches": proxy.get("matches"),
                "proxy_inliers": proxy.get("inliers"),
                "proxy_inlier_ratio": proxy.get("inlier_ratio"),

                # Strict final LightGlue/RANSAC from get_georef()
                "strict_matches": _deep_find(strict_metrics, ["filtered_matches", "matches", "num_matches", "n_matches", "raw_matches"]),
                "strict_inliers": _deep_find(strict_metrics, ["inliers", "num_inliers", "n_inliers"]),
                "strict_inlier_ratio": _deep_find(strict_metrics, ["inlier_ratio", "ratio"]),

                # Real/sim similarity diagnostics
                "PAN_corr": ae.get("PAN_corr"),
                "NIR_corr": ae.get("NIR_corr"),
                "MB_corr_mean": ae.get("MB_corr_mean"),
                "MB_ssim_global_mean": ae.get("MB_ssim_global_mean"),
                "PAN_phase_shift_mag_px_fullres_approx": ae.get("PAN_phase_shift_mag_px_fullres_approx"),

                # Georef
                "quality": georef.get("quality"),
                "center_lonlat": georef.get("center_lonlat"),
                "polygon_geojson": georef.get("polygon_geojson"),

                "triplet_report_path": str(triplet_report_path),
                "georef_json": str(georef_out),
                "error": "",
            }

            rows.append(row)
            full_results[str(pid)] = {
                "status": "SUCCESS",
                "product_id": str(pid),
                "elapsed_s": elapsed_s,
                "triplet_report": triplet_report,
                "georef": georef,
            }

            if verbose:
                print(f"[Phiesta] SUCCESS product_id={pid}")
                print(f"[Phiesta] georef_json={georef_out}")

        except Exception as exc:
            elapsed_s = time.time() - t0
            row = {
                "product_id": str(pid),
                "status": "FAILED",
                "elapsed_s": elapsed_s,
                "satellite": "",
                "delta_days": "",
                "cloud_cover": "",
                "coverage": "",
                "source_score": "",
                "proxy_matches": "",
                "proxy_inliers": "",
                "proxy_inlier_ratio": "",
                "strict_matches": "",
                "strict_inliers": "",
                "strict_inlier_ratio": "",
                "PAN_corr": "",
                "NIR_corr": "",
                "MB_corr_mean": "",
                "MB_ssim_global_mean": "",
                "PAN_phase_shift_mag_px_fullres_approx": "",
                "quality": "",
                "center_lonlat": "",
                "polygon_geojson": "",
                "triplet_report_path": "",
                "georef_json": "",
                "error": f"{type(exc).__name__}: {exc}",
            }

            rows.append(row)
            full_results[str(pid)] = {
                "status": "FAILED",
                "product_id": str(pid),
                "elapsed_s": elapsed_s,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

            if verbose:
                print(f"[Phiesta] FAILED product_id={pid}")
                print(f"[Phiesta] error={type(exc).__name__}: {exc}")

            if not continue_on_error:
                raise

    elapsed_batch_s = time.time() - t_batch

    summary = {
        "status": "SUCCESS",
        "num_products": len(list(product_ids)),
        "num_success": sum(1 for r in rows if r["status"] == "SUCCESS"),
        "num_failed": sum(1 for r in rows if r["status"] != "SUCCESS"),
        "elapsed_batch_s": elapsed_batch_s,
        "product_ids": list(product_ids),
        "rows": rows,
    }

    summary_csv = batch_output_dir / "georef_batch_summary.csv"
    summary_json = batch_output_dir / "georef_batch_summary.json"
    full_json = batch_output_dir / "georef_batch_results.json"

    _write_csv(summary_csv, rows)
    summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if save_full_results_json:
        full_json.write_text(json.dumps(full_results, indent=2, default=str), encoding="utf-8")

    if verbose:
        print()
        print("=" * 80)
        print("[Phiesta] Georeferenced batch complete")
        print(f"[Phiesta] success={summary['num_success']} failed={summary['num_failed']}")
        print(f"[Phiesta] elapsed_batch_s={elapsed_batch_s:.1f}")
        print(f"[Phiesta] summary_csv={summary_csv}")
        print(f"[Phiesta] summary_json={summary_json}")
        if save_full_results_json:
            print(f"[Phiesta] full_results_json={full_json}")

    return {
        **summary,
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "full_results_json": str(full_json) if save_full_results_json else None,
        "results": full_results,
    }
