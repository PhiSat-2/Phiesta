from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import phiesta
from phiesta import connect_insula, interband_shift_table


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Confirmatory L1C inter-band audit: native-resolution primary "
            "measurement plus a pre-specified half-resolution stability check."
        )
    )
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--master-band", default="RED")
    p.add_argument(
        "--max-sides",
        type=int,
        nargs="+",
        default=[4096, 2048],
        help="First value is primary; remaining values are stability checks.",
    )
    p.add_argument(
        "--plausibility-flag-px",
        type=float,
        default=80.0,
        help=(
            "Diagnostic flag only. Measurements are NOT rejected for exceeding "
            "this value."
        ),
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cleanup", action="store_true")
    return p.parse_args()


def _normalize_ids(df):
    out = df.copy()
    if "product_id" in out.columns:
        out["product_id"] = (
            out["product_id"]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
        )
    return out


def _read(path):
    if not path.exists():
        return pd.DataFrame()
    return _normalize_ids(pd.read_csv(path))


def _atomic_write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _upsert(path, rows, keys):
    old = _read(path)
    both = pd.concat([old, rows], ignore_index=True)
    both = _normalize_ids(both)
    if not both.empty:
        both = both.drop_duplicates(keys, keep="last")
    _atomic_write(both, path)


def _event_folder(event):
    for attr in ("product_folder", "_product_folder"):
        if hasattr(event, attr):
            value = getattr(event, attr)
            if callable(value):
                value = value()
            if value:
                return Path(value)
    return None


def _cleanup(event, cache_dir):
    folder = _event_folder(event)
    if folder is None or not folder.exists():
        return
    try:
        folder.resolve().relative_to(cache_dir.resolve())
    except ValueError:
        print(f"[cleanup] keeping external path: {folder}")
        return
    shutil.rmtree(folder, ignore_errors=True)
    print(f"[cleanup] removed {folder}")


def _decorate(table, row, product_id, identifier, max_side, plausibility):
    table = table.copy()
    table["product_id"] = product_id
    table["level"] = "L1C"
    table["analysis_max_side"] = int(max_side)

    for col in (
        "sample_order",
        "stratum",
        "stratum_population_n",
        "stratum_sample_n",
        "inclusion_probability",
        "design_weight",
        "start_datetime",
        "center_lon",
        "center_lat",
        "time_bin",
        "lat_band",
        "lon_band",
    ):
        if col in row.index:
            table[col] = row[col]

    table["source_identifier"] = str(identifier)
    table["phiesta_version"] = phiesta.__version__
    table["extracted_utc"] = datetime.now(timezone.utc).isoformat()

    dx = pd.to_numeric(table["dx_px"], errors="coerce")
    dy = pd.to_numeric(table["dy_px"], errors="coerce")
    table["exceeds_plausibility_flag"] = (
        (dx.abs() > float(plausibility)) | (dy.abs() > float(plausibility))
    ).fillna(False)

    corr_before = pd.to_numeric(table["corr_before"], errors="coerce")
    corr_after = pd.to_numeric(table["corr_after"], errors="coerce")
    table["corr_improved"] = (corr_after >= corr_before).where(
        corr_before.notna() & corr_after.notna(),
        np.nan,
    )
    return table


def _make_pairs(metrics, primary_side, check_side):
    if metrics.empty:
        return pd.DataFrame()

    keys = ["product_id", "target_band"]
    keep = [
        "product_id",
        "target_band",
        "target_band_name",
        "target_wavelength_nm",
        "dx_px",
        "dy_px",
        "shift_px",
        "response",
        "corr_before",
        "corr_after",
        "corr_gain",
        "status",
        "error",
        "exceeds_plausibility_flag",
    ]

    a = metrics[metrics["analysis_max_side"] == int(primary_side)].copy()
    b = metrics[metrics["analysis_max_side"] == int(check_side)].copy()

    for col in keep:
        if col not in a.columns:
            a[col] = np.nan
        if col not in b.columns:
            b[col] = np.nan

    a = a[keep].rename(
        columns={c: f"{c}_primary" for c in keep if c not in keys}
    )
    b = b[keep].rename(
        columns={c: f"{c}_check" for c in keep if c not in keys}
    )

    pairs = a.merge(b, on=keys, how="outer", validate="one_to_one")

    # Canonical band labels are identical at both scales. Keep unsuffixed
    # convenience columns in addition to the scale-specific provenance columns.
    for col in ("target_band_name", "target_wavelength_nm"):
        pcol = f"{col}_primary"
        ccol = f"{col}_check"
        if pcol in pairs.columns and ccol in pairs.columns:
            pairs[col] = pairs[pcol].combine_first(pairs[ccol])

    pairs["primary_max_side"] = int(primary_side)
    pairs["check_max_side"] = int(check_side)

    dx0 = pd.to_numeric(pairs["dx_px_primary"], errors="coerce")
    dy0 = pd.to_numeric(pairs["dy_px_primary"], errors="coerce")
    dx1 = pd.to_numeric(pairs["dx_px_check"], errors="coerce")
    dy1 = pd.to_numeric(pairs["dy_px_check"], errors="coerce")

    pairs["scale_delta_dx_px"] = dx0 - dx1
    pairs["scale_delta_dy_px"] = dy0 - dy1
    pairs["scale_delta_px"] = np.hypot(
        pairs["scale_delta_dx_px"],
        pairs["scale_delta_dy_px"],
    )
    pairs["scale_pair_available"] = (
        dx0.notna() & dy0.notna() & dx1.notna() & dy1.notna()
    )

    meta_cols = [
        "sample_order",
        "stratum",
        "stratum_population_n",
        "stratum_sample_n",
        "inclusion_probability",
        "design_weight",
        "start_datetime",
        "center_lon",
        "center_lat",
        "time_bin",
        "lat_band",
        "lon_band",
        "source_identifier",
    ]
    primary = metrics[metrics["analysis_max_side"] == int(primary_side)]
    meta_cols = [c for c in meta_cols if c in primary.columns]
    if meta_cols:
        meta = primary[["product_id"] + meta_cols].drop_duplicates("product_id")
        pairs = pairs.merge(meta, on="product_id", how="left", validate="many_to_one")

    return pairs


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    sides = [int(x) for x in args.max_sides]
    if len(sides) < 2:
        raise ValueError("--max-sides needs at least two values, e.g. 4096 2048")
    if len(set(sides)) != len(sides):
        raise ValueError("--max-sides values must be unique.")

    sample = pd.read_csv(args.sample, dtype={"product_id": "string"})
    sample = _normalize_ids(sample)
    if args.limit is not None:
        sample = sample.head(int(args.limit)).copy()

    metrics_path = args.out_dir / "interband_multiscale.csv"
    pairs_path = args.out_dir / "interband_scale_pairs.csv"
    runs_path = args.out_dir / "interband_runs.csv"

    runs = _read(runs_path)
    done = set()
    if not runs.empty:
        ok = runs[runs["status"].astype(str) == "SUCCESS"]
        done = set(ok["product_id"].astype(str))

    client = connect_insula(cache_dir=args.cache_dir)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        product_id = str(row["product_id"])
        if product_id in done:
            print(f"[{i}/{len(sample)}] SKIP {product_id}: already SUCCESS")
            continue

        identifier = row.get("identifier")
        if pd.isna(identifier) or not str(identifier).strip():
            identifier = product_id

        event = None
        started = datetime.now(timezone.utc)

        try:
            print(f"[{i}/{len(sample)}] loading L1C {identifier}")
            event = client.load_l1c(str(identifier))
            all_tables = []

            for side in sides:
                print(f"[{i}/{len(sample)}] {product_id}: measuring max_side={side}")
                table = interband_shift_table(
                    event,
                    master_band=args.master_band,
                    max_side=side,
                    max_shifts=None,
                )
                table = _decorate(
                    table,
                    row,
                    product_id,
                    identifier,
                    side,
                    args.plausibility_flag_px,
                )
                all_tables.append(table)

            combined = pd.concat(all_tables, ignore_index=True)
            _upsert(
                metrics_path,
                combined,
                [
                    "product_id",
                    "level",
                    "master_band_index",
                    "target_band",
                    "analysis_max_side",
                ],
            )

            ok_rows = int((combined["status"] == "ok").sum())
            expected_rows = 7 * len(sides)
            run_status = (
                "SUCCESS"
                if len(combined) == expected_rows and ok_rows == expected_rows
                else "PARTIAL"
            )
            run = pd.DataFrame([{
                "product_id": product_id,
                "level": "L1C",
                "status": run_status,
                "rows": int(len(combined)),
                "ok_rows": ok_rows,
                "non_ok_rows": int(len(combined) - ok_rows),
                "expected_rows": expected_rows,
                "error": "",
                "started_utc": started.isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }])
            _upsert(runs_path, run, ["product_id", "level"])

            if run_status == "SUCCESS":
                done.add(product_id)

            print(
                f"[{i}/{len(sample)}] {run_status} {product_id}: "
                f"{ok_rows}/{expected_rows} scale-band rows ok"
            )

        except PermissionError as exc:
            run = pd.DataFrame([{
                "product_id": product_id,
                "level": "L1C",
                "status": "ACCESS_DENIED",
                "rows": 0,
                "ok_rows": 0,
                "non_ok_rows": 0,
                "expected_rows": 7 * len(sides),
                "error": f"{type(exc).__name__}: {exc}",
                "started_utc": started.isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }])
            _upsert(runs_path, run, ["product_id", "level"])
            print(f"[{i}/{len(sample)}] ACCESS_DENIED {product_id}: {exc}")

        except Exception as exc:
            run = pd.DataFrame([{
                "product_id": product_id,
                "level": "L1C",
                "status": "FAILED",
                "rows": 0,
                "ok_rows": 0,
                "non_ok_rows": 0,
                "expected_rows": 7 * len(sides),
                "error": f"{type(exc).__name__}: {exc}",
                "started_utc": started.isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }])
            _upsert(runs_path, run, ["product_id", "level"])
            print(
                f"[{i}/{len(sample)}] FAILED {product_id}: "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if args.cleanup and event is not None:
                _cleanup(event, args.cache_dir)

        metrics_now = _read(metrics_path)
        pairs = _make_pairs(metrics_now, sides[0], sides[1])
        if not pairs.empty:
            _atomic_write(pairs, pairs_path)

    metrics = _read(metrics_path)
    runs = _read(runs_path)
    pairs = _make_pairs(metrics, sides[0], sides[1])
    if not pairs.empty:
        _atomic_write(pairs, pairs_path)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_sample_n": int(len(sample)),
        "successful_products": int((runs["status"] == "SUCCESS").sum())
        if not runs.empty else 0,
        "partial_products": int((runs["status"] == "PARTIAL").sum())
        if not runs.empty else 0,
        "failed_products": int((runs["status"] == "FAILED").sum())
        if not runs.empty else 0,
        "access_denied_products": int((runs["status"] == "ACCESS_DENIED").sum())
        if not runs.empty else 0,
        "metric_rows": int(len(metrics)),
        "scale_pair_rows": int(len(pairs)),
        "scale_pair_available_rows": int(pairs["scale_pair_available"].sum())
        if not pairs.empty else 0,
        "master_band": args.master_band,
        "primary_max_side": int(sides[0]),
        "stability_max_side": int(sides[1]),
        "plausibility_flag_px": float(args.plausibility_flag_px),
        "hard_shift_rejection": False,
        "cleanup": bool(args.cleanup),
    }
    (args.out_dir / "interband_multiscale_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
