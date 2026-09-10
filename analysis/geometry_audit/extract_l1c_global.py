from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import phiesta
from phiesta import connect_insula, interband_shift_table


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract global inter-band geometry for an L1C study sample."
    )
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--master-band", default="RED")
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument(
        "--max-shift",
        type=float,
        default=80.0,
        help="Maximum plausible target->master displacement in native pixels.",
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


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(args.sample, dtype={"product_id": "string"})
    sample = _normalize_ids(sample)
    if args.limit is not None:
        sample = sample.head(int(args.limit)).copy()

    metrics_path = args.out_dir / "interband_global.csv"
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

            table = interband_shift_table(
                event,
                master_band=args.master_band,
                max_side=int(args.max_side),
                max_shifts=(float(args.max_shift), float(args.max_shift)),
            )

            table["product_id"] = product_id
            table["level"] = "L1C"
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
            ):
                if col in row.index:
                    table[col] = row[col]

            table["source_identifier"] = str(identifier)
            table["phiesta_version"] = phiesta.__version__
            table["extracted_utc"] = datetime.now(timezone.utc).isoformat()

            _upsert(
                metrics_path,
                table,
                [
                    "product_id",
                    "level",
                    "master_band_index",
                    "target_band",
                    "max_side",
                ],
            )

            ok_rows = int((table["status"] == "ok").sum())
            run_status = "SUCCESS" if ok_rows == len(table) else "PARTIAL"
            run = pd.DataFrame([{
                "product_id": product_id,
                "level": "L1C",
                "status": run_status,
                "rows": int(len(table)),
                "ok_rows": ok_rows,
                "non_ok_rows": int(len(table) - ok_rows),
                "error": "",
                "started_utc": started.isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }])
            _upsert(runs_path, run, ["product_id", "level"])

            if run_status == "SUCCESS":
                done.add(product_id)

            print(
                f"[{i}/{len(sample)}] {run_status} {product_id}: "
                f"{ok_rows}/{len(table)} rows ok"
            )

        except PermissionError as exc:
            run = pd.DataFrame([{
                "product_id": product_id,
                "level": "L1C",
                "status": "ACCESS_DENIED",
                "rows": 0,
                "ok_rows": 0,
                "non_ok_rows": 0,
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

    metrics = _read(metrics_path)
    runs = _read(runs_path)
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
        "master_band": args.master_band,
        "max_side": int(args.max_side),
        "max_shift_native_px": float(args.max_shift),
        "cleanup": bool(args.cleanup),
    }
    (args.out_dir / "interband_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
