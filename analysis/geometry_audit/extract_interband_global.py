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
    parser = argparse.ArgumentParser(
        description="Extract paired L1A/L1C global inter-band geometry metrics."
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--master-band", default="RED")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--max-shift", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=("L1A", "L1C"),
        default=("L1A", "L1C"),
        help="Product levels to extract. Example: --levels L1C",
    )
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def _normalize_product_ids(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if "product_id" in out.columns:
        out["product_id"] = (
            out["product_id"]
            .astype("string")
            .str.replace(r"\\.0$", "", regex=True)
        )
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _normalize_product_ids(pd.read_csv(path))


def _write_csv_atomic(table: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(path)


def _append_deduplicated(path: Path, rows: pd.DataFrame):
    previous = _read_csv(path)
    combined = pd.concat([previous, rows], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(
            subset=[
                "product_id",
                "level",
                "master_band_index",
                "target_band",
                "max_side",
            ],
            keep="last",
        )
    _write_csv_atomic(combined, path)


def _record_run(path: Path, record: dict):
    previous = _read_csv(path)
    combined = pd.concat(
        [previous, pd.DataFrame([record])],
        ignore_index=True,
    )
    combined = _normalize_product_ids(combined)
    combined = combined.drop_duplicates(
        subset=["product_id", "level"],
        keep="last",
    )
    _write_csv_atomic(combined, path)


def _success_keys(runs: pd.DataFrame) -> set[tuple[str, str]]:
    if runs.empty:
        return set()
    ok = runs[runs["status"].astype(str) == "SUCCESS"]
    return {
        (str(row["product_id"]), str(row["level"]))
        for _, row in ok.iterrows()
    }


def _safe_cleanup_event(event, cache_root: Path):
    folder = None
    for attr in ("product_folder", "_product_folder"):
        if not hasattr(event, attr):
            continue
        value = getattr(event, attr)
        try:
            value = value() if callable(value) else value
        except TypeError:
            pass
        if value:
            folder = Path(value)
            break

    if folder is None or not folder.exists():
        return

    cache_root = cache_root.resolve()
    try:
        folder.resolve().relative_to(cache_root)
    except ValueError:
        print(f"[cleanup] keeping external path: {folder}")
        return

    shutil.rmtree(folder, ignore_errors=True)
    print(f"[cleanup] removed {folder}")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(
        args.cohort,
        dtype={"product_id": "string"},
    )
    usable = cohort[
        cohort["strict_usable"].astype(str).str.lower().isin(
            {"true", "1"}
        )
    ].copy()

    usable["_numeric_id"] = pd.to_numeric(
        usable["product_id"], errors="coerce"
    )
    usable = usable.sort_values(
        ["_numeric_id", "product_id"],
        na_position="last",
    ).drop(columns="_numeric_id")

    if args.limit is not None:
        usable = usable.head(int(args.limit))

    metrics_path = args.out_dir / "interband_global.csv"
    runs_path = args.out_dir / "interband_runs.csv"
    runs = _read_csv(runs_path)
    success = _success_keys(runs)

    client = connect_insula(cache_dir=args.cache_dir)

    print(
        f"[geometry] strict usable pairs={len(usable)}, "
        f"master={args.master_band}, max_side={args.max_side}"
    )

    for pair_index, (_, row) in enumerate(usable.iterrows(), start=1):
        product_id = str(row["product_id"])
        print()
        print(
            f"[geometry] pair {pair_index}/{len(usable)} "
            f"product={product_id}"
        )

        available_levels = {
            "L1A": ("l1a_identifier", client.load_l1a),
            "L1C": ("l1c_identifier", client.load_l1c),
        }
        for level in args.levels:
            identifier_col, loader = available_levels[level]
            key = (product_id, level)
            if key in success:
                print(f"[geometry] SKIP {product_id} {level}: already SUCCESS")
                continue

            identifier = row.get(identifier_col)
            if pd.isna(identifier) or not str(identifier).strip():
                identifier = product_id

            event = None
            started = datetime.now(timezone.utc)

            try:
                print(f"[geometry] loading {level} {identifier}")
                event = loader(str(identifier))

                table = interband_shift_table(
                    event,
                    master_band=args.master_band,
                    max_side=args.max_side,
                    max_shifts=(
                        int(args.max_shift),
                        int(args.max_shift),
                    ),
                )

                table["product_id"] = product_id
                table["level"] = level
                table["pair_delta_start_seconds"] = row.get(
                    "delta_start_seconds"
                )
                table["source_identifier"] = str(identifier)
                table["phiesta_version"] = phiesta.__version__
                table["extracted_utc"] = datetime.now(
                    timezone.utc
                ).isoformat()

                _append_deduplicated(metrics_path, table)

                ok_rows = int(
                    (table["status"].astype(str) == "ok").sum()
                )
                failed_rows = int(len(table) - ok_rows)

                record = {
                    "product_id": product_id,
                    "level": level,
                    "status": "SUCCESS",
                    "rows": int(len(table)),
                    "ok_rows": ok_rows,
                    "failed_rows": failed_rows,
                    "error": "",
                    "started_utc": started.isoformat(),
                    "finished_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
                _record_run(runs_path, record)
                success.add(key)

                print(
                    f"[geometry] SUCCESS {product_id} {level}: "
                    f"{ok_rows}/{len(table)} band rows ok"
                )

            except PermissionError as exc:
                record = {
                    "product_id": product_id,
                    "level": level,
                    "status": "ACCESS_DENIED",
                    "rows": 0,
                    "ok_rows": 0,
                    "failed_rows": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "started_utc": started.isoformat(),
                    "finished_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
                _record_run(runs_path, record)
                print(
                    f"[geometry] ACCESS_DENIED {product_id} {level}: "
                    f"{record['error']}"
                )

            except Exception as exc:
                record = {
                    "product_id": product_id,
                    "level": level,
                    "status": "FAILED",
                    "rows": 0,
                    "ok_rows": 0,
                    "failed_rows": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "started_utc": started.isoformat(),
                    "finished_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
                _record_run(runs_path, record)
                print(
                    f"[geometry] FAILED {product_id} {level}: "
                    f"{record['error']}"
                )

            finally:
                if args.cleanup and event is not None:
                    _safe_cleanup_event(event, args.cache_dir)

    final_metrics = _read_csv(metrics_path)
    final_runs = _read_csv(runs_path)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_pairs": int(len(usable)),
        "successful_product_levels": int(
            (final_runs["status"].astype(str) == "SUCCESS").sum()
        )
        if not final_runs.empty
        else 0,
        "failed_product_levels": int(
            (final_runs["status"].astype(str) == "FAILED").sum()
        )
        if not final_runs.empty
        else 0,
        "access_denied_product_levels": int(
            (final_runs["status"].astype(str) == "ACCESS_DENIED").sum()
        )
        if not final_runs.empty
        else 0,
        "metric_rows": int(len(final_metrics)),
        "master_band": args.master_band,
        "max_side": int(args.max_side),
        "max_shift": int(args.max_shift),
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
