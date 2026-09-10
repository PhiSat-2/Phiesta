from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from phiesta import connect_insula
from phiesta.remote.catalog_geometry import (
    catalog_geo_from_feature,
    get_catalog_center,
)
from phiesta.remote.constants import (
    PHISAT2_L1A_COLLECTION,
    PHISAT2_L1C_COLLECTION,
)


PRODUCT_RE = re.compile(
    r"PHISAT-2_(?:L0|L1A|L1C|L1)_0*(\d+)_",
    flags=re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a strict paired L1A/L1C PhiSat-2 geometry cohort."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data"),
        help="Phiesta cache root used by connect_insula().",
    )
    parser.add_argument("--results-per-page", type=int, default=100)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional smoke-test limit per collection.",
    )
    parser.add_argument(
        "--max-pair-time-delta-seconds",
        type=float,
        default=120.0,
    )
    return parser.parse_args()


def _first_nonempty(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _extract_product_id(feature: dict) -> str | None:
    props = feature.get("properties") or {}
    for value in (
        props.get("productIdentifier"),
        props.get("filename"),
        props.get("identifier"),
        feature.get("id"),
    ):
        if value is None:
            continue
        match = PRODUCT_RE.search(str(value))
        if match:
            return str(int(match.group(1)))
    return None


def _feature_row(feature: dict, *, level: str) -> dict:
    props = feature.get("properties") or {}
    catalog_geo = catalog_geo_from_feature(feature)

    center = None
    if catalog_geo:
        try:
            center = get_catalog_center(catalog_geo, order="lonlat")
        except Exception:
            center = None

    return {
        "level": level,
        "product_id": _extract_product_id(feature),
        "identifier": _first_nonempty(
            props.get("productIdentifier"),
            props.get("identifier"),
            props.get("filename"),
        ),
        "filename": _first_nonempty(
            props.get("filename"),
            props.get("productIdentifier"),
        ),
        "feature_id": feature.get("id"),
        "start_datetime": _first_nonempty(
            props.get("startDate"),
            props.get("start_datetime"),
            props.get("datetime"),
            catalog_geo.get("start_datetime") if catalog_geo else None,
        ),
        "completion_datetime": _first_nonempty(
            props.get("completionDate"),
            props.get("completion_datetime"),
            props.get("end"),
        ),
        "center_lon": center[0] if center else None,
        "center_lat": center[1] if center else None,
    }


def scan_collection(
    client,
    *,
    collection: str,
    level: str,
    results_per_page: int,
    max_pages: int | None,
) -> pd.DataFrame:
    rows = []
    page = 0

    while True:
        data = client.search_ref_data(
            ref_data_collection=collection,
            page=page,
            results_per_page=results_per_page,
        )
        features = data.get("features", [])
        if not features:
            break

        rows.extend(_feature_row(feature, level=level) for feature in features)

        page_info = data.get("page") or {}
        total_pages = page_info.get("totalPages")
        suffix = f"/{total_pages}" if total_pages is not None else ""
        print(
            f"[{level}] page {page + 1}{suffix}: "
            f"{len(features)} features, {len(rows)} total"
        )

        page += 1
        if max_pages is not None and page >= max_pages:
            break
        if total_pages is not None and page >= int(total_pages):
            break

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table["product_id"] = table["product_id"].astype("string")
    table["start_datetime"] = pd.to_datetime(
        table["start_datetime"], utc=True, errors="coerce"
    )
    table["completion_datetime"] = pd.to_datetime(
        table["completion_datetime"], utc=True, errors="coerce"
    )

    return table.sort_values(
        ["product_id", "start_datetime"],
        na_position="last",
    ).reset_index(drop=True)


def _representatives(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                f"{prefix}_count",
                f"{prefix}_identifier",
                f"{prefix}_filename",
                f"{prefix}_feature_id",
                f"{prefix}_start_datetime",
                f"{prefix}_completion_datetime",
                f"{prefix}_center_lon",
                f"{prefix}_center_lat",
            ]
        )

    valid = table[table["product_id"].notna()].copy()

    counts = (
        valid.groupby("product_id", dropna=False)
        .size()
        .rename(f"{prefix}_count")
        .reset_index()
    )

    reps = (
        valid.sort_values(
            ["product_id", "start_datetime", "identifier"],
            na_position="last",
        )
        .groupby("product_id", as_index=False)
        .first()
    )

    keep = [
        "product_id",
        "identifier",
        "filename",
        "feature_id",
        "start_datetime",
        "completion_datetime",
        "center_lon",
        "center_lat",
    ]
    reps = reps[keep].rename(
        columns={
            col: f"{prefix}_{col}"
            for col in keep
            if col != "product_id"
        }
    )
    return counts.merge(reps, on="product_id", how="left")


def pair_catalogs(
    l1a: pd.DataFrame,
    l1c: pd.DataFrame,
    *,
    max_pair_time_delta_seconds: float,
) -> pd.DataFrame:
    a = _representatives(l1a, "l1a")
    c = _representatives(l1c, "l1c")
    cohort = a.merge(c, on="product_id", how="outer")

    for col in ("l1a_count", "l1c_count"):
        cohort[col] = cohort[col].fillna(0).astype(int)

    def status(row):
        na = int(row["l1a_count"])
        nc = int(row["l1c_count"])
        if na == 1 and nc == 1:
            return "paired_unique"
        if na == 0 and nc > 0:
            return "l1c_only"
        if nc == 0 and na > 0:
            return "l1a_only"
        if na > 1 and nc > 1:
            return "duplicate_both"
        if na > 1:
            return "duplicate_l1a"
        if nc > 1:
            return "duplicate_l1c"
        return "unresolved"

    cohort["pair_status"] = cohort.apply(status, axis=1)

    a_time = pd.to_datetime(
        cohort.get("l1a_start_datetime"), utc=True, errors="coerce"
    )
    c_time = pd.to_datetime(
        cohort.get("l1c_start_datetime"), utc=True, errors="coerce"
    )
    cohort["delta_start_seconds"] = (a_time - c_time).dt.total_seconds().abs()
    cohort["timing_suspicious"] = (
        (cohort["pair_status"] == "paired_unique")
        & cohort["delta_start_seconds"].notna()
        & (
            cohort["delta_start_seconds"]
            > float(max_pair_time_delta_seconds)
        )
    )
    cohort["strict_usable"] = (
        (cohort["pair_status"] == "paired_unique")
        & (~cohort["timing_suspicious"])
    )

    cohort["_numeric_id"] = pd.to_numeric(
        cohort["product_id"], errors="coerce"
    )
    cohort = cohort.sort_values(
        ["_numeric_id", "product_id"],
        na_position="last",
    ).drop(columns="_numeric_id")
    return cohort.reset_index(drop=True)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = connect_insula(cache_dir=args.cache_dir)

    l1a = scan_collection(
        client,
        collection=PHISAT2_L1A_COLLECTION,
        level="L1A",
        results_per_page=args.results_per_page,
        max_pages=args.max_pages,
    )
    l1c = scan_collection(
        client,
        collection=PHISAT2_L1C_COLLECTION,
        level="L1C",
        results_per_page=args.results_per_page,
        max_pages=args.max_pages,
    )

    cohort = pair_catalogs(
        l1a,
        l1c,
        max_pair_time_delta_seconds=args.max_pair_time_delta_seconds,
    )

    l1a.to_csv(args.out_dir / "catalog_l1a.csv", index=False)
    l1c.to_csv(args.out_dir / "catalog_l1c.csv", index=False)
    cohort.to_csv(args.out_dir / "cohort.csv", index=False)
    cohort[~cohort["strict_usable"]].to_csv(
        args.out_dir / "pairing_issues.csv", index=False
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "l1a_catalog_rows": int(len(l1a)),
        "l1c_catalog_rows": int(len(l1c)),
        "cohort_ids": int(len(cohort)),
        "strict_usable_pairs": int(cohort["strict_usable"].sum()),
        "timing_suspicious_pairs": int(
            cohort["timing_suspicious"].sum()
        ),
        "pair_status_counts": {
            str(k): int(v)
            for k, v in cohort["pair_status"].value_counts().items()
        },
        "max_pair_time_delta_seconds": float(
            args.max_pair_time_delta_seconds
        ),
        "max_pages": args.max_pages,
    }

    (args.out_dir / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2))
    print()
    print("Wrote:", args.out_dir / "cohort.csv")


if __name__ == "__main__":
    main()
