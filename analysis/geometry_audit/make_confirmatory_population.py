from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _norm_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Freeze the untouched L1C population for confirmatory geometry audit."
    )
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--exclusions", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(args.catalog, dtype={"product_id": "string"})
    exclusions = pd.read_csv(args.exclusions, dtype={"product_id": "string"})

    if "product_id" not in catalog.columns:
        raise ValueError("Catalog must contain product_id.")
    if "product_id" not in exclusions.columns:
        raise ValueError("Exclusion manifest must contain product_id.")

    catalog = catalog.copy()
    exclusions = exclusions.copy()
    catalog["product_id"] = _norm_id(catalog["product_id"])
    exclusions["product_id"] = _norm_id(exclusions["product_id"])

    if catalog["product_id"].duplicated().any():
        dup = sorted(catalog.loc[catalog["product_id"].duplicated(), "product_id"].unique())
        raise ValueError(f"Catalog has duplicate product IDs: {dup[:10]}")

    excluded_ids = set(exclusions["product_id"].dropna().astype(str))
    catalog_ids = set(catalog["product_id"].astype(str))
    present_exclusions = sorted(excluded_ids.intersection(catalog_ids))
    missing_exclusions = sorted(excluded_ids.difference(catalog_ids))

    population = catalog[~catalog["product_id"].isin(excluded_ids)].copy()

    out_csv = args.out_dir / "catalog_l1c_confirmatory_eligible_v1.csv"
    population.to_csv(out_csv, index=False)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_catalog": str(args.catalog),
        "exclusion_manifest": str(args.exclusions),
        "source_catalog_n": int(len(catalog)),
        "requested_exclusion_n": int(len(excluded_ids)),
        "present_exclusion_n": int(len(present_exclusions)),
        "missing_exclusion_n": int(len(missing_exclusions)),
        "confirmatory_eligible_n": int(len(population)),
        "present_exclusion_ids": present_exclusions,
        "missing_exclusion_ids": missing_exclusions,
        "principle": (
            "Products whose geometry outcomes were inspected during method development "
            "are excluded before drawing the confirmatory probability sample."
        ),
    }
    out_json = args.out_dir / "confirmatory_population_summary_v1.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
