from __future__ import annotations

import argparse
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Draw a reproducible probability sample of PhiSat-2 L1C products, "
            "stratified jointly by time and coarse equal-area geography."
        )
    )
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=96)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--time-bins", type=int, default=6)
    return p.parse_args()


def _allocate_with_minimum_one(counts: pd.Series, n_total: int) -> pd.Series:
    counts = counts.astype(int)
    h = len(counts)
    if n_total < h:
        raise ValueError(
            f"sample-size={n_total} is smaller than the {h} occupied strata. "
            "Increase --sample-size so every occupied stratum has non-zero "
            "inclusion probability."
        )

    alloc = pd.Series(1, index=counts.index, dtype=int)
    capacity = counts - alloc
    remaining = int(n_total - alloc.sum())

    while remaining > 0 and int(capacity.sum()) > 0:
        active = capacity[capacity > 0]
        raw = remaining * active / active.sum()
        base = np.floor(raw).astype(int)
        base = np.minimum(base, active)

        if int(base.sum()) > 0:
            alloc.loc[base.index] += base
            capacity.loc[base.index] -= base
            remaining -= int(base.sum())
            continue

        frac = (raw - np.floor(raw)).sort_values(ascending=False)
        for key in frac.index:
            if remaining <= 0:
                break
            if capacity.loc[key] <= 0:
                continue
            alloc.loc[key] += 1
            capacity.loc[key] -= 1
            remaining -= 1

    return alloc


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.catalog, dtype={"product_id": "string"})
    required = {"product_id", "start_datetime", "center_lon", "center_lat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Catalog missing columns: {sorted(missing)}")

    df = df.copy()
    df["product_id"] = (
        df["product_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    )
    df["start_datetime"] = pd.to_datetime(
        df["start_datetime"], utc=True, errors="coerce"
    )
    df["center_lon"] = pd.to_numeric(df["center_lon"], errors="coerce")
    df["center_lat"] = pd.to_numeric(df["center_lat"], errors="coerce")

    df = df.sort_values(["product_id", "start_datetime"]).drop_duplicates(
        "product_id", keep="first"
    )

    valid = (
        df["product_id"].notna()
        & df["start_datetime"].notna()
        & df["center_lon"].between(-180, 180)
        & df["center_lat"].between(-90, 90)
    )
    population = df[valid].copy().reset_index(drop=True)
    if population.empty:
        raise ValueError("No valid rows remain after time/location QC.")

    sample_size = min(int(args.sample_size), len(population))
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive.")

    t0 = population["start_datetime"].min()
    t1 = population["start_datetime"].max()
    if t0 == t1:
        population["time_bin"] = 0
        time_edges = [t0.isoformat(), t1.isoformat()]
    else:
        t_ns = population["start_datetime"].astype("int64")
        n_time_bins = int(args.time_bins)
        lo_ns = int(t_ns.min())
        hi_ns = int(t_ns.max())

        # Build edges with integer arithmetic. np.linspace on ~1e18-ns
        # timestamps goes through float64 and can round the final (max + 1)
        # edge back onto max, excluding the latest acquisition with right=False.
        span_ns = hi_ns - lo_ns + 1
        edges_ns = np.array(
            [
                lo_ns + (span_ns * i) // n_time_bins
                for i in range(n_time_bins)
            ]
            + [hi_ns + 1],
            dtype=np.int64,
        )
        if np.any(np.diff(edges_ns) <= 0):
            raise ValueError(
                "Temporal range is too narrow for the requested number of bins."
            )

        time_bin = pd.cut(
            t_ns,
            bins=edges_ns,
            labels=False,
            include_lowest=True,
            right=False,
        )
        if time_bin.isna().any():
            bad = int(time_bin.isna().sum())
            raise ValueError(
                f"Temporal stratification left {bad} acquisition(s) unassigned."
            )
        population["time_bin"] = time_bin.astype(int)
        time_edges = [
            pd.Timestamp(int(v), tz="UTC").isoformat()
            for v in edges_ns
        ]

    sin_lat = np.sin(np.deg2rad(population["center_lat"].to_numpy()))
    lat_edges = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
    population["lat_band"] = pd.cut(
        sin_lat,
        bins=lat_edges,
        labels=False,
        include_lowest=True,
    ).astype(int)

    lon_edges = np.array([-180.0, -90.0, 0.0, 90.0, 180.0000001])
    population["lon_band"] = pd.cut(
        population["center_lon"],
        bins=lon_edges,
        labels=False,
        include_lowest=True,
        right=False,
    ).astype(int)

    population["stratum"] = (
        "T" + population["time_bin"].astype(str)
        + "_A" + population["lat_band"].astype(str)
        + "_L" + population["lon_band"].astype(str)
    )

    counts = population.groupby("stratum").size().sort_index()
    allocation = _allocate_with_minimum_one(counts, sample_size)

    selected_parts = []
    strata_rows = []

    for stratum, n_population in counts.items():
        n_sample = int(allocation.loc[stratum])
        group = population[population["stratum"] == stratum].copy()

        local_seed = (
            int(args.seed)
            + int(zlib.crc32(str(stratum).encode("utf-8")))
        ) % (2**32 - 1)

        chosen = group.sample(
            n=n_sample,
            replace=False,
            random_state=local_seed,
        ).copy()

        inclusion_probability = n_sample / float(n_population)
        chosen["stratum_population_n"] = int(n_population)
        chosen["stratum_sample_n"] = int(n_sample)
        chosen["inclusion_probability"] = inclusion_probability
        chosen["design_weight"] = 1.0 / inclusion_probability
        selected_parts.append(chosen)

        strata_rows.append(
            {
                "stratum": stratum,
                "population_n": int(n_population),
                "sample_n": int(n_sample),
                "inclusion_probability": inclusion_probability,
                "design_weight": 1.0 / inclusion_probability,
            }
        )

    sample = pd.concat(selected_parts, ignore_index=True)
    sample = sample.sort_values(
        ["start_datetime", "product_id"]
    ).reset_index(drop=True)
    sample["sample_order"] = np.arange(1, len(sample) + 1)

    strata = pd.DataFrame(strata_rows).sort_values("stratum")
    population.to_csv(args.out_dir / "l1c_sampling_population.csv", index=False)
    strata.to_csv(args.out_dir / "l1c_sampling_strata.csv", index=False)
    sample.to_csv(args.out_dir / "l1c_primary_sample.csv", index=False)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalog": str(args.catalog),
        "population_n": int(len(population)),
        "sample_n": int(len(sample)),
        "seed": int(args.seed),
        "time_bins": int(args.time_bins),
        "occupied_strata": int(len(strata)),
        "time_start": t0.isoformat(),
        "time_end": t1.isoformat(),
        "time_edges": time_edges,
        "latitude_stratification": (
            "3 fixed equal-area bands in sin(latitude): [-1,-1/3,1/3,1]"
        ),
        "longitude_stratification": (
            "4 fixed 90-degree sectors: [-180,-90,0,90,180]"
        ),
        "allocation": (
            "minimum 1 per occupied stratum, then proportional to remaining "
            "stratum population; simple random sampling without replacement "
            "within stratum"
        ),
    }
    (args.out_dir / "l1c_sampling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print()
    print("Wrote:", args.out_dir / "l1c_primary_sample.csv")


if __name__ == "__main__":
    main()
