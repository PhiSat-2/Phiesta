from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(r"analysis\geometry_audit\frozen\confirmatory_v1\results_raw")
OUT = Path(r"analysis\geometry_audit\confirmatory_v1_analysis")
OUT.mkdir(parents=True, exist_ok=True)

# Public focal-plane line positions, RED/MS3 is the master.
DETECTOR_LINE = {
    "PAN": 1536,
    "MS1": 2176,
    "MS2": 1964,
    "MS3": 1748,
    "MS4": 1108,
    "MS5": 896,
    "MS6": 680,
    "MS7": 1324,
}

m = pd.read_csv(
    RAW / "interband_multiscale.csv",
    dtype={"product_id": "string"},
)

pairs = pd.read_csv(
    RAW / "interband_scale_pairs.csv",
    dtype={"product_id": "string"},
)

# Frozen primary endpoint.
x = m[
    (m["analysis_max_side"] == 4096)
    & (m["status"].str.lower() == "ok")
].copy()

q = pairs[[
    "product_id",
    "target_band",
    "scale_pair_available",
    "dx_px_check",
    "dy_px_check",
    "shift_px_check",
    "scale_delta_px",
]].copy()

x = x.merge(
    q,
    on=["product_id", "target_band"],
    how="left",
    validate="one_to_one",
)

# Descriptive / exploratory diagnostics.
x["tail80"] = x["shift_px"] > 80
x["rel_scale_delta"] = x["scale_delta_px"] / x["shift_px"]

x["cross_scale_repeatable_10pct"] = (
    x["tail80"]
    & x["scale_pair_available"].astype(str).str.lower().isin(["true", "1"])
    & x["rel_scale_delta"].le(0.10)
)

x["detector_dy_expected"] = x["target_band_name"].map(
    lambda b: DETECTOR_LINE["MS3"] - DETECTOR_LINE.get(b, DETECTOR_LINE["MS3"])
)

x["detector_abs_error_px"] = (
    x["dy_px"] - x["detector_dy_expected"]
).abs()

tail_products = sorted(
    x.loc[x["tail80"], "product_id"].astype(str).unique()
)

rows = []

for pid in tail_products:
    g = x[x["product_id"] == pid].copy()
    t = g[g["tail80"]].copy()
    r = g[g["cross_scale_repeatable_10pct"]].copy()

    # Vector coherence among repeatable large shifts.
    median_pairwise = np.nan
    min_pairwise = np.nan
    median_nearest = np.nan

    if len(r) >= 2:
        v = r[["dx_px", "dy_px"]].to_numpy(float)
        d = np.sqrt(
            ((v[:, None, :] - v[None, :, :]) ** 2).sum(axis=2)
        )

        tri = d[np.triu_indices(len(v), 1)]

        d_near = d.copy()
        np.fill_diagonal(d_near, np.inf)

        median_pairwise = float(np.median(tri))
        min_pairwise = float(np.min(tri))
        median_nearest = float(np.median(d_near.min(axis=1)))

    # Detector-geometry similarity.
    n_det25 = int((g["detector_abs_error_px"] <= 25).sum())
    n_det75 = int((g["detector_abs_error_px"] <= 75).sum())
    n_det150 = int((g["detector_abs_error_px"] <= 150).sum())

    # Very deliberately descriptive phenotype, not a confirmatory label.
    if (
        len(r) >= 3
        and np.isfinite(median_nearest)
        and median_nearest <= 10
    ):
        phenotype = "cross-band coherent offset candidate"
    elif n_det75 >= 3:
        phenotype = "detector-geometry-like candidate"
    elif (
        len(t) >= 4
        and t["response"].median() < 0.01
        and t["scale_delta_px"].median() > 80
    ):
        phenotype = "estimator/acquisition failure-like"
    else:
        phenotype = "mixed / isolated tail"

    first = g.iloc[0]

    rows.append({
        "product_id": pid,
        "sample_order": first.get("sample_order", np.nan),
        "start_datetime": first.get("start_datetime", ""),
        "center_lon": first.get("center_lon", np.nan),
        "center_lat": first.get("center_lat", np.nan),
        "stratum": first.get("stratum", ""),
        "design_weight": first.get("design_weight", np.nan),

        "n_valid_primary_bands": int(len(g)),
        "n_tail_gt80": int(len(t)),
        "tail_bands": ",".join(t["target_band_name"].astype(str)),

        "max_shift_px": float(t["shift_px"].max()),
        "median_tail_shift_px": float(t["shift_px"].median()),

        "median_tail_response": float(t["response"].median()),
        "max_tail_response": float(t["response"].max()),

        "median_tail_corr_gain": float(t["corr_gain"].median()),
        "n_tail_corr_improved": int((t["corr_gain"] > 0).sum()),

        "median_tail_scale_delta_px":
            float(t["scale_delta_px"].median(skipna=True)),
        "n_tail_scale_delta_le80":
            int((t["scale_delta_px"] <= 80).sum()),

        "n_large_cross_scale_repeatable_10pct": int(len(r)),
        "repeatable_large_bands":
            ",".join(r["target_band_name"].astype(str)),

        "median_pairwise_repeatable_vector_distance_px":
            median_pairwise,
        "min_pairwise_repeatable_vector_distance_px":
            min_pairwise,
        "median_nearest_repeatable_vector_distance_px":
            median_nearest,

        "n_within_25px_detector_model": n_det25,
        "n_within_75px_detector_model": n_det75,
        "n_within_150px_detector_model": n_det150,

        "exploratory_phenotype": phenotype,
    })

res = pd.DataFrame(rows)

# Put the most structured anomalies first.
priority = {
    "cross-band coherent offset candidate": 0,
    "detector-geometry-like candidate": 1,
    "estimator/acquisition failure-like": 2,
    "mixed / isolated tail": 3,
}
res["_priority"] = res["exploratory_phenotype"].map(priority)

res = res.sort_values(
    ["_priority", "n_tail_gt80", "max_shift_px"],
    ascending=[True, False, False],
).drop(columns="_priority")

out_csv = OUT / "tail_acquisition_typology_v1.csv"
res.to_csv(out_csv, index=False)

print("=" * 110)
print("TAIL ACQUISITION TYPOLOGY V1")
print("=" * 110)
print("Tail acquisitions:", len(res))
print()
print(
    res[[
        "product_id",
        "n_tail_gt80",
        "tail_bands",
        "median_tail_response",
        "median_tail_scale_delta_px",
        "n_large_cross_scale_repeatable_10pct",
        "median_nearest_repeatable_vector_distance_px",
        "n_within_75px_detector_model",
        "exploratory_phenotype",
    ]].to_string(index=False)
)

print("\nPhenotype counts:")
print(res["exploratory_phenotype"].value_counts().to_string())

print("\nWrote:", out_csv)
print()
print("NOTE: phenotype labels are exploratory descriptive summaries only;")
print("      the 80 px flag was pre-specified, but the 10% repeatability and")
print("      vector-coherence rules are post-hoc diagnostics.")
