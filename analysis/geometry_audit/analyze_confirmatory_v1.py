from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PRIMARY_SIDE = 4096
CHECK_SIDE = 2048
EXPECTED_SAMPLE_N = 96
EXPECTED_METRIC_ROWS = 1344
EXPECTED_PAIR_ROWS = 672

BAND_ORDER = ["PAN", "MS1", "MS2", "MS4", "MS5", "MS6", "MS7"]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Descriptive analysis of the frozen PhiSat-2 L1C confirmatory "
            "inter-band geometry audit. This script intentionally does not "
            "compute confidence intervals."
        )
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("analysis/geometry_audit/frozen/confirmatory_v1"),
        help="Frozen confirmatory_v1 directory.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/geometry_audit/confirmatory_v1_analysis"),
        help="Directory for derived tables and figures.",
    )
    return p.parse_args()


def weighted_quantile(values, weights, q):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[ok]
    w = w[ok]
    if len(v) == 0:
        return np.nan

    order = np.argsort(v)
    v = v[order]
    w = w[order]

    c = np.cumsum(w)
    target = float(q) * c[-1]
    idx = np.searchsorted(c, target, side="left")
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def weighted_mean(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    return float(np.average(v[ok], weights=w[ok]))


def weighted_fraction(mask, weights):
    m = np.asarray(mask)
    w = np.asarray(weights, dtype=float)
    ok = pd.notna(m) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    return float(np.average(m[ok].astype(float), weights=w[ok]))


def summarize_numeric(df, value_col, weight_col="analysis_design_weight"):
    x = pd.to_numeric(df[value_col], errors="coerce")
    w = pd.to_numeric(df[weight_col], errors="coerce")
    ok = x.notna() & w.notna() & (w > 0)
    x = x[ok]
    w = w[ok]
    if len(x) == 0:
        return {
            "n": 0,
            "weighted_mean": np.nan,
            "weighted_q25": np.nan,
            "weighted_median": np.nan,
            "weighted_q75": np.nan,
            "weighted_p90": np.nan,
            "weighted_p95": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(len(x)),
        "weighted_mean": weighted_mean(x, w),
        "weighted_q25": weighted_quantile(x, w, 0.25),
        "weighted_median": weighted_quantile(x, w, 0.50),
        "weighted_q75": weighted_quantile(x, w, 0.75),
        "weighted_p90": weighted_quantile(x, w, 0.90),
        "weighted_p95": weighted_quantile(x, w, 0.95),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def add_analysis_weights(df, weights):
    keep = weights[
        [
            "product_id",
            "stratum",
            "analysis_stratum_population_n",
            "analysis_inclusion_probability",
            "analysis_design_weight",
        ]
    ].copy()

    df = df.drop(
        columns=[
            "analysis_stratum_population_n",
            "analysis_inclusion_probability",
            "analysis_design_weight",
        ],
        errors="ignore",
    )
    out = df.merge(keep, on="product_id", how="left", suffixes=("", "_weight"))

    if out["analysis_design_weight"].isna().any():
        missing = sorted(
            out.loc[out["analysis_design_weight"].isna(), "product_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise RuntimeError(f"Missing analysis weights for products: {missing}")

    return out


def weighted_ecdf(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[ok]
    w = w[ok]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    y = np.cumsum(w) / np.sum(w)
    return x, y


def main():
    args = parse_args()
    root = args.root
    raw = root / "results_raw"
    out = args.out_dir
    figs = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(
        raw / "interband_multiscale.csv",
        dtype={"product_id": "string"},
    )
    pairs = pd.read_csv(
        raw / "interband_scale_pairs.csv",
        dtype={"product_id": "string"},
    )
    runs = pd.read_csv(
        raw / "interband_runs.csv",
        dtype={"product_id": "string"},
    )
    weights = pd.read_csv(
        root / "analysis_weights_v1.csv",
        dtype={"product_id": "string"},
    )

    for df in (metrics, pairs, runs, weights):
        df["product_id"] = (
            df["product_id"].astype("string").str.replace(r"\.0$", "", regex=True)
        )

    # Freeze integrity checks.
    if len(weights) != EXPECTED_SAMPLE_N or weights["product_id"].nunique() != EXPECTED_SAMPLE_N:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLE_N} unique sample products; "
            f"got rows={len(weights)}, unique={weights['product_id'].nunique()}."
        )
    if len(metrics) != EXPECTED_METRIC_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_METRIC_ROWS} metric rows; got {len(metrics)}."
        )
    if len(pairs) != EXPECTED_PAIR_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIR_ROWS} scale-pair rows; got {len(pairs)}."
        )
    if set(metrics["product_id"].unique()) != set(weights["product_id"].unique()):
        raise RuntimeError("Metric/sample product ID sets differ.")
    if set(pairs["product_id"].unique()) != set(weights["product_id"].unique()):
        raise RuntimeError("Scale-pair/sample product ID sets differ.")

    metrics = add_analysis_weights(metrics, weights)
    pairs = add_analysis_weights(pairs, weights)

    primary = metrics[
        (pd.to_numeric(metrics["analysis_max_side"], errors="coerce") == PRIMARY_SIDE)
        & metrics["status"].astype(str).str.lower().eq("ok")
    ].copy()

    check = metrics[
        (pd.to_numeric(metrics["analysis_max_side"], errors="coerce") == CHECK_SIDE)
        & metrics["status"].astype(str).str.lower().eq("ok")
    ].copy()

    pair_ok = pairs[
        pairs["scale_pair_available"].astype(str).str.lower().isin(["true", "1"])
    ].copy()

    # Primary band table.
    band_rows = []
    thresholds = [0.5, 1.0, 2.0, 5.0, 10.0, 80.0]

    for band in BAND_ORDER:
        g = primary[primary["target_band_name"].eq(band)].copy()
        row = {"target_band_name": band}
        row.update(
            {
                f"shift_{k}": v
                for k, v in summarize_numeric(g, "shift_px").items()
            }
        )
        row["dx_weighted_mean_px"] = weighted_mean(
            pd.to_numeric(g["dx_px"], errors="coerce"),
            g["analysis_design_weight"],
        )
        row["dy_weighted_mean_px"] = weighted_mean(
            pd.to_numeric(g["dy_px"], errors="coerce"),
            g["analysis_design_weight"],
        )
        for t in thresholds:
            row[f"weighted_fraction_shift_le_{str(t).replace('.', 'p')}px"] = (
                weighted_fraction(
                    pd.to_numeric(g["shift_px"], errors="coerce") <= t,
                    g["analysis_design_weight"],
                )
            )
        row["weighted_fraction_exceeds_80px"] = weighted_fraction(
            g["exceeds_plausibility_flag"].astype(str).str.lower().isin(["true", "1"]),
            g["analysis_design_weight"],
        )
        band_rows.append(row)

    band_summary = pd.DataFrame(band_rows)
    band_summary.to_csv(out / "primary_band_summary.csv", index=False)

    # QC table by band.
    qc_rows = []
    for band in BAND_ORDER:
        g = primary[primary["target_band_name"].eq(band)].copy()
        response = summarize_numeric(g, "response")
        corr_gain = summarize_numeric(g, "corr_gain")
        qc_rows.append(
            {
                "target_band_name": band,
                "n_valid_primary": int(len(g)),
                "response_weighted_median": response["weighted_median"],
                "response_weighted_p90": response["weighted_p90"],
                "corr_gain_weighted_median": corr_gain["weighted_median"],
                "weighted_fraction_corr_improved": weighted_fraction(
                    g["corr_improved"].astype(str).str.lower().isin(["true", "1"]),
                    g["analysis_design_weight"],
                ),
                "weighted_fraction_exceeds_plausibility_flag": weighted_fraction(
                    g["exceeds_plausibility_flag"].astype(str).str.lower().isin(["true", "1"]),
                    g["analysis_design_weight"],
                ),
            }
        )
    pd.DataFrame(qc_rows).to_csv(out / "primary_qc_by_band.csv", index=False)

    # Scale stability table by band.
    scale_rows = []
    for band in BAND_ORDER:
        g = pair_ok[pair_ok["target_band_name"].eq(band)].copy()
        row = {"target_band_name": band}
        row.update(
            {
                f"scale_delta_{k}": v
                for k, v in summarize_numeric(g, "scale_delta_px").items()
            }
        )
        for t in [1.0, 2.0, 5.0, 10.0, 80.0]:
            row[f"weighted_fraction_scale_delta_le_{str(t).replace('.', 'p')}px"] = (
                weighted_fraction(
                    pd.to_numeric(g["scale_delta_px"], errors="coerce") <= t,
                    g["analysis_design_weight"],
                )
            )
        scale_rows.append(row)
    scale_summary = pd.DataFrame(scale_rows)
    scale_summary.to_csv(out / "scale_stability_by_band.csv", index=False)

    # Acquisition-level summary: keeps the acquisition as the independent unit.
    acq_rows = []
    for pid, g_all in metrics.groupby("product_id", sort=False):
        w = float(
            weights.loc[weights["product_id"].eq(pid), "analysis_design_weight"].iloc[0]
        )
        g = g_all[
            (pd.to_numeric(g_all["analysis_max_side"], errors="coerce") == PRIMARY_SIDE)
            & g_all["status"].astype(str).str.lower().eq("ok")
        ].copy()
        shifts = pd.to_numeric(g["shift_px"], errors="coerce").dropna()
        responses = pd.to_numeric(g["response"], errors="coerce").dropna()
        gains = pd.to_numeric(g["corr_gain"], errors="coerce").dropna()

        acq_rows.append(
            {
                "product_id": pid,
                "analysis_design_weight": w,
                "valid_primary_bands": int(len(shifts)),
                "median_shift_px_across_bands": float(shifts.median()) if len(shifts) else np.nan,
                "max_shift_px_across_bands": float(shifts.max()) if len(shifts) else np.nan,
                "median_response_across_bands": float(responses.median()) if len(responses) else np.nan,
                "median_corr_gain_across_bands": float(gains.median()) if len(gains) else np.nan,
            }
        )

    acq = pd.DataFrame(acq_rows)
    acq.to_csv(out / "acquisition_summary.csv", index=False)

    # Overall descriptive summary.
    overall = {
        "sample_n": int(len(weights)),
        "run_status_counts": {
            str(k): int(v) for k, v in runs["status"].value_counts().to_dict().items()
        },
        "metric_rows": int(len(metrics)),
        "primary_valid_rows": int(len(primary)),
        "stability_valid_rows": int(len(check)),
        "scale_pair_available_rows": int(len(pair_ok)),
        "primary_shift_px": summarize_numeric(primary, "shift_px"),
        "primary_response": summarize_numeric(primary, "response"),
        "primary_corr_gain": summarize_numeric(primary, "corr_gain"),
        "scale_delta_px": summarize_numeric(pair_ok, "scale_delta_px"),
        "acquisition_median_shift_px": summarize_numeric(
            acq, "median_shift_px_across_bands"
        ),
        "notes": [
            "Point estimates use analysis_design_weight from analysis_weights_v1.csv.",
            "No failed measurement is imputed.",
            "4096 max-side is the frozen primary analysis; 2048 is a stability diagnostic.",
            "This script intentionally does not compute confidence intervals.",
            "A naive within-original-stratum bootstrap is inappropriate because many original strata contain one sampled acquisition.",
            "Large shifts and poor cross-scale agreement must be interpreted jointly with response/correlation QC; they are estimator outputs, not automatically physical displacements.",
        ],
    }
    (out / "descriptive_summary.json").write_text(
        json.dumps(overall, indent=2) + "\n",
        encoding="utf-8",
    )

    # Figure 1: weighted median primary shift by band.
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(BAND_ORDER))
    vals = (
        band_summary.set_index("target_band_name")
        .loc[BAND_ORDER, "shift_weighted_median"]
        .to_numpy(dtype=float)
    )
    ax.bar(x, vals)
    ax.set_xticks(x, BAND_ORDER)
    ax.set_ylabel("Weighted median shift (px)")
    ax.set_xlabel("Target band relative to RED/MS3")
    ax.set_title("L1C residual inter-band shift: primary 4096 max-side analysis")
    fig.tight_layout()
    fig.savefig(figs / "primary_weighted_median_shift_by_band.pdf")
    fig.savefig(figs / "primary_weighted_median_shift_by_band.png", dpi=180)
    plt.close(fig)

    # Figure 2: weighted ECDF of primary shift by band.
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for band in BAND_ORDER:
        g = primary[primary["target_band_name"].eq(band)]
        x_ecdf, y_ecdf = weighted_ecdf(
            pd.to_numeric(g["shift_px"], errors="coerce"),
            g["analysis_design_weight"],
        )
        if len(x_ecdf):
            # step() uses matplotlib's automatic color cycle; no colors are specified.
            ax.step(x_ecdf, y_ecdf, where="post", label=band)
    ax.set_xscale("log")
    ax.set_xlabel("Shift magnitude (px, log scale)")
    ax.set_ylabel("Weighted cumulative fraction")
    ax.set_title("Weighted ECDF of L1C inter-band shift")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figs / "primary_shift_weighted_ecdf_by_band.pdf")
    fig.savefig(figs / "primary_shift_weighted_ecdf_by_band.png", dpi=180)
    plt.close(fig)

    # Figure 3: weighted median scale disagreement by band.
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    vals = (
        scale_summary.set_index("target_band_name")
        .loc[BAND_ORDER, "scale_delta_weighted_median"]
        .to_numpy(dtype=float)
    )
    ax.bar(x, vals)
    ax.set_xticks(x, BAND_ORDER)
    ax.set_ylabel("|shift4096 - shift2048| (px)")
    ax.set_xlabel("Target band relative to RED/MS3")
    ax.set_title("Scale sensitivity of the inter-band shift estimator")
    fig.tight_layout()
    fig.savefig(figs / "scale_delta_weighted_median_by_band.pdf")
    fig.savefig(figs / "scale_delta_weighted_median_by_band.png", dpi=180)
    plt.close(fig)

    print("=" * 80)
    print("CONFIRMATORY V1 DESCRIPTIVE ANALYSIS PASS")
    print("=" * 80)
    print(f"Sample acquisitions:        {len(weights)}")
    print(f"Primary valid rows:         {len(primary)}")
    print(f"Stability valid rows:       {len(check)}")
    print(f"Available scale pairs:      {len(pair_ok)}")
    print()
    print("Primary weighted shift medians by band:")
    print(
        band_summary[
            ["target_band_name", "shift_n", "shift_weighted_median", "shift_weighted_p95"]
        ].to_string(index=False)
    )
    print()
    print("Scale-delta weighted medians by band:")
    print(
        scale_summary[
            ["target_band_name", "scale_delta_n", "scale_delta_weighted_median", "scale_delta_weighted_p95"]
        ].to_string(index=False)
    )
    print()
    print("Wrote:", out)


if __name__ == "__main__":
    main()
