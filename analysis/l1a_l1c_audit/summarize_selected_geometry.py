from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs/l1a_l1c_audit_selected")

local = pd.read_csv(OUT / "local_shift_field.csv")
local = local[local["response"] >= 0.2].copy()

# shift vector médian par produit / niveau / bande = composante dominante globale
med = (
    local.groupby(["product_id", "level", "band"])[["dy", "dx"]]
    .median()
    .rename(columns={"dy": "global_median_dy", "dx": "global_median_dx"})
)

local = local.join(med, on=["product_id", "level", "band"])

# résidu après retrait de la composante globale
local["residual_dy"] = local["dy"] - local["global_median_dy"]
local["residual_dx"] = local["dx"] - local["global_median_dx"]
local["residual_mag"] = np.sqrt(local["residual_dy"]**2 + local["residual_dx"]**2)

def q90(x):
    return np.quantile(x, 0.90)

def q95(x):
    return np.quantile(x, 0.95)

print("\n===== ALL SELECTED: RAW LOCAL SHIFT =====")
s1 = (
    local.groupby("level")["shift_mag"]
    .agg(["count", "mean", "median", q90, q95, "max"])
    .reset_index()
)
print(s1.to_string(index=False))
s1.to_csv(OUT / "final_raw_local_shift_summary.csv", index=False)

print("\n===== ALL SELECTED: RESIDUAL AFTER GLOBAL BAND SHIFT REMOVAL =====")
s2 = (
    local.groupby("level")["residual_mag"]
    .agg(["count", "mean", "median", q90, q95, "max"])
    .reset_index()
)
print(s2.to_string(index=False))
s2.to_csv(OUT / "final_local_residual_after_global_shift_summary.csv", index=False)

clean = local[~local["product_id"].isin([5978, 6045])].copy()

print("\n===== CLEAN SET WITHOUT 5978 AND 6045: RAW LOCAL SHIFT =====")
s3 = (
    clean.groupby("level")["shift_mag"]
    .agg(["count", "mean", "median", q90, q95, "max"])
    .reset_index()
)
print(s3.to_string(index=False))
s3.to_csv(OUT / "final_clean_raw_local_shift_summary.csv", index=False)

print("\n===== CLEAN SET WITHOUT 5978 AND 6045: RESIDUAL AFTER GLOBAL BAND SHIFT REMOVAL =====")
s4 = (
    clean.groupby("level")["residual_mag"]
    .agg(["count", "mean", "median", q90, q95, "max"])
    .reset_index()
)
print(s4.to_string(index=False))
s4.to_csv(OUT / "final_clean_local_residual_after_global_shift_summary.csv", index=False)

print("\n===== PER PRODUCT CLEAN/OUTLIER VIEW =====")
s5 = (
    local.groupby(["product_id", "level"])
    .agg(
        n=("shift_mag", "count"),
        raw_median=("shift_mag", "median"),
        raw_mean=("shift_mag", "mean"),
        residual_median=("residual_mag", "median"),
        residual_mean=("residual_mag", "mean"),
        residual_max=("residual_mag", "max"),
    )
    .reset_index()
)
print(s5.to_string(index=False))
s5.to_csv(OUT / "final_per_product_shift_vs_residual.csv", index=False)
