from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(r"analysis\geometry_audit\frozen\confirmatory_v1")
ANA  = Path(r"analysis\geometry_audit\confirmatory_v1_analysis")

weights = pd.read_csv(
    ROOT / "analysis_weights_v1.csv",
    dtype={"product_id": "string"},
)

typ = pd.read_csv(
    ANA / "tail_acquisition_typology_v1.csv",
    dtype={"product_id": "string"},
)

weights["product_id"] = weights["product_id"].astype("string")
typ["product_id"] = typ["product_id"].astype("string")

a = weights[[
    "product_id",
    "stratum",
    "analysis_stratum_population_n",
    "stratum_sample_n",
    "analysis_design_weight",
]].copy()

a = a.merge(
    typ[["product_id", "n_tail_gt80"]],
    on="product_id",
    how="left",
    validate="one_to_one",
)

a["n_tail_gt80"] = a["n_tail_gt80"].fillna(0).astype(int)

# Validate the corrected frozen design.
check = (
    a.groupby("stratum")
    .agg(
        sample_rows=("product_id", "size"),
        N_h=("analysis_stratum_population_n", "first"),
        declared_n_h=("stratum_sample_n", "first"),
    )
)

if not (check["sample_rows"] == check["declared_n_h"]).all():
    raise RuntimeError("Observed sample count does not match frozen stratum_sample_n.")

N_total = int(check["N_h"].sum())

if abs(a["analysis_design_weight"].sum() - N_total) > 1e-6:
    raise RuntimeError("Analysis weights do not sum to corrected target population.")

H = len(check)
H_singleton = int((check["sample_rows"] == 1).sum())
singleton_population = int(
    check.loc[check["sample_rows"] == 1, "N_h"].sum()
)

print("=" * 100)
print("STRICT DESIGN-BASED UNCERTAINTY FOR ACQUISITION-LEVEL >80 PX BURDEN")
print("=" * 100)
print(f"Target population N:                  {N_total}")
print(f"Occupied strata:                      {H}")
print(f"Sample-singleton strata (n_h = 1):    {H_singleton}")
print(
    f"Population in singleton-sample strata: {singleton_population} "
    f"({100*singleton_population/N_total:.2f}%)"
)
print()

criteria = {
    ">=1 band >80 px": a["n_tail_gt80"] >= 1,
    ">=2 bands >80 px": a["n_tail_gt80"] >= 2,
    ">=4 bands >80 px": a["n_tail_gt80"] >= 4,
    ">=6 bands >80 px": a["n_tail_gt80"] >= 6,
    "all 7 bands >80 px": a["n_tail_gt80"] >= 7,
}

ALPHA = 0.05
Z975 = 1.959963984540054

rows = []

for label, mask in criteria.items():
    d = a.copy()
    d["y"] = mask.astype(int)

    # Horvitz-Thompson / stratified expansion estimator.
    total_hat = float(
        (d["analysis_design_weight"] * d["y"]).sum()
    )
    p_hat = total_hat / N_total

    var_upper = 0.0
    hoeffding_proxy = 0.0

    for stratum, g in d.groupby("stratum"):
        N_h = int(g["analysis_stratum_population_n"].iloc[0])
        n_h = int(len(g))

        if N_h < n_h:
            raise RuntimeError(
                f"{stratum}: N_h={N_h} < n_h={n_h}"
            )

        W_h = N_h / N_total
        f_h = n_h / N_h

        # Exact maximum possible finite-population variance of a binary variable:
        #
        # S_h^2 = M_h (N_h-M_h) / [N_h (N_h-1)]
        #
        # maximized over integer M_h.
        if N_h <= 1 or n_h == N_h:
            s2_max = 0.0
        else:
            s2_max = (
                math.floor((N_h * N_h) / 4)
                / (N_h * (N_h - 1))
            )

        # Variance of stratified SRS estimator of population proportion:
        # sum_h W_h^2 (1-f_h) S_h^2 / n_h
        var_upper += (
            (W_h ** 2)
            * (1.0 - f_h)
            * s2_max
            / n_h
        )

        # Distribution-free Hoeffding proxy.
        # Sampling without replacement is no less concentrated than
        # corresponding sampling with replacement; we deliberately omit
        # finite-population improvement here.
        hoeffding_proxy += (W_h ** 2) / n_h

    se_upper = math.sqrt(var_upper)

    # Approximate Wald interval, but using a strict upper variance bound.
    wald_lo = max(0.0, p_hat - Z975 * se_upper)
    wald_hi = min(1.0, p_hat + Z975 * se_upper)

    # Finite-sample Chebyshev interval:
    # P(|error| >= t) <= Var/t^2.
    cheb_half = math.sqrt(var_upper / ALPHA)
    cheb_lo = max(0.0, p_hat - cheb_half)
    cheb_hi = min(1.0, p_hat + cheb_half)

    # Two-sided Hoeffding:
    # P(|error| >= t) <= 2 exp(-2 t^2 / A)
    hoeff_half = math.sqrt(
        0.5 * hoeffding_proxy * math.log(2.0 / ALPHA)
    )
    hoeff_lo = max(0.0, p_hat - hoeff_half)
    hoeff_hi = min(1.0, p_hat + hoeff_half)

    rows.append({
        "criterion": label,
        "sample_n_positive": int(mask.sum()),
        "HT_total_hat": total_hat,
        "point_percent": 100 * p_hat,
        "variance_upper_bound": var_upper,
        "se_upper_bound_percent": 100 * se_upper,

        "wald_uppervar_95_lo_percent": 100 * wald_lo,
        "wald_uppervar_95_hi_percent": 100 * wald_hi,

        "hoeffding_95_lo_percent": 100 * hoeff_lo,
        "hoeffding_95_hi_percent": 100 * hoeff_hi,

        "chebyshev_95_lo_percent": 100 * cheb_lo,
        "chebyshev_95_hi_percent": 100 * cheb_hi,
    })

res = pd.DataFrame(rows)

print("===== RESULTS =====")
print(
    res[[
        "criterion",
        "sample_n_positive",
        "point_percent",
        "se_upper_bound_percent",
        "wald_uppervar_95_lo_percent",
        "wald_uppervar_95_hi_percent",
        "hoeffding_95_lo_percent",
        "hoeffding_95_hi_percent",
        "chebyshev_95_lo_percent",
        "chebyshev_95_hi_percent",
    ]].to_string(index=False)
)

print()
print("Interpretation:")
print("- point_percent: design-weighted population estimate.")
print("- Wald: approximate normal interval using a worst-case binary variance bound.")
print("- Hoeffding: finite-sample distribution-free bound; no normal approximation.")
print("- Chebyshev: finite-sample variance-bound interval; usually the widest.")
print("- No bootstrap, no post-hoc stratum collapsing, no imputation.")

out = ANA / "tail_burden_design_uncertainty_v1.csv"
res.to_csv(out, index=False)

print()
print("Wrote:", out)
