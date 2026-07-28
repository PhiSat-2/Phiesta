from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

PRODUCT_IDS = ["6008", "6025", "6038", "6041"]
OUT = Path("outputs/l1a_l1c_audit_selected/radiometry")
OUT.mkdir(parents=True, exist_ok=True)

def find_product(root, level_pattern, pid):
    return sorted(Path(root).glob(f"PHISAT-2_{level_pattern}_{int(pid):09d}_*"))[0]

def read_band(product, band):
    tif = product / "bands" / "scene_0_BC_multiband.tiff"
    with rasterio.open(tif) as src:
        return src.read(band + 1).astype(np.float32)

def sample_valid(a, c, max_points=300000):
    m = np.isfinite(a) & np.isfinite(c)
    aa = a[m]
    cc = c[m]
    if len(aa) > max_points:
        idx = np.linspace(0, len(aa) - 1, max_points).astype(int)
        aa = aa[idx]
        cc = cc[idx]
    return aa, cc

def robust_norm(x):
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

rows = []

for pid in PRODUCT_IDS:
    print("product", pid)

    l1a = find_product("data/l1a", "L1A", pid)
    l1c = find_product("data/l1", "L1", pid)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()

    for band in range(8):
        print("  band", band)

        a = read_band(l1a, band)
        c = read_band(l1c, band)

        aa, cc = sample_valid(a, c)

        corr = float(np.corrcoef(aa, cc)[0, 1]) if len(aa) > 10 else np.nan

        # linear fit L1C ≈ slope * L1A + intercept
        slope, intercept = np.polyfit(aa, cc, 1)

        an = robust_norm(aa)
        cn = robust_norm(cc)
        norm_mae = float(np.mean(np.abs(an - cn)))

        row = {
            "product_id": pid,
            "band": band,
            "n": len(aa),
            "l1a_min": float(np.min(aa)),
            "l1a_p1": float(np.percentile(aa, 1)),
            "l1a_p5": float(np.percentile(aa, 5)),
            "l1a_median": float(np.median(aa)),
            "l1a_p95": float(np.percentile(aa, 95)),
            "l1a_p99": float(np.percentile(aa, 99)),
            "l1a_max": float(np.max(aa)),
            "l1c_min": float(np.min(cc)),
            "l1c_p1": float(np.percentile(cc, 1)),
            "l1c_p5": float(np.percentile(cc, 5)),
            "l1c_median": float(np.median(cc)),
            "l1c_p95": float(np.percentile(cc, 95)),
            "l1c_p99": float(np.percentile(cc, 99)),
            "l1c_max": float(np.max(cc)),
            "corr_l1a_l1c": corr,
            "linear_slope_l1c_vs_l1a": float(slope),
            "linear_intercept_l1c_vs_l1a": float(intercept),
            "robust_norm_mae": norm_mae,
        }
        rows.append(row)

        ax = axes[band]
        ax.hist(an, bins=80, alpha=0.5, density=True, label="L1A")
        ax.hist(cn, bins=80, alpha=0.5, density=True, label="L1C")
        ax.set_title(f"band {band} corr={corr:.2f} mae={norm_mae:.3f}")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = OUT / f"{pid}_radiometry_histograms.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print("wrote", fig_path)

df = pd.DataFrame(rows)
csv_path = OUT / "radiometry_per_product_band.csv"
df.to_csv(csv_path, index=False)

summary = (
    df.groupby("band")
    .agg(
        corr_median=("corr_l1a_l1c", "median"),
        corr_min=("corr_l1a_l1c", "min"),
        norm_mae_median=("robust_norm_mae", "median"),
        slope_median=("linear_slope_l1c_vs_l1a", "median"),
        intercept_median=("linear_intercept_l1c_vs_l1a", "median"),
    )
    .reset_index()
)
summary_path = OUT / "radiometry_band_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n===== RADIOMETRY BAND SUMMARY =====")
print(summary.to_string(index=False))

print("\nwrote", csv_path)
print("wrote", summary_path)
