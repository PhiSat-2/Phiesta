from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

PRODUCT_IDS = ["6008", "6025", "6038", "6041"]
OUT = Path("outputs/l1a_l1c_audit_selected/overleaf_report/figures")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)

def find_product(root, pattern, pid):
    return sorted(Path(root).glob(f"PHISAT-2_{pattern}_{int(pid):09d}_*"))[0]

def read_band(product, band):
    tif = product / "bands" / "scene_0_BC_multiband.tiff"
    with rasterio.open(tif) as src:
        return src.read(band + 1).astype(np.float32)

def robust_norm_sample(x, max_points=250000):
    x = x[np.isfinite(x)]
    if len(x) > max_points:
        idx = rng.choice(len(x), size=max_points, replace=False)
        x = x[idx]

    lo, hi = np.percentile(x, [2, 98])
    y = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    return y

qs = np.linspace(0, 100, 301)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()

for band in range(8):
    ax = axes[band]

    l1a_curves = []
    l1c_curves = []

    for pid in PRODUCT_IDS:
        l1a = find_product("data/l1a", "L1A", pid)
        l1c = find_product("data/l1", "L1", pid)

        a = robust_norm_sample(read_band(l1a, band))
        c = robust_norm_sample(read_band(l1c, band))

        l1a_curves.append(np.percentile(a, qs))
        l1c_curves.append(np.percentile(c, qs))

    l1a_med = np.median(np.stack(l1a_curves), axis=0)
    l1c_med = np.median(np.stack(l1c_curves), axis=0)

    ax.plot(qs, l1a_med, label="L1A")
    ax.plot(qs, l1c_med, label="L1C")
    ax.set_title(f"band {band}")
    ax.set_xlabel("percentile")
    ax.set_ylabel("robust-normalized value")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

plt.tight_layout()
out = OUT / "radiometry_quantile_curves_clean.png"
plt.savefig(out, dpi=170)
plt.close()
print("wrote", out)
