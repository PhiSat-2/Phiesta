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

def sample(x, n=250000):
    x = x[np.isfinite(x)]
    if len(x) > n:
        x = x[rng.choice(len(x), size=n, replace=False)]
    return x

for pid in PRODUCT_IDS:
    l1a = find_product("data/l1a", "L1A", pid)
    l1c = find_product("data/l1", "L1", pid)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    axes = axes.ravel()

    for band in range(8):
        a = sample(read_band(l1a, band))
        c = sample(read_band(l1c, band))

        lo = min(np.percentile(a, 0.5), np.percentile(c, 0.5))
        hi = max(np.percentile(a, 99.5), np.percentile(c, 99.5))

        ax = axes[band]
        ax.hist(a, bins=120, range=(lo, hi), density=True, histtype="step", linewidth=1.2, label="L1A")
        ax.hist(c, bins=120, range=(lo, hi), density=True, histtype="step", linewidth=1.2, label="L1C")
        ax.set_title(f"{pid} band {band}")
        ax.set_xlabel("raw pixel value")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = OUT / f"{pid}_radiometry_raw_histograms.png"
    plt.savefig(out, dpi=170)
    plt.close()
    print("wrote", out)
