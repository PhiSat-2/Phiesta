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

def sample_pair(a, c, n=100000):
    m = np.isfinite(a) & np.isfinite(c)
    aa = a[m]
    cc = c[m]
    if len(aa) > n:
        idx = rng.choice(len(aa), size=n, replace=False)
        aa = aa[idx]
        cc = cc[idx]
    return aa, cc

def robust01(x, lo=None, hi=None):
    if lo is None:
        lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

for pid in PRODUCT_IDS:
    print("scatter", pid)

    l1a = find_product("data/l1a", "L1A", pid)
    l1c = find_product("data/l1", "L1", pid)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for band in range(8):
        a = read_band(l1a, band)
        c = read_band(l1c, band)
        aa, cc = sample_pair(a, c)

        # same robust scale per axis pair, not separate per product display
        alo, ahi = np.percentile(aa, [1, 99])
        clo, chi = np.percentile(cc, [1, 99])
        an = robust01(aa, alo, ahi)
        cn = robust01(cc, clo, chi)

        corr = np.corrcoef(aa, cc)[0, 1]

        ax = axes[band]
        ax.hexbin(an, cn, gridsize=55, mincnt=1)
        ax.plot([0, 1], [0, 1], linewidth=1)
        ax.set_title(f"band {band}, corr={corr:.2f}")
        ax.set_xlabel("L1A robust-scaled")
        ax.set_ylabel("L1C robust-scaled")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    out = OUT / f"{pid}_radiometry_scatter.png"
    plt.savefig(out, dpi=170)
    plt.close()
    print("wrote", out)
