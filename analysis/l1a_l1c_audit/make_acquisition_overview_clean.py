from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

PRODUCT_IDS = ["6008", "6025", "6038", "6041"]
OUT = Path("outputs/l1a_l1c_audit_selected/overleaf_report/figures")
OUT.mkdir(parents=True, exist_ok=True)

def find_product(root, pattern, pid):
    return sorted(Path(root).glob(f"PHISAT-2_{pattern}_{int(pid):09d}_*"))[0]

def read_rgb(product, size=768):
    tif = product / "bands" / "scene_0_BC_multiband.tiff"
    with rasterio.open(tif) as src:
        arr = src.read([4, 3, 2], out_shape=(3, size, size)).astype(np.float32)

    rgb = []
    for x in arr:
        m = np.isfinite(x)
        lo, hi = np.percentile(x[m], [2, 98])
        rgb.append(np.clip((x - lo) / (hi - lo + 1e-6), 0, 1))
    return np.stack(rgb, axis=-1)

fig, axes = plt.subplots(len(PRODUCT_IDS), 2, figsize=(10, 18))

for i, pid in enumerate(PRODUCT_IDS):
    l1a = find_product("data/l1a", "L1A", pid)
    l1c = find_product("data/l1", "L1", pid)

    axes[i, 0].imshow(read_rgb(l1a))
    axes[i, 0].set_title(f"{pid} L1A display RGB from BC_multiband")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(read_rgb(l1c))
    axes[i, 1].set_title(f"{pid} L1C display RGB from BC_multiband")
    axes[i, 1].axis("off")

plt.tight_layout()
out = OUT / "acquisition_overview_clean.png"
plt.savefig(out, dpi=170)
plt.close()
print("wrote", out)
