from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

PRODUCT_IDS = ["6048", "6047", "6042", "6038", "6028"]
OUT = Path("outputs/l1a_l1c_audit_deep/quicklooks")
OUT.mkdir(parents=True, exist_ok=True)

def find_product(root, level, pid):
    return sorted(Path(root).glob(f"PHISAT-2_{level}_{int(pid):09d}_*"))[0]

def read_rgb(multiband):
    with rasterio.open(multiband) as src:
        # bands 3,2,1 in zero-based convention -> rasterio 4,3,2
        arr = src.read([4, 3, 2], out_shape=(3, 1024, 1024)).astype(np.float32)
    out = []
    for x in arr:
        lo, hi = np.percentile(x[np.isfinite(x)], [2, 98])
        out.append(np.clip((x - lo) / (hi - lo + 1e-6), 0, 1))
    return np.stack(out, axis=-1)

for pid in PRODUCT_IDS:
    l1a = find_product("data/l1a", "L1A", pid) / "bands" / "scene_0_BC_multiband.tiff"
    l1c = find_product("data/l1", "L1", pid) / "bands" / "scene_0_BC_multiband.tiff"

    a = read_rgb(l1a)
    c = read_rgb(l1c)
    diff = np.mean(np.abs(a - c), axis=-1)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(a)
    plt.title(f"{pid} L1A BC RGB")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(c)
    plt.title(f"{pid} L1C BC RGB")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap="gray")
    plt.title("|L1A-L1C| normalized")
    plt.axis("off")

    out = OUT / f"{pid}_l1a_vs_l1c_rgb.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(out)
