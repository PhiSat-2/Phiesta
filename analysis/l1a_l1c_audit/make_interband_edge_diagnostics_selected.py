from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import cv2

PRODUCT_IDS = ["6008", "6025", "6038", "6041"]
OUT = Path("outputs/l1a_l1c_audit_selected/final_edge_diagnostics")
OUT.mkdir(parents=True, exist_ok=True)

MASTER_BAND = 2
TEST_BAND = 6
CROP_SIZE = 768

def find_product(root, level_pattern, pid):
    return sorted(Path(root).glob(f"PHISAT-2_{level_pattern}_{int(pid):09d}_*"))[0]

def read_band(product, band):
    tif = product / "bands" / "scene_0_BC_multiband.tiff"
    with rasterio.open(tif) as src:
        return src.read(band + 1).astype(np.float32)

def norm(x):
    m = np.isfinite(x)
    lo, hi = np.percentile(x[m], [2, 98])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

def edge(x):
    x = norm(x)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    e = np.sqrt(gx * gx + gy * gy)
    return norm(e)

def best_crop(a, b, size=CROP_SIZE):
    e = edge(a) + edge(b)
    H, W = e.shape
    step = size // 2
    best = None
    for y in range(0, H - size + 1, step):
        for x in range(0, W - size + 1, step):
            score = float(e[y:y+size, x:x+size].mean())
            if best is None or score > best[0]:
                best = (score, y, x)
    return best[1], best[2]

def edge_overlay(master, other, y, x, size=CROP_SIZE):
    em = edge(master[y:y+size, x:x+size])
    eo = edge(other[y:y+size, x:x+size])

    rgb = np.zeros((size, size, 3), dtype=np.float32)
    rgb[..., 0] = em          # master edges red
    rgb[..., 1] = eo          # other edges green
    rgb[..., 2] = eo          # other edges blue = cyan
    return np.clip(rgb, 0, 1)

for pid in PRODUCT_IDS:
    print("edge diagnostic", pid)

    l1a = find_product("data/l1a", "L1A", pid)
    l1c = find_product("data/l1", "L1", pid)

    a_master = read_band(l1a, MASTER_BAND)
    a_test = read_band(l1a, TEST_BAND)
    c_master = read_band(l1c, MASTER_BAND)
    c_test = read_band(l1c, TEST_BAND)

    y, x = best_crop(a_master, a_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(edge_overlay(a_master, a_test, y, x))
    axes[0].set_title(f"{pid} L1A edges: band {MASTER_BAND} red, band {TEST_BAND} cyan")
    axes[0].axis("off")

    axes[1].imshow(edge_overlay(c_master, c_test, y, x))
    axes[1].set_title(f"{pid} L1C edges: band {MASTER_BAND} red, band {TEST_BAND} cyan")
    axes[1].axis("off")

    plt.tight_layout()
    out = OUT / f"{pid}_edge_b{MASTER_BAND}_vs_b{TEST_BAND}.png"
    plt.savefig(out, dpi=180)
    plt.close()

    print(out)
