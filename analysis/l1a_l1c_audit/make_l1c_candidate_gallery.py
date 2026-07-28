from pathlib import Path
import math
import numpy as np
import rasterio
import matplotlib.pyplot as plt

ids = Path("outputs/l1a_l1c_audit_deep/common_ids.txt").read_text().splitlines()
PRODUCT_IDS = ids[-80:]

OUT = Path("outputs/l1a_l1c_audit_deep/l1c_candidate_gallery")
OUT.mkdir(parents=True, exist_ok=True)

def find_l1c(pid):
    c = sorted(Path("data/l1").glob(f"PHISAT-2_L1_{int(pid):09d}_*"))
    return c[0] if c else None

def read_rgb(product):
    tif = product / "bands" / "scene_0_BC_multiband.tiff"
    with rasterio.open(tif) as src:
        arr = src.read([4, 3, 2], out_shape=(3, 512, 512)).astype(np.float32)

    rgb = []
    for x in arr:
        m = np.isfinite(x)
        lo, hi = np.percentile(x[m], [2, 98])
        rgb.append(np.clip((x - lo) / (hi - lo + 1e-6), 0, 1))
    return np.stack(rgb, axis=-1)

imgs = []
labels = []

for pid in PRODUCT_IDS:
    p = find_l1c(pid)
    if p is None:
        continue
    try:
        imgs.append(read_rgb(p))
        labels.append(pid)
    except Exception as e:
        print("skip", pid, type(e).__name__, e)

cols = 5
rows = math.ceil(len(imgs) / cols)

plt.figure(figsize=(cols * 4, rows * 4))
for i, (img, pid) in enumerate(zip(imgs, labels), 1):
    ax = plt.subplot(rows, cols, i)
    ax.imshow(img)
    ax.set_title(pid)
    ax.axis("off")

out = OUT / "gallery_last80_l1c_rgb.png"
plt.tight_layout()
plt.savefig(out, dpi=140)
plt.close()

print("wrote", out)
