from pathlib import Path
import csv
import math

import cv2
import numpy as np
import rasterio

PRODUCT_IDS = ["6048", "6047", "6042", "6038", "6028"]
LEVELS = [
    ("L1A", "data/l1a", "L1A"),
    ("L1C", "data/l1", "L1"),
]
MASTER = 2
BANDS = list(range(8))
TILE = 512
STRIDE = 512
OUT = Path("outputs/l1a_l1c_audit_deep/local_shift_field.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def find_product(root, level_pattern, pid):
    return sorted(Path(root).glob(f"PHISAT-2_{level_pattern}_{int(pid):09d}_*"))[0]

def standardize(x):
    x = x.astype(np.float32)
    m = np.isfinite(x)
    if m.sum() < 100:
        return None
    mu = float(x[m].mean())
    sig = float(x[m].std())
    if sig < 1e-6:
        return None
    return (x - mu) / sig

rows = []

for pid in PRODUCT_IDS:
    print("product", pid)

    for level_name, root, level_pattern in LEVELS:
        prod = find_product(root, level_pattern, pid)
        tif = prod / "bands" / "scene_0_BC_multiband.tiff"

        with rasterio.open(tif) as src:
            master = src.read(MASTER + 1)

            for band in BANDS:
                if band == MASTER:
                    continue

                arr = src.read(band + 1)

                H, W = master.shape
                for y0 in range(0, H - TILE + 1, STRIDE):
                    for x0 in range(0, W - TILE + 1, STRIDE):
                        a = master[y0:y0+TILE, x0:x0+TILE]
                        b = arr[y0:y0+TILE, x0:x0+TILE]

                        aa = standardize(a)
                        bb = standardize(b)
                        if aa is None or bb is None:
                            continue

                        # cv2 returns (dx, dy), response
                        (dx, dy), response = cv2.phaseCorrelate(aa, bb)

                        rows.append({
                            "product_id": pid,
                            "level": level_name,
                            "master_band": MASTER,
                            "band": band,
                            "tile_y": y0,
                            "tile_x": x0,
                            "dy": float(dy),
                            "dx": float(dx),
                            "shift_mag": float(math.sqrt(dx*dx + dy*dy)),
                            "response": float(response),
                        })

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("wrote", OUT)
