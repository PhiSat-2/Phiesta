from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from phiesta import L1_event

CSV_PATH = Path("outputs/mangrove_candidates_filename_datetime_corners.csv")
L1_ROOT = Path("/shared/projects/phisat2/data/raw/phisat2/L1")
OUT_DIR = Path("outputs/mangrove_rgb_first30")
N = 30

OUT_DIR.mkdir(parents=True, exist_ok=True)

def locate_product(full_filename: str) -> Path:
    p = L1_ROOT / full_filename
    if p.exists():
        return p
    matches = list(L1_ROOT.glob(f"{full_filename}*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find local L1 product for {full_filename}")

df = pd.read_csv(CSV_PATH).head(N)

rows = []

for i, row in df.iterrows():
    full_filename = row["full_filename"]
    print(f"\n[{i+1}/{len(df)}] RGB for {full_filename}")

    try:
        product_path = locate_product(full_filename)
        event = L1_event.from_path(str(product_path))

        rgb = event.rgb(bands=("RED", "GREEN", "BLUE"), stretch=(1, 99))
        rgb = np.asarray(rgb, dtype=np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)

        out_png = OUT_DIR / f"{full_filename}_rgb.png"
        plt.imsave(out_png, rgb)

        rows.append({
            "full_filename": full_filename,
            "product_path": str(product_path),
            "rgb_path": str(out_png),
            "status": "OK",
        })

    except Exception as e:
        print(f"[FAILED] {full_filename}: {type(e).__name__}: {e}")
        rows.append({
            "full_filename": full_filename,
            "product_path": "",
            "rgb_path": "",
            "status": f"FAILED: {type(e).__name__}: {e}",
        })

pd.DataFrame(rows).to_csv(OUT_DIR / "rgb_summary.csv", index=False)
print(f"\nSaved RGB summary: {OUT_DIR / 'rgb_summary.csv'}")
