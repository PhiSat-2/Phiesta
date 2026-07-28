from pathlib import Path
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt

PRODUCT_IDS = ["6008", "6025", "6038", "6041"]
OUT = Path("outputs/l1a_l1c_audit_selected/overleaf_report/figures")
OUT.mkdir(parents=True, exist_ok=True)

rows = []

def find_l1a(pid):
    return sorted(Path("data/l1a").glob(f"PHISAT-2_L1A_{int(pid):09d}_*"))[0]

for pid in PRODUCT_IDS:
    p = find_l1a(pid)
    rc = p / "bands" / "scene_0_RC_multiband.tiff"
    bc = p / "bands" / "scene_0_BC_multiband.tiff"

    with rasterio.open(rc) as rsrc, rasterio.open(bc) as bsrc:
        for band in range(1, rsrc.count + 1):
            a = rsrc.read(band).astype(np.float32)
            b = bsrc.read(band).astype(np.float32)
            d = np.abs(a - b)
            corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
            rows.append({
                "product": pid,
                "band": band - 1,
                "max_abs_diff": float(np.nanmax(d)),
                "mean_abs_diff": float(np.nanmean(d)),
                "corr": float(corr),
            })

df = pd.DataFrame(rows)
df.to_csv(OUT / "l1a_rc_vs_bc_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 5))
for pid in PRODUCT_IDS:
    sub = df[df["product"] == pid]
    ax.plot(sub["band"], sub["max_abs_diff"], marker="o", label=pid)
ax.set_title("L1A RC vs BC: max absolute pixel difference per band")
ax.set_xlabel("band")
ax.set_ylabel("max |RC - BC|")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "l1a_rc_vs_bc_summary.png", dpi=170)
plt.close()

print(df.to_string(index=False))
print("wrote", OUT / "l1a_rc_vs_bc_summary.png")
