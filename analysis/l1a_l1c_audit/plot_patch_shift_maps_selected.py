from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRODUCT_IDS = ["6008", "6025", "6038", "6041", "5978", "6045"]
MASTER_BAND = 2
TEST_BAND = 6
RESPONSE_THR = 0.2

ROOT = Path("outputs/l1a_l1c_audit_selected")
CSV = ROOT / "local_shift_field.csv"
OUT = ROOT / "patch_shift_maps"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
df = df[
    (df["master_band"] == MASTER_BAND)
    & (df["band"] == TEST_BAND)
    & (df["response"] >= RESPONSE_THR)
].copy()

def make_grid(d, col):
    xs = sorted(d["tile_x"].unique())
    ys = sorted(d["tile_y"].unique())
    xi = {x: i for i, x in enumerate(xs)}
    yi = {y: i for i, y in enumerate(ys)}

    arr = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    for _, r in d.iterrows():
        arr[yi[r["tile_y"]], xi[r["tile_x"]]] = r[col]
    return arr

summary_rows = []

for pid in PRODUCT_IDS:
    print("plot", pid)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, level in enumerate(["L1A", "L1C"]):
        d = df[(df["product_id"] == int(pid)) & (df["level"] == level)].copy()

        if len(d) == 0:
            for col in range(3):
                axes[row, col].axis("off")
            continue

        med_dx = float(d["dx"].median())
        med_dy = float(d["dy"].median())

        d["residual_dx"] = d["dx"] - med_dx
        d["residual_dy"] = d["dy"] - med_dy
        d["residual_mag"] = np.sqrt(d["residual_dx"] ** 2 + d["residual_dy"] ** 2)

        summary_rows.append({
            "product_id": pid,
            "level": level,
            "band": TEST_BAND,
            "n_patches": len(d),
            "median_dx": med_dx,
            "median_dy": med_dy,
            "raw_median": float(d["shift_mag"].median()),
            "raw_mean": float(d["shift_mag"].mean()),
            "raw_max": float(d["shift_mag"].max()),
            "residual_median": float(d["residual_mag"].median()),
            "residual_mean": float(d["residual_mag"].mean()),
            "residual_max": float(d["residual_mag"].max()),
            "response_median": float(d["response"].median()),
        })

        raw = make_grid(d, "shift_mag")
        residual = make_grid(d, "residual_mag")
        response = make_grid(d, "response")

        im0 = axes[row, 0].imshow(raw, vmin=0, vmax=80)
        axes[row, 0].set_title(
            f"{pid} {level} raw shift | b{TEST_BAND}->b{MASTER_BAND}\n"
            f"median={d['shift_mag'].median():.2f}px, max={d['shift_mag'].max():.2f}px"
        )
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(residual, vmin=0, vmax=8)
        axes[row, 1].set_title(
            f"{pid} {level} residual after median shift\n"
            f"median={d['residual_mag'].median():.2f}px, max={d['residual_mag'].max():.2f}px"
        )
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(response, vmin=0, vmax=1)
        axes[row, 2].set_title(
            f"{pid} {level} phase-correlation response\n"
            f"median={d['response'].median():.2f}"
        )
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = OUT / f"{pid}_patch_shift_b{TEST_BAND}_to_b{MASTER_BAND}.png"
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)

summary = pd.DataFrame(summary_rows)
summary_path = OUT / f"patch_shift_summary_b{TEST_BAND}_to_b{MASTER_BAND}.csv"
summary.to_csv(summary_path, index=False)

print("\n===== SUMMARY =====")
print(summary.to_string(index=False))
print("wrote", summary_path)
