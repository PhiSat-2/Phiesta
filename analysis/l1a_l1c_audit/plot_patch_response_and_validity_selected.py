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
OUT = ROOT / "patch_validity_maps"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
df = df[
    (df["master_band"] == MASTER_BAND)
    & (df["band"] == TEST_BAND)
].copy()

def make_grid(d, col):
    xs = sorted(df["tile_x"].unique())
    ys = sorted(df["tile_y"].unique())
    xi = {x: i for i, x in enumerate(xs)}
    yi = {y: i for i, y in enumerate(ys)}

    arr = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    for _, r in d.iterrows():
        arr[yi[r["tile_y"]], xi[r["tile_x"]]] = r[col]
    return arr

for pid in PRODUCT_IDS:
    print("validity map", pid)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, level in enumerate(["L1A", "L1C"]):
        d = df[(df["product_id"] == int(pid)) & (df["level"] == level)].copy()

        if len(d) == 0:
            for col in range(3):
                axes[row, col].axis("off")
            continue

        response = make_grid(d, "response")
        raw_shift_all = make_grid(d, "shift_mag")

        valid = response >= RESPONSE_THR
        raw_shift_valid = raw_shift_all.copy()
        raw_shift_valid[~valid] = np.nan

        invalid = np.zeros_like(response, dtype=np.float32)
        invalid[np.isnan(response)] = 2.0      # absent from CSV / not measurable
        invalid[(~np.isnan(response)) & (~valid)] = 1.0  # measured but low response
        invalid[valid] = 0.0

        n_total = np.isfinite(response).sum()
        n_valid = valid.sum()
        valid_pct = 100 * n_valid / max(n_total, 1)

        im0 = axes[row, 0].imshow(raw_shift_all, vmin=0, vmax=80)
        axes[row, 0].set_title(f"{pid} {level} raw shift, all measured patches")
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(response, vmin=0, vmax=1)
        axes[row, 1].set_title(
            f"{pid} {level} phase-corr response\n"
            f"valid={n_valid}/{n_total} ({valid_pct:.1f}%)"
        )
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(invalid, vmin=0, vmax=2)
        axes[row, 2].set_title(
            f"{pid} {level} validity mask\n"
            "0=valid, 1=low response, 2=not measured"
        )
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = OUT / f"{pid}_validity_b{TEST_BAND}_to_b{MASTER_BAND}.png"
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)
