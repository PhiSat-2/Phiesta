from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyrawph import connect_insula


def normalize_band(x, percentiles=(1, 99)):
    x = x.astype("float32")
    valid = np.isfinite(x)

    vals = x[valid]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.percentile(vals, percentiles)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    return np.clip((x - lo) / (hi - lo), 0, 1)


def downsample(img, max_side=900):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img

    step = int(np.ceil(m / max_side))
    return img[::step, ::step]


def safe_product_id(x):
    s = str(x)
    if s.endswith(".0"):
        s = s[:-2]
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/mangrove_candidates_cloud_scored.csv")
    parser.add_argument("--out-dir", default="outputs/mangrove_review/nir_contact_sheets")
    parser.add_argument("--cloud-status", default="low_cloud")
    parser.add_argument("--include-moderate", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--band", default="NIR")
    parser.add_argument("--sort-by", default="mangrove_pixels")
    parser.add_argument("--per-page", type=int, default=25)
    parser.add_argument("--max-side", type=int, default=900)
    parser.add_argument("--percentiles", nargs=2, type=float, default=(1, 99))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    statuses = [args.cloud_status]
    if args.include_moderate:
        statuses.append("moderate")

    df = df[df["cloud_status"].isin(statuses)].copy()

    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=False)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print(f"Rows selected: {len(df)}")
    print(df[[
        "product_id",
        "mangrove_pixels",
        "mangrove_fraction",
        "cloud_like_fraction",
        "cloud_status",
    ]].head(20))

    client = connect_insula()

    records = []

    per_page = int(args.per_page)
    n_pages = int(np.ceil(len(df) / per_page))

    for page_idx in range(n_pages):
        sub = df.iloc[page_idx * per_page : (page_idx + 1) * per_page]

        n = len(sub)
        ncols = 5
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.5 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax in axes:
            ax.axis("off")

        for ax, (_, row) in zip(axes, sub.iterrows()):
            pid = safe_product_id(row["product_id"])

            try:
                print(f"[page {page_idx+1}/{n_pages}] loading {pid}")
                ev = client.load_l1(pid)

                band = ev.get_band(args.band)
                img = normalize_band(band, percentiles=tuple(args.percentiles))
                img = downsample(img, max_side=args.max_side)

                ax.imshow(img, cmap="gray")
                ax.axis("off")

                title = (
                    f"{pid}\n"
                    f"mang={row['mangrove_fraction']:.3f} "
                    f"cloud={row['cloud_like_fraction']:.3f}\n"
                    f"{row['cloud_status']}"
                )
                ax.set_title(title, fontsize=9)

                records.append({
                    "product_id": pid,
                    "status": "SUCCESS",
                    "error": "",
                    "page": page_idx + 1,
                    "mangrove_fraction": row["mangrove_fraction"],
                    "mangrove_pixels": row["mangrove_pixels"],
                    "cloud_like_fraction": row["cloud_like_fraction"],
                    "cloud_status": row["cloud_status"],
                })

            except Exception as exc:
                ax.text(0.5, 0.5, f"{pid}\nFAILED", ha="center", va="center")
                ax.axis("off")

                records.append({
                    "product_id": pid,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "page": page_idx + 1,
                    "mangrove_fraction": row.get("mangrove_fraction", np.nan),
                    "mangrove_pixels": row.get("mangrove_pixels", np.nan),
                    "cloud_like_fraction": row.get("cloud_like_fraction", np.nan),
                    "cloud_status": row.get("cloud_status", ""),
                })

        fig.suptitle(
            f"Mangrove candidates — {args.band} quicklook — page {page_idx+1}/{n_pages}",
            fontsize=16,
        )
        plt.tight_layout()

        out_png = out_dir / f"mangrove_{args.band.lower()}_page_{page_idx+1:03d}.png"
        fig.savefig(out_png, dpi=170, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {out_png}")

        pd.DataFrame(records).to_csv(out_dir / "quicklook_generation_log.csv", index=False)

    print("\nDone")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
