from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pyrawph import connect_insula


def _sample_band(x: np.ndarray, stride: int = 4) -> np.ndarray:
    return x[::stride, ::stride].astype(np.float32)


def estimate_cloud_like_fraction(event, stride: int = 4) -> dict:
    """
    Conservative quick cloud-like score from raw PhiSat-2 L1 visible bands.

    This is not a physical cloud mask. It is meant to remove obviously cloudy
    acquisitions from a visual review shortlist.

    Logic:
    - use visible bands BLUE, GREEN, RED;
    - normalize each band robustly with percentiles;
    - cloud-like pixels are bright and spectrally white-ish.
    """
    blue = _sample_band(event.get_band("BLUE"), stride=stride)
    green = _sample_band(event.get_band("GREEN"), stride=stride)
    red = _sample_band(event.get_band("RED"), stride=stride)

    arr = np.dstack([blue, green, red])
    finite = np.isfinite(arr).all(axis=-1)

    if finite.sum() == 0:
        return {
            "cloud_like_fraction": np.nan,
            "bright_fraction": np.nan,
            "white_bright_fraction": np.nan,
            "status": "NO_VALID_PIXELS",
        }

    # Robust per-band display-like normalization.
    norm = np.zeros_like(arr, dtype=np.float32)
    for c in range(3):
        vals = arr[..., c][finite]
        lo, hi = np.percentile(vals, [1, 99])
        if hi <= lo:
            norm[..., c] = 0
        else:
            norm[..., c] = np.clip((arr[..., c] - lo) / (hi - lo), 0, 1)

    brightness = norm.mean(axis=-1)
    whiteness = norm.max(axis=-1) - norm.min(axis=-1)

    # Main cloud-like mask.
    # Bright + low chromatic spread = white-ish.
    cloud_like = finite & (brightness > 0.82) & (whiteness < 0.28)

    # Stricter core cloud mask, useful for diagnostics.
    cloud_core = finite & (brightness > 0.90) & (whiteness < 0.22)

    # Bright regardless of whiteness, to detect scenes dominated by bright material.
    bright = finite & (brightness > 0.82)

    n = finite.sum()

    return {
        "cloud_like_fraction": float(cloud_like.sum() / n),
        "cloud_core_fraction": float(cloud_core.sum() / n),
        "bright_fraction": float(bright.sum() / n),
        "valid_sample_pixels": int(n),
        "status": "OK",
    }


def classify_cloud_status(cloud_like_fraction: float) -> str:
    if not np.isfinite(cloud_like_fraction):
        return "unknown"
    if cloud_like_fraction >= 0.35:
        return "very_cloudy"
    if cloud_like_fraction >= 0.20:
        return "cloudy"
    if cloud_like_fraction >= 0.08:
        return "moderate"
    return "low_cloud"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", default="outputs/mangrove_candidates.csv")
    parser.add_argument("--out-csv", default="outputs/mangrove_candidates_cloud_scored.csv")
    parser.add_argument("--filtered-csv", default="outputs/mangrove_candidates_cloud_filtered.csv")
    parser.add_argument("--max-cloud-like", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--sort-by", default="mangrove_pixels")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=False).reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    client = connect_insula()

    rows = []

    for idx, row in df.iterrows():
        pid = str(int(row["product_id"])) if str(row["product_id"]).replace(".0", "").isdigit() else str(row["product_id"])

        print(f"[{idx+1}/{len(df)}] {pid}")

        out = row.to_dict()
        out["cloud_eval_status"] = "NOT_RUN"
        out["cloud_like_fraction"] = np.nan
        out["cloud_core_fraction"] = np.nan
        out["bright_fraction"] = np.nan
        out["cloud_status"] = "unknown"
        out["cloud_error"] = ""

        try:
            ev = client.load_l1(pid)
            score = estimate_cloud_like_fraction(ev, stride=args.stride)

            out.update(score)
            out["cloud_status"] = classify_cloud_status(out["cloud_like_fraction"])
            out["cloud_eval_status"] = "SUCCESS"

            print(
                f"    cloud_like={out['cloud_like_fraction']:.3f} "
                f"core={out['cloud_core_fraction']:.3f} "
                f"bright={out['bright_fraction']:.3f} "
                f"status={out['cloud_status']}"
            )

        except Exception as exc:
            out["cloud_eval_status"] = "FAILED"
            out["cloud_error"] = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {out['cloud_error']}")

        rows.append(out)

        # Progressive save so we do not lose work.
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    scored = pd.DataFrame(rows)

    keep = scored[
        (scored["cloud_eval_status"] == "SUCCESS")
        & (scored["cloud_like_fraction"] <= args.max_cloud_like)
    ].copy()

    keep = keep.sort_values(
        ["mangrove_pixels", "mangrove_fraction"],
        ascending=False,
    )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out_csv, index=False)
    keep.to_csv(args.filtered_csv, index=False)

    print("\nDone")
    print(f"Input rows:    {len(df)}")
    print(f"Scored rows:   {len(scored)}")
    print(f"Kept rows:     {len(keep)}")
    print(f"Scored CSV:    {args.out_csv}")
    print(f"Filtered CSV:  {args.filtered_csv}")


if __name__ == "__main__":
    main()
