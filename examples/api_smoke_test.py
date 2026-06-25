from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from phiesta import L1_event, connect_insula


def test_event_api(event):
    print("\n=== event.show_event_info() ===")
    event.show_event_info()

    print("\n=== cube / patch helpers ===")
    cube = event.to_cube()
    print("cube:", cube.shape, cube.dtype)

    rgb = event.to_cube(bands=("RED", "GREEN", "BLUE"), band_axis=-1)
    print("rgb cube:", rgb.shape, rgb.dtype)

    patch = event.get_patch(
        x_min=1000,
        y_min=1000,
        width=256,
        height=256,
        bands=("RED", "GREEN", "BLUE"),
        band_axis=-1,
    )
    print("patch:", patch.shape, patch.dtype)

    patch_norm = event.normalize(
        patch,
        percentiles=(1, 99),
        per_band=True,
        band_axis=-1,
    )
    print("patch normalized:", patch_norm.shape, float(np.nanmin(patch_norm)), float(np.nanmax(patch_norm)))

    patch_index = event.build_patch_index(patch_size=1024, stride=1024)
    print("patch index:", patch_index.shape)

    first_patch = next(event.iter_patches(
        index=patch_index.head(1),
        bands=("NIR",),
        squeeze=True,
    ))
    print("first indexed patch:", first_patch["patch_id"], first_patch["patch"].shape)

    print("\n=== band stats ===")
    stats = event.band_stats(
        bands=("BLUE", "GREEN", "RED", "NIR"),
        percentiles=(1, 50, 99),
        sample=100_000,
    )
    print(stats)

    print("\nOK: event API smoke test passed.")


def test_insula_api(product_id: str):
    print("\n=== Insula search table ===")
    client = connect_insula()

    df = client.search_l1_table(page=0, results_per_page=5)
    print(df[["product_id", "filename", "start_datetime", "center_lon", "center_lat"]].head().to_string(index=False))

    print("\n=== Insula load L1 ===")
    event = client.load_l1(product_id)
    test_event_api(event)

    print("\nOK: Insula API smoke test passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-id", default="5359")
    parser.add_argument("--local-l1", default=None)
    parser.add_argument("--insula", action="store_true")
    args = parser.parse_args()

    if args.insula:
        test_insula_api(args.product_id)
        return

    if args.local_l1 is None:
        args.local_l1 = (
            "data/l1/"
            "PHISAT-2_L1_000005359_20260512183354_20260512183357_236B0FFC"
        )

    path = Path(args.local_l1)
    if not path.exists():
        raise FileNotFoundError(
            f"Local L1 product not found: {path}\n"
            "Pass --local-l1 PATH or run with --insula."
        )

    print(f"Loading local L1: {path}")
    event = L1_event.from_path(path)
    test_event_api(event)


if __name__ == "__main__":
    main()
