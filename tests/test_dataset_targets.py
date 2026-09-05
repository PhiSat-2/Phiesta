from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from phiesta.datasets import PhiestaDataset, column_target, raster_target


def _dataset(tmp_path, acquisitions, patches=None):
    root = tmp_path / "dataset"
    root.mkdir(parents=True)
    acquisitions = pd.DataFrame(acquisitions)
    patches = pd.DataFrame() if patches is None else pd.DataFrame(patches)
    return PhiestaDataset(
        root=root,
        acquisitions=acquisitions,
        patches=patches,
        metadata={"format": "phiesta-dataset-v1"},
    )


def test_column_target_on_acquisitions(tmp_path):
    ds = _dataset(
        tmp_path,
        {
            "product_id": ["1", "2"],
            "label": ["mangrove", "water"],
        },
    )
    ds.add_target(
        "class",
        column_target("label"),
        level="acquisitions",
        verbose=False,
    )
    assert ds.acquisitions["target_class_value"].tolist() == [
        "mangrove",
        "water",
    ]
    assert set(ds.acquisitions["target_class_status"]) == {"SUCCESS"}


def test_array_target_is_saved_for_patches(tmp_path):
    ds = _dataset(
        tmp_path,
        {"product_id": ["1"]},
        [
            {"product_id": "1", "dataset_patch_id": "1_p0"},
            {"product_id": "1", "dataset_patch_id": "1_p1"},
        ],
    )

    def provider(row, *, context):
        value = 1 if context.item_id.endswith("p0") else 2
        return np.full((4, 5), value, dtype=np.uint8)

    ds.add_target("mask", provider, verbose=False)

    paths = [Path(p) for p in ds.patches["target_mask_path"]]
    assert all(path.exists() for path in paths)
    assert np.load(paths[0]).shape == (4, 5)
    assert ds.patches["target_mask_dtype"].tolist() == ["uint8", "uint8"]


def test_target_resume_skips_success(tmp_path):
    ds = _dataset(
        tmp_path,
        {"product_id": ["1"]},
        [{"product_id": "1", "dataset_patch_id": "1_p0"}],
    )
    calls = {"n": 0}

    def provider(row, *, context):
        calls["n"] += 1
        return 7

    ds.add_target("score", provider, verbose=False)
    ds.add_target("score", provider, verbose=False)
    assert calls["n"] == 1
    assert ds.patches.iloc[0]["target_score_value"] == 7


def test_raster_target_aligns_to_patch_grid(tmp_path):
    image_path = tmp_path / "image.tif"
    target_path = tmp_path / "target.tif"
    transform = from_origin(0, 4, 1, 1)

    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((4, 4), dtype=np.uint8), 1)

    target = np.arange(16, dtype=np.uint8).reshape(4, 4)
    with rasterio.open(
        target_path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(target, 1)

    ds = _dataset(
        tmp_path / "nested",
        {"product_id": ["1"]},
        [
            {
                "product_id": "1",
                "dataset_patch_id": "1_p0",
                "raster_path": str(image_path),
                "x_min": 1,
                "y_min": 1,
                "x_max": 3,
                "y_max": 3,
            }
        ],
    )

    ds.add_target(
        "landcover",
        raster_target(target_path),
        verbose=False,
    )

    out = np.load(ds.patches.iloc[0]["target_landcover_path"])
    np.testing.assert_array_equal(out, target[1:3, 1:3])
