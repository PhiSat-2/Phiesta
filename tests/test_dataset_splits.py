import json
import pandas as pd
import pytest

from phiesta.datasets import PhiestaDataset, open_dataset
from phiesta.datasets.splits import _haversine_km


def _dataset(tmp_path, acquisitions, patches=None):
    root = tmp_path / "dataset"
    root.mkdir()
    acq = pd.DataFrame(acquisitions).copy()
    if "build_status" not in acq.columns:
        acq["build_status"] = "SUCCESS"
    patches = pd.DataFrame() if patches is None else pd.DataFrame(patches)
    metadata = {"format": "phiesta-dataset-v1"}
    acq.to_csv(root / "acquisitions.csv", index=False)
    patches.to_csv(root / "patches.csv", index=False)
    (root / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
    return PhiestaDataset(root, acq, patches, metadata)


def test_random_split_and_patch_propagation(tmp_path):
    acq = {"product_id": [str(i) for i in range(20)]}
    patches = [
        {"product_id": str(i), "patch_id": f"{i}_{j}"}
        for i in range(20)
        for j in range(2)
    ]
    ds = _dataset(tmp_path, acq, patches)
    summary = ds.make_splits(seed=7, verbose=False)
    counts = dict(zip(summary["split"], summary["acquisitions"]))
    assert counts == {"train": 16, "val": 2, "test": 2}

    mapping = dict(
        zip(ds.acquisitions["product_id"].astype(str), ds.acquisitions["split"])
    )
    assert (
        ds.patches["split"]
        == ds.patches["product_id"].astype(str).map(mapping)
    ).all()
    assert ds.split_manifest_path.exists()

    reopened = open_dataset(ds.root)
    assert "split" in reopened.acquisitions.columns
    assert "split" in reopened.patches.columns


def test_group_by_keeps_group_together(tmp_path):
    ds = _dataset(tmp_path, {
        "product_id": [str(i) for i in range(12)],
        "pass_id": [f"pass_{i // 2}" for i in range(12)],
    })
    ds.make_splits(
        train=0.5,
        val=0.25,
        test=0.25,
        group_by="pass_id",
        seed=3,
        verbose=False,
    )
    assert int(ds.acquisitions.groupby("pass_id")["split"].nunique().max()) == 1


def test_spatial_split_guarantees_distance(tmp_path):
    ds = _dataset(tmp_path, {
        "product_id": ["a", "b", "c", "d", "e", "f"],
        "center_lon": [0, 0.05, 2, 2.05, 4, 4.05],
        "center_lat": [0, 0, 0, 0, 0, 0],
    })
    ds.make_splits(
        train=1/3,
        val=1/3,
        test=1/3,
        method="spatial",
        min_distance_km=10,
        seed=11,
        verbose=False,
    )

    by_id = ds.acquisitions.set_index("product_id")["split"]
    assert by_id["a"] == by_id["b"]
    assert by_id["c"] == by_id["d"]
    assert by_id["e"] == by_id["f"]

    table = ds.acquisitions.reset_index(drop=True)
    for i in range(len(table)):
        for j in range(i + 1, len(table)):
            if table.loc[i, "split"] != table.loc[j, "split"]:
                distance = float(_haversine_km(
                    table.loc[i, "center_lon"],
                    table.loc[i, "center_lat"],
                    table.loc[j, "center_lon"],
                    table.loc[j, "center_lat"],
                ))
                assert distance >= 10


def test_failed_rows_unassigned(tmp_path):
    ds = _dataset(tmp_path, {
        "product_id": ["ok1", "bad", "ok2"],
        "build_status": ["SUCCESS", "FAILED", "SUCCESS"],
    })
    ds.make_splits(train=0.5, val=0, test=0.5, verbose=False)
    row = ds.acquisitions[ds.acquisitions["product_id"] == "bad"].iloc[0]
    assert pd.isna(row["split"])


def test_existing_split_requires_overwrite(tmp_path):
    ds = _dataset(tmp_path, {
        "product_id": ["a", "b"],
        "split": ["train", "test"],
    })
    with pytest.raises(ValueError, match="overwrite=True"):
        ds.make_splits(verbose=False)


def test_spatial_requires_coordinates(tmp_path):
    ds = _dataset(tmp_path, {"product_id": ["a", "b"]})
    with pytest.raises(ValueError, match="center_lon"):
        ds.make_splits(method="spatial", min_distance_km=100, verbose=False)


def test_get_split_auto_returns_patches(tmp_path):
    ds = _dataset(
        tmp_path,
        {"product_id": ["a", "b"]},
        [
            {"product_id": "a", "patch_id": "a0"},
            {"product_id": "b", "patch_id": "b0"},
        ],
    )
    ds.make_splits(train=0.5, val=0, test=0.5, seed=1, verbose=False)
    train = ds.get_split("train")
    assert "patch_id" in train.columns
    assert set(train["split"]) == {"train"}
