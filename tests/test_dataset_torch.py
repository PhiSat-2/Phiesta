from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from phiesta.datasets import PhiestaDataset


def _make_dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "patches_files").mkdir()
    (root / "targets_files").mkdir()

    rows = []
    for i, split in enumerate(["train", "train", "test"]):
        image_path = root / "patches_files" / f"p{i}.npy"
        mask_path = root / "targets_files" / f"p{i}.npy"

        np.save(image_path, np.full((3, 4, 5), i + 1, dtype=np.uint16))
        np.save(mask_path, np.full((4, 5), i, dtype=np.uint8))

        rows.append({
            "product_id": str(i),
            "dataset_patch_id": f"{i}_p0",
            "patch_path": str(image_path),
            "split": split,
            "target_mask_path": str(mask_path),
            "target_mask_status": "SUCCESS",
            "target_score_value": float(i) + 0.5,
            "target_score_status": "SUCCESS",
            "target_class_value": "mangrove" if i < 2 else "water",
            "target_class_status": "SUCCESS",
        })

    return PhiestaDataset(
        root=root,
        acquisitions=pd.DataFrame({
            "product_id": ["0", "1", "2"],
            "build_status": ["SUCCESS"] * 3,
        }),
        patches=pd.DataFrame(rows),
        metadata={"format": "phiesta-dataset-v1"},
    )


def test_to_torch_split_and_array_target(tmp_path):
    ds = _make_dataset(tmp_path)
    train = ds.to_torch(split="train", targets="mask")
    assert len(train) == 2
    image, target = train[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert image.shape == (3, 4, 5)
    assert target.shape == (4, 5)
    assert image.dtype == torch.float32
    assert target.dtype == torch.uint8


def test_to_torch_multiple_targets(tmp_path):
    ds = _make_dataset(tmp_path)
    image, target = ds.to_torch(
        split="train",
        targets=["mask", "score", "class"],
    )[1]
    assert image.shape == (3, 4, 5)
    assert set(target) == {"mask", "score", "class"}
    assert target["mask"].shape == (4, 5)
    assert torch.is_tensor(target["score"])
    assert target["class"] == "mangrove"


def test_to_torch_metadata(tmp_path):
    ds = _make_dataset(tmp_path)
    image, score, meta = ds.to_torch(
        split="test",
        targets="score",
        return_metadata=True,
    )[0]
    assert image.shape == (3, 4, 5)
    assert torch.is_tensor(score)
    assert meta["dataset_patch_id"] == "2_p0"


def test_to_dataloader_batches(tmp_path):
    ds = _make_dataset(tmp_path)
    loader = ds.to_dataloader(
        split="train",
        targets="mask",
        batch_size=2,
        shuffle=False,
    )
    images, masks = next(iter(loader))
    assert images.shape == (2, 3, 4, 5)
    assert masks.shape == (2, 4, 5)


def test_to_torch_no_targets(tmp_path):
    ds = _make_dataset(tmp_path)
    image = ds.to_torch(split="test")[0]
    assert torch.is_tensor(image)
    assert image.shape == (3, 4, 5)


def test_to_torch_requires_patches(tmp_path):
    ds = PhiestaDataset(
        root=tmp_path,
        acquisitions=pd.DataFrame({"product_id": ["1"]}),
        patches=pd.DataFrame(),
        metadata={},
    )
    with pytest.raises(ValueError, match="patch dataset"):
        ds.to_torch()


def test_to_torch_rejects_missing_target(tmp_path):
    ds = _make_dataset(tmp_path)
    with pytest.raises(ValueError, match="not present"):
        ds.to_torch(targets="missing")
