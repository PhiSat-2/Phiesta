from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for Phiesta training adapters. "
            'Install it with `pip install -e ".[ml]"` or install torch directly.'
        ) from exc
    return torch


def _safe_name(value: str) -> str:
    chars = []
    for ch in str(value):
        chars.append(ch if (ch.isalnum() or ch in "-_.") else "_")
    return "".join(chars).strip("._") or "target"


def _target_names(targets):
    if targets is None:
        return []
    if isinstance(targets, str):
        return [targets]
    return [str(x) for x in targets]


def _to_tensor(value, dtype=None):
    torch = _require_torch()

    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if not arr.flags.writeable:
            arr = arr.copy()
        tensor = torch.from_numpy(arr)
    elif isinstance(value, (bool, int, float, np.number)):
        tensor = torch.as_tensor(value)
    else:
        return value

    if dtype is not None:
        if isinstance(dtype, str):
            dtype_obj = getattr(torch, dtype, None)
            if dtype_obj is None:
                raise ValueError(f"Unknown torch dtype {dtype!r}.")
        else:
            dtype_obj = dtype
        tensor = tensor.to(dtype=dtype_obj)

    return tensor


class PhiestaTorchDataset:
    """PyTorch-compatible lazy adapter over Phiesta patch manifests."""

    def __init__(
        self,
        dataset,
        *,
        split=None,
        targets=None,
        image_dtype="float32",
        target_dtype=None,
        image_transform: Callable | None = None,
        target_transform=None,
        return_metadata=False,
        mmap_mode=None,
        require_target_success=True,
    ):
        _require_torch()

        if dataset.patches.empty:
            raise ValueError(
                "to_torch() currently requires a patch dataset. "
                "Build with patch_size=... first."
            )

        table = dataset.patches.copy()

        if split is not None:
            if "split" not in table.columns:
                raise ValueError(
                    "Requested split but patch manifest has no 'split' column. "
                    "Run dataset.make_splits() first."
                )
            table = table[table["split"].astype(str) == str(split)].copy()

        if "patch_path" not in table.columns:
            raise ValueError("Patch manifest has no 'patch_path' column.")

        self.dataset = dataset
        self.table = table.reset_index(drop=True)
        self.split = split
        self.targets = _target_names(targets)
        self.image_dtype = image_dtype
        self.target_dtype = target_dtype
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.return_metadata = bool(return_metadata)
        self.mmap_mode = mmap_mode
        self.require_target_success = bool(require_target_success)
        self._target_specs = {}

        for name in self.targets:
            safe = _safe_name(name)
            path_col = f"target_{safe}_path"
            value_col = f"target_{safe}_value"
            status_col = f"target_{safe}_status"

            if path_col in self.table.columns:
                spec = ("path", path_col)
            elif value_col in self.table.columns:
                spec = ("value", value_col)
            else:
                raise ValueError(
                    f"Target {name!r} is not present in the patch manifest."
                )

            if self.require_target_success and status_col in self.table.columns:
                bad = self.table[status_col].astype(str) != "SUCCESS"
                if bad.any():
                    raise ValueError(
                        f"Target {name!r} has {int(bad.sum())} non-SUCCESS row(s)."
                    )

            self._target_specs[name] = spec

    def __len__(self):
        return len(self.table)

    def _dtype_for_target(self, name):
        if isinstance(self.target_dtype, dict):
            return self.target_dtype.get(name)
        return self.target_dtype

    def _transform_for_target(self, name):
        if isinstance(self.target_transform, dict):
            return self.target_transform.get(name)
        return self.target_transform

    def _load_target(self, row, name):
        kind, column = self._target_specs[name]

        if kind == "path":
            path = Path(str(row[column]))
            if not path.exists():
                raise FileNotFoundError(path)
            value = np.load(path, mmap_mode=self.mmap_mode)
        else:
            value = row[column]

        value = _to_tensor(value, dtype=self._dtype_for_target(name))

        transform = self._transform_for_target(name)
        if transform is not None:
            value = transform(value)

        return value

    def __getitem__(self, index):
        row = self.table.iloc[int(index)]

        patch_path = Path(str(row["patch_path"]))
        if not patch_path.exists():
            raise FileNotFoundError(patch_path)

        image = np.load(patch_path, mmap_mode=self.mmap_mode)
        image = _to_tensor(image, dtype=self.image_dtype)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if not self.targets:
            if self.return_metadata:
                return image, row.to_dict()
            return image

        values = {
            name: self._load_target(row, name)
            for name in self.targets
        }
        target = values[self.targets[0]] if len(self.targets) == 1 else values

        if self.return_metadata:
            return image, target, row.to_dict()
        return image, target


def to_torch(dataset, *, split=None, targets=None, **kwargs):
    return PhiestaTorchDataset(
        dataset,
        split=split,
        targets=targets,
        **kwargs,
    )


def to_dataloader(
    dataset,
    *,
    split=None,
    targets=None,
    batch_size=1,
    shuffle=None,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    dataset_kwargs=None,
    **loader_kwargs,
):
    torch = _require_torch()

    adapter = to_torch(
        dataset,
        split=split,
        targets=targets,
        **dict(dataset_kwargs or {}),
    )

    if shuffle is None:
        shuffle = str(split).lower() == "train" if split is not None else False

    return torch.utils.data.DataLoader(
        adapter,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        **loader_kwargs,
    )
