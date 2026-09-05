"""
End-to-end Phiesta dataset -> split -> target -> PyTorch training example.

Input CSV must contain product_id,label.
Run:
    python examples/dataset_training_quickstart.py selection.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from phiesta import column_target, connect_insula


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("selection_csv", type=Path)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--out-dir", type=Path, default=Path("datasets/training_example"))
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def encode_labels(selection):
    if "product_id" not in selection.columns or "label" not in selection.columns:
        raise ValueError("selection CSV must contain product_id and label columns.")
    if selection["label"].isna().any():
        raise ValueError("selection CSV contains missing labels.")

    classes = sorted(selection["label"].astype(str).unique().tolist())
    mapping = {name: i for i, name in enumerate(classes)}

    out = selection.copy()
    out["label_name"] = out["label"].astype(str)
    out["label_id"] = out["label_name"].map(mapping).astype(int)
    return out, classes


def main():
    args = parse_args()
    selection, classes = encode_labels(pd.read_csv(args.selection_csv))

    if len(selection) < 3:
        raise ValueError(
            "Use at least 3 acquisitions; 10+ is preferable for meaningful split ratios."
        )

    client = connect_insula(cache_dir=args.cache_dir)

    dataset = client.build_l1_dataset(
        selection,
        out_dir=args.out_dir,
        patch_size=args.patch_size,
        stride=args.patch_size,
        # Keep source values unchanged. Choose task-specific radiometric
        # normalization explicitly for real experiments.
        normalize=False,
    )

    dataset.make_splits(
        train=0.8,
        val=0.1,
        test=0.1,
        seed=args.seed,
        overwrite=True,
    )

    # label_id was propagated from the selection table to every patch.
    dataset.add_target("class", column_target("label_id"))

    loader = dataset.to_dataloader(
        split="train",
        targets="class",
        batch_size=args.batch_size,
        shuffle=True,
        dataset_kwargs={"target_dtype": "long"},
    )

    images, labels = next(iter(loader))

    print("classes:", classes)
    print("images:", tuple(images.shape), images.dtype)
    print("labels:", tuple(labels.shape), labels.dtype)
    print(dataset.split_summary())

    # Minimal optimization step only to validate the complete training contract.
    # Replace this scaling/model with task-appropriate choices.
    scale = (
        images.abs()
        .flatten(1)
        .amax(dim=1)
        .clamp_min(1.0)
        .view(-1, 1, 1, 1)
    )
    model_input = images / scale

    model = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LazyLinear(len(classes)),
    )

    logits = model(model_input)
    loss = nn.CrossEntropyLoss()(logits, labels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("one training step completed; loss =", float(loss.detach()))


if __name__ == "__main__":
    main()
