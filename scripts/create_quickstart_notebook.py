from __future__ import annotations

import json
from pathlib import Path

OUT = Path("examples/Phiesta_Quickstart.ipynb")


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


cells = []

cells.append(md(r"""
# Phiesta Quickstart

This is the main user-facing notebook for Phiesta.

It follows the current public workflow from mission data to machine learning:

1. connect to Insula;
2. search/select ΦSat-2 products;
3. load and inspect L1 data;
4. visualize bands and composites;
5. optionally improve georeferencing with Sentinel-2;
6. build reproducible patch datasets;
7. create leakage-safe train/validation/test splits;
8. attach labels or raster targets;
9. consume the result with PyTorch;
10. optionally run WorldCover, L0, and full Sentinel/simulator workflows.

Heavy or network-intensive operations are disabled by default.
"""))

cells.append(md(r"""
## 0. Installation

From a clone of the repository:

```bash
pip install -e .
```

For Sentinel/simulator workflows:

```bash
pip install -e ".[triplets]"
```

For the PyTorch adapter:

```bash
pip install -e ".[ml]"
```

Both optional stacks can be installed together with:

```bash
pip install -e ".[triplets,ml]"
```
"""))

cells.append(md(r"""
## 1. Configuration

The defaults are deliberately safe. Loading one L1 acquisition is enabled; expensive
georeferencing, WorldCover scanning, dataset export, L0, and full triplet generation
must be opted into explicitly.
"""))

cells.append(code(r"""
PRODUCT_ID = "5359"

# Optional: put caches/datasets on another disk.
# Example on Windows: r"D:\PhiestaCache"
CACHE_DIR = None
DATASET_DIR = "datasets/quickstart"
TRIPLET_ROOT = "data/triplets"

RUN_SEARCH = True
RUN_WORLDCOVER_SEARCH = False
RUN_GEOREFERENCE = False
RUN_DATASET_WORKFLOW = False
RUN_TORCH_DEMO = False
RUN_L0_EXAMPLE = False
RUN_FULL_TRIPLET = False
"""))

cells.append(md(r"""
## 2. Imports and Insula connection

`connect_insula()` prompts for credentials interactively; credentials should not be
hard-coded in notebooks.
"""))

cells.append(code(r"""
from pathlib import Path

import numpy as np
import pandas as pd

from phiesta import (
    L0_event,
    L1_event,
    column_target,
    connect_insula,
    open_dataset,
)

client = connect_insula(cache_dir=CACHE_DIR)
"""))

cells.append(md(r"""
## 3. Search before downloading

Search tables are useful selections for later dataset construction because they retain
catalog metadata such as product ids, acquisition times, and approximate geometry.
"""))

cells.append(code(r"""
if RUN_SEARCH:
    search_table = client.search_l1_table(page=0, results_per_page=5)
    display(search_table)
else:
    print("Skipped. Set RUN_SEARCH = True to query Insula.")
"""))

cells.append(md(r"""
### 3.1 Optional land-cover-aware selection

WorldCover is only one possible selection mechanism. The default threshold corresponds
to presence (`min_fraction=1e-6`), and the default spatial tolerance accounts for the
known catalog geolocation uncertainty.

This operation can issue many remote requests, so it is disabled by default.
"""))

cells.append(code(r"""
if RUN_WORLDCOVER_SEARCH:
    mangrove_candidates = client.search_l1_worldcover(
        "mangrove",
        min_fraction=1e-6,
        max_catalog_products=50,
    )
    display(
        mangrove_candidates[
            [
                "product_id",
                "worldcover_fraction",
                "worldcover_pixels",
                "spatial_tolerance_km",
            ]
        ]
    )
else:
    print("Skipped. Set RUN_WORLDCOVER_SEARCH = True to scan WorldCover.")
"""))

cells.append(md(r"""
## 4. Load one ΦSat-2 L1 acquisition

The acquisition is downloaded if necessary or reused from the configured cache.
"""))

cells.append(code(r"""
event = client.load_l1(PRODUCT_ID)
event.show_event_info()
"""))

cells.append(md(r"""
## 5. Inspect the product

The event API works directly with bands, cubes, metadata, and display utilities.
"""))

cells.append(code(r"""
cube = event.to_cube()
print("shape:", cube.shape)
print("dtype:", cube.dtype)
print("metadata keys:", sorted(event.meta.keys()))

for band_name in ["PAN", "BLUE", "GREEN", "RED", "RE1", "RE2", "RE3", "NIR"]:
    band = event.get_band(band_name)
    print(
        f"{band_name:>5}: shape={band.shape}, dtype={band.dtype}, "
        f"min={float(np.nanmin(band)):.2f}, max={float(np.nanmax(band)):.2f}"
    )
"""))

cells.append(md(r"""
## 6. Visualize RGB, NIR, and false color
"""))

cells.append(code(r"""
event.show_rgb(
    normalization="percentile",
    percentiles=(1, 99),
    per_band=True,
)

event.show_band(
    "NIR",
    normalization="percentile",
    percentiles=(1, 99),
)

event.show_rgb(
    bands=("NIR", "RED", "GREEN"),
    normalization="percentile",
    percentiles=(1, 99),
    per_band=True,
)
"""))

cells.append(md(r"""
### 6.1 Optional display-only inter-band registration

This helps visualization; it does not rewrite the source product.
"""))

cells.append(code(r"""
event.show_rgb(
    bands=("NIR", "RED", "GREEN"),
    registered=True,
    registration_master="NIR",
    normalization="percentile",
    percentiles=(1, 99),
    per_band=True,
)
"""))

cells.append(md(r"""
## 7. Optional Sentinel-assisted georeferencing

`event.georeference()` builds the Sentinel-assisted registration, exports a standard
georeferenced GeoTIFF, and returns a new `L1_event` backed by the corrected raster.

The normal event API continues to work on the returned product.

The default Sentinel search horizon is ±60 days. Use `window_days=7` if a ±7-day
window is desired.
"""))

cells.append(code(r"""
if RUN_GEOREFERENCE:
    georef_product = event.georeference()

    print("path:", georef_product.meta["path"])
    print("crs:", georef_product.meta["crs"])
    print("transform:", georef_product.meta["transform"])

    georef_product.show_rgb()
else:
    print("Skipped. Set RUN_GEOREFERENCE = True to run Sentinel-assisted georeferencing.")
"""))

cells.append(md(r"""
## 8. Generic dataset construction

Dataset construction is independent from selection. The same builder accepts:

- a list of product ids;
- an Insula search table;
- a WorldCover-filtered table;
- a CSV;
- any custom pandas DataFrame containing `product_id`.

Every additional selection column is propagated into the acquisition and patch
manifests.

This small demo uses one acquisition and at most four patches. It is disabled by
default because it writes data to disk.
"""))

cells.append(code(r"""
if RUN_DATASET_WORKFLOW:
    selection = pd.DataFrame(
        {
            "product_id": [PRODUCT_ID],
            "label_name": ["example"],
            "label_id": [0],
        }
    )

    dataset = client.build_l1_dataset(
        selection,
        out_dir=DATASET_DIR,
        patch_size=512,
        stride=512,
        patch_limit=4,
        normalize=False,
    )

    print(dataset)
    display(dataset.acquisitions)
    display(dataset.patches.head())
else:
    dataset = None
    print("Skipped. Set RUN_DATASET_WORKFLOW = True to build a small dataset.")
"""))

cells.append(md(r"""
## 9. Leakage-safe splits

Splits are assigned at acquisition/group level and then propagated to all patches, so
patches from the same acquisition cannot leak across train/validation/test.

Because the default demo contains only one acquisition, it uses a train-only split.
For a real multi-acquisition dataset, use e.g.:

```python
dataset.make_splits(train=0.8, val=0.1, test=0.1, seed=42)
```

Related acquisitions can be kept together with `group_by="pass_id"` (or any manifest
column), and a spatial mode is available with `method="spatial"`.
"""))

cells.append(code(r"""
if dataset is not None:
    dataset.make_splits(
        train=1.0,
        val=0.0,
        test=0.0,
        seed=42,
        overwrite=True,
    )
    display(dataset.split_summary())
else:
    print("Dataset workflow was not run.")
"""))

cells.append(md(r"""
## 10. Targets and labels

Targets are also independent from selection and splitting. Here we formalize the
`label_id` column as a classification target.

More generally, `dataset.add_target(...)` can attach scalars, arrays, file paths, or
raster-aligned masks.
"""))

cells.append(code(r"""
if dataset is not None:
    dataset.add_target(
        "class",
        column_target("label_id"),
    )

    display(
        dataset.patches[
            [
                "dataset_patch_id",
                "split",
                "target_class_value",
                "target_class_status",
            ]
        ]
    )
else:
    print("Dataset workflow was not run.")
"""))

cells.append(md(r"""
## 11. Optional PyTorch DataLoader

Install the `ml` extra first. Image patches are loaded lazily from `.npy` files.

`to_torch()` converts image arrays to `float32` by default; this dtype conversion is
**not** a radiometric normalization. Choose task-appropriate normalization explicitly
for real experiments.
"""))

cells.append(code(r"""
if RUN_TORCH_DEMO:
    if dataset is None:
        dataset = open_dataset(DATASET_DIR)

    loader = dataset.to_dataloader(
        split="train",
        targets="class",
        batch_size=2,
        shuffle=False,
        dataset_kwargs={"target_dtype": "long"},
    )

    images, labels = next(iter(loader))
    print("images:", images.shape, images.dtype)
    print("labels:", labels.shape, labels.dtype)
else:
    print("Skipped. Set RUN_TORCH_DEMO = True after building/opening the dataset.")
"""))

cells.append(md(r"""
## 12. Optional georeferenced training data

For spatially aligned external labels, build corrected L1 rasters before patch
extraction:

```python
dataset = client.build_l1_dataset(
    selection,
    out_dir="datasets/georeferenced",
    georeference=True,
    patch_size=512,
)
```

Then raster targets can be aligned exactly to each patch:

```python
from phiesta import raster_target

dataset.add_target(
    "landcover",
    raster_target("labels/landcover.tif"),
)
```

For local ESA WorldCover categorical rasters, use `worldcover_target(...)`, which uses
nearest-neighbour resampling to preserve class codes.
"""))

cells.append(md(r"""
## 13. Optional L0 example

L0 support remains available through the same client. It is disabled by default because
the quickstart's main path is L1 → dataset → ML.
"""))

cells.append(code(r"""
if RUN_L0_EXAMPLE:
    l0 = client.load_l0(PRODUCT_ID)
    l0.show_event_info()
else:
    print("Skipped. Set RUN_L0_EXAMPLE = True to load the matching L0 product.")
"""))

cells.append(md(r"""
## 14. Advanced Sentinel / simulator triplet

The full triplet workflow constructs aligned:

1. Sentinel-2;
2. simulated ΦSat-2;
3. real ΦSat-2.

This is intentionally disabled by default because it can take several minutes, uses the
ΦSat-2 simulator, and benefits strongly from a GPU environment.
"""))

cells.append(code(r"""
if RUN_FULL_TRIPLET:
    full_triplet = event.build_full_sentinel_triplet(
        output_root=TRIPLET_ROOT,
        sentinel_backend="download",
        buffer_km=20.0,
        proxy_target_size=(1024, 1024),
        verbose=True,
    )

    event.inspect_full_sentinel_triplet(full_triplet)
    event.show_full_sentinel_triplet(full_triplet)
else:
    print("Skipped. Set RUN_FULL_TRIPLET = True to build the full triplet.")
"""))

cells.append(md(r"""
## 15. Where to go next

- `README.md` — public overview and installation.
- `docs/api_quick_reference.rst` — compact API reference.
- `examples/dataset_training_quickstart.py` — end-to-end generic dataset → PyTorch training example.
- `examples/full_sentinel_triplet_demo.py` — focused advanced triplet example.
- `THIRD_PARTY_NOTICES.md` — provenance and third-party notices.

A typical research workflow is now:

```text
search / custom selection
        ↓
build_l1_dataset
        ↓
optional georeference
        ↓
patches
        ↓
leakage-safe splits
        ↓
0..N targets
        ↓
PyTorch Dataset / DataLoader
```
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(f"Wrote {OUT} with {len(cells)} cells")
