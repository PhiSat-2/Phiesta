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

This notebook is the main user-facing entry point for Phiesta.

It shows the practical API workflows:

1. connect to Insula;
2. search ΦSat-2 L1/L0 products;
3. inspect product search results, dates, filenames, and catalog geometry;
4. load/download products from Insula;
5. load products from local folders;
6. inspect metadata and band information;
7. display RGB composites, individual bands, and all bands;
8. use false-color and vegetation composites;
9. use display registration between bands;
10. analyze band distributions and display stretches;
11. work directly with NumPy arrays, cubes, and patches;
12. inspect Insula catalog geometry;
13. register L0 to L1 space;
14. build Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplets;
15. run batch workflows.

Heavy workflows are controlled by boolean flags, so the notebook can be run safely step by step.
"""))

cells.append(md(r"""
## 0. Configuration

Choose one acquisition and decide whether to run heavy workflows.

`RUN_FULL_TRIPLET = False` by default because full triplet generation can take several minutes and requires GPU support.
"""))

cells.append(code(r"""
PRODUCT_ID = "5359"

RUN_SEARCH_EXAMPLES = True
RUN_L0_EXAMPLES = False
RUN_L0_L1_REGISTRATION = False
RUN_SENTINEL_LIGHT_WORKFLOW = False
RUN_FULL_TRIPLET = False
RUN_BATCH = False

OUTPUT_ROOT = "data/triplets"
"""))

cells.append(md(r"""
## 1. Imports and connection to Insula
"""))

cells.append(code(r"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phiesta import connect_insula, L1_event, L0_event

client = connect_insula()
"""))

cells.append(md(r"""
## 2. Search products on Insula

The search results are useful before downloading anything.

They usually contain:

- product identifier / filename;
- acquisition datetime;
- catalog geometry;
- centroid;
- footprint corners.

This is the right place to inspect dates and approximate georeferencing.
"""))

cells.append(code(r"""
if RUN_SEARCH_EXAMPLES:
    l1_results = client.search_l1(page=0, results_per_page=5)
    print(type(l1_results))
    print(l1_results if isinstance(l1_results, dict) else l1_results[:1])
else:
    print("Skipped. Set RUN_SEARCH_EXAMPLES = True to run this cell.")
"""))

cells.append(md(r"""
### 2.1 Extract filename, datetime, and catalog geometry from search results

The exact feature structure can vary slightly depending on the Insula client, so this cell uses robust helpers.
"""))

cells.append(code(r"""
from phiesta.remote.catalog_geometry import (
    catalog_geo_from_feature,
    get_catalog_center,
    get_catalog_bbox_lonlat,
    get_catalog_identifier,
)

def iter_features(search_result):
    if isinstance(search_result, dict):
        return search_result.get("features", [])
    return list(search_result)

def compact_feature_summary(feature):
    catalog_geo = catalog_geo_from_feature(feature)
    props = feature.get("properties", {}) if isinstance(feature, dict) else {}

    identifier = get_catalog_identifier(catalog_geo) if catalog_geo else None
    center = get_catalog_center(catalog_geo, order="latlon") if catalog_geo else None
    bbox = get_catalog_bbox_lonlat(catalog_geo) if catalog_geo else None

    return {
        "feature_id": feature.get("id") if isinstance(feature, dict) else None,
        "identifier": identifier or props.get("identifier") or props.get("title") or props.get("filename"),
        "filename": (
            props.get("filename")
            or props.get("productIdentifier")
            or props.get("product_identifier")
            or props.get("title")
        ),
        "start_datetime": (
            props.get("start_datetime")
            or props.get("startDate")
            or props.get("start")
            or props.get("datetime")
        ),
        "completion_datetime": (
            props.get("completion_datetime")
            or props.get("completionDate")
            or props.get("end")
        ),
        "center_lat": center[0] if center else None,
        "center_lon": center[1] if center else None,
        "bbox_lonlat": bbox,
        "corners_lonlat": catalog_geo.get("corners_lonlat") if catalog_geo else None,
    }

if RUN_SEARCH_EXAMPLES:
    features = iter_features(l1_results)
    summaries = [compact_feature_summary(f) for f in features]
    display(pd.DataFrame(summaries))
else:
    print("Skipped.")
"""))

cells.append(md(r"""
## 3. Main ways to open or download products

Phiesta supports several entry points.

### Remote L1 by acquisition id

```python
event = client.load_l1("5359")
```

### Remote L1 by full product name or identifier

```python
event = client.load_l1("PHISAT-2_L1_000005359_20260512183354_20260512183357_236B0FFC")
```

### Search first, then load

```python
results = client.search_l1(page=0, results_per_page=5)
feature = results["features"][0]
# Then load using the id / identifier / filename depending on the feature structure.
```

### Remote L0

```python
l0 = client.load_l0("5359")
```

### Local L1

```python
event = L1_event.from_path("path/to/local/PHISAT-2_L1_product")
```

### Local L0

```python
l0 = L0_event.from_path("path/to/local/PHISAT-2_L0_product")
```
"""))

cells.append(md(r"""
## 4. Load one ΦSat-2 L1 acquisition

The product is downloaded or reused from the local cache.
"""))

cells.append(code(r"""
event = client.load_l1(PRODUCT_ID)
"""))

cells.append(md(r"""
## 5. Product information

`show_event_info()` should be the quick entry point for a loaded product.

It should show product identity, raster metadata, band information, catalog geometry, local paths, and useful API calls.
"""))

cells.append(code(r"""
event.show_event_info()
"""))

cells.append(md(r"""
## 6. Raw metadata

The event stores useful metadata in `event.meta`.
"""))

cells.append(code(r"""
print("Metadata keys:")
print(sorted(event.meta.keys()))

print("\nRaster info:")
for key in ["count", "width", "height", "dtype", "crs", "transform"]:
    print(f"{key}: {event.meta.get(key)}")

print("\nWavelengths:")
print(event.meta.get("band_wavelength_nm"))

print("\nSensing time:")
print(event.meta.get("sensing_time"))

print("\nLocal path:")
print(event.meta.get("resolved_product_folder") or event.meta.get("path"))
"""))

cells.append(md(r"""
## 7. Band access

Bands can be accessed by:

- integer index: `0`, `1`, `2`, ...
- wavelength: `842`, `842.0`
- alias: `PAN`, `BLUE`, `GREEN`, `RED`, `RE1`, `RE2`, `RE3`, `NIR`
"""))

cells.append(code(r"""
for selector in ["PAN", "BLUE", "GREEN", "RED", "RE1", "RE2", "RE3", "NIR"]:
    band = event.get_band(selector)
    print(
        f"{selector:>5}: shape={band.shape}, dtype={band.dtype}, "
        f"min={float(np.nanmin(band)):.2f}, max={float(np.nanmax(band)):.2f}"
    )
"""))

cells.append(md(r"""
## 8. Show all bands

This is the simplest way to visually inspect the whole product.
"""))

cells.append(code(r"""
event.show_all_bands(
    normalization="percentile",
    percentiles=(1, 99),
    figsize=(16, 10),
)
"""))

cells.append(md(r"""
## 9. Natural RGB

For raw ΦSat-2 data, percentile normalization is usually necessary.
"""))

cells.append(code(r"""
event.show_rgb(
    bands=("RED", "GREEN", "BLUE"),
    normalization="percentile",
    percentiles=(1, 99),
    per_band=True,
    figsize=(8, 8),
)
"""))

cells.append(md(r"""
## 10. Individual bands

Single-band visualization is useful to inspect PAN, NIR, and red-edge bands.
"""))

cells.append(code(r"""
event.show_band(
    "NIR",
    normalization="percentile",
    percentiles=(1, 99),
    figsize=(8, 8),
)
"""))

cells.append(code(r"""
event.show_band(
    "PAN",
    normalization="percentile",
    percentiles=(1, 99),
    figsize=(8, 8),
)
"""))

cells.append(md(r"""
## 11. False-color and vegetation composites

For vegetation and coastal scenes, false-color composites are often more informative than natural RGB.

In `("NIR", "RED", "GREEN")`, vegetation appears red because NIR is displayed in the red channel.
"""))

cells.append(code(r"""
event.show_rgb(
    bands=("NIR", "RED", "GREEN"),
    normalization="percentile",
    percentiles=(1, 99),
    per_band=True,
    figsize=(8, 8),
)
"""))

cells.append(code(r"""
event.show_rgb(
    bands=("NIR", "RE1", "RED"),
    normalization="percentile",
    percentiles=(1, 99),
    per_band=True,
    figsize=(8, 8),
)
"""))

cells.append(md(r"""
## 12. Display registration between bands

Some ΦSat-2 bands can be slightly shifted relative to each other.

For display, Phiesta can register bands to a master band before building the composite.
This is especially useful for RGB and false-color images.

This is a display operation: it helps visualization, but it does not rewrite the original product on disk.
"""))

cells.append(code(r"""
event.show_rgb(
    bands=("NIR", "RED", "GREEN"),
    registered=True,
    registration_master="NIR",
    normalization="percentile",
    percentiles=(1, 99),
    per_band=True,
    figsize=(8, 8),
)
"""))

cells.append(md(r"""
### 12.1 Show all bands after display registration

This is useful to visually check whether the bands align better.
"""))

cells.append(code(r"""
event.show_all_bands(
    registered=True,
    registration_master="NIR",
    normalization="percentile",
    percentiles=(1, 99),
    figsize=(16, 10),
)
"""))

cells.append(md(r"""
## 13. Display diagnostics

`plot_display_diagnostics()` gives a compact visual report:

- natural RGB;
- false color;
- vegetation composite;
- normalized NIR;
- raw NDVI-like map;
- raw NDVI-like distribution;
- band statistics.

Important: the raw NDVI-like panel is computed from raw DN values. It is useful as a display diagnostic, not as a physically calibrated NDVI.
"""))

cells.append(code(r"""
diag = event.plot_display_diagnostics(
    product_id=PRODUCT_ID,
    percentiles=(1, 99),
    registered=True,
    registration_master="NIR",
    out_png=f"outputs/display_diagnostics/display_diagnostics_{PRODUCT_ID}.png",
)

pd.DataFrame(diag["stats"]).T
"""))

cells.append(md(r"""
## 14. Compare display stretches

This helps choose whether `(0.5, 99.5)`, `(1, 99)`, `(2, 98)`, or `(5, 95)` is best for a scene.
"""))

cells.append(code(r"""
event.compare_display_stretches(
    bands=("NIR", "RED", "GREEN"),
    registered=True,
    registration_master="NIR",
    out_png=f"outputs/display_diagnostics/stretch_nir_red_green_{PRODUCT_ID}.png",
)
"""))

cells.append(code(r"""
event.compare_display_stretches(
    bands=("NIR", "RE1", "RED"),
    registered=True,
    registration_master="NIR",
    out_png=f"outputs/display_diagnostics/stretch_nir_re1_red_{PRODUCT_ID}.png",
)
"""))

cells.append(md(r"""
## 15. Band statistics and distributions

Phiesta can compute sampled statistics and plot value distributions.

This is useful because raw ΦSat-2 bands are not necessarily on comparable physical scales.
"""))

cells.append(code(r"""
stats = event.band_stats(
    bands=("PAN", "BLUE", "GREEN", "RED", "RE1", "RE2", "RE3", "NIR"),
    percentiles=(0, 1, 2, 5, 50, 95, 98, 99, 100),
    sample=1_000_000,
)

pd.DataFrame(stats).T
"""))

cells.append(code(r"""
event.plot_distribution(
    bands=("BLUE", "GREEN", "RED", "RE1", "NIR"),
    bins=256,
    sample=1_000_000,
    log_y=True,
    percentiles=(1, 2, 5, 50, 95, 98, 99),
    hist_range_percentiles=(0.1, 99.9),
    figsize=(12, 7),
)
"""))

cells.append(md(r"""
## 16. Work directly with NumPy arrays

Every band can be extracted as a NumPy array.

You can also build a full cube manually.
"""))

cells.append(code(r"""
# Single band
nir = event.get_band("NIR")
print("NIR:", nir.shape, nir.dtype)

# Full cube: shape = (bands, height, width)
cube = np.stack([event.get_band(i) for i in range(event.meta["count"])])
print("cube:", cube.shape, cube.dtype)
"""))

cells.append(md(r"""
## 17. Crop a patch from an acquisition

This is useful for quick experiments, prototyping, or patch-based ML.
"""))

cells.append(code(r"""
y0, y1 = 1000, 1600
x0, x1 = 1000, 1600

patch_nir = event.get_band("NIR")[y0:y1, x0:x1]
patch_rgb = np.dstack([
    event.get_band("RED")[y0:y1, x0:x1],
    event.get_band("GREEN")[y0:y1, x0:x1],
    event.get_band("BLUE")[y0:y1, x0:x1],
])

print("patch_nir:", patch_nir.shape)
print("patch_rgb:", patch_rgb.shape)
"""))

cells.append(md(r"""
## 18. Display normalization for custom arrays

If a band is technically between 0 and 255 but most values are between 20 and 100, percentile clipping is usually more useful than min-max display.
"""))

cells.append(code(r"""
def normalize_band_for_display(x, percentiles=(1, 99)):
    x = x.astype("float32")
    valid = np.isfinite(x)
    vals = x[valid]
    lo, hi = np.percentile(vals, percentiles)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

plt.figure(figsize=(6, 6))
plt.imshow(normalize_band_for_display(patch_nir, percentiles=(1, 99)), cmap="gray")
plt.title("NIR patch, percentile normalized")
plt.axis("off")
plt.show()
"""))

cells.append(md(r"""
## 19. Clip a full 8-band cube for analysis

Display clipping is useful, but sometimes you also want a clipped/normalized cube for downstream experiments.

This example keeps the original cube shape and applies percentile clipping independently to each band.
"""))

cells.append(code(r"""
def normalize_cube_per_band(cube, percentiles=(1, 99)):
    cube = cube.astype("float32")
    out = np.zeros_like(cube, dtype="float32")

    for b in range(cube.shape[0]):
        x = cube[b]
        valid = np.isfinite(x)
        vals = x[valid]
        lo, hi = np.percentile(vals, percentiles)
        out[b] = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

    return out

cube_display = normalize_cube_per_band(cube, percentiles=(1, 99))
print(cube_display.shape, cube_display.min(), cube_display.max())
"""))

cells.append(md(r"""
## 20. Catalog geometry from Insula

If the acquisition was loaded from Insula, the catalog geometry is stored in the metadata.

This geometry is useful for search and screening workflows, but it should not be treated as perfect pixel-level georeferencing.
"""))

cells.append(code(r"""
catalog_geo = event.meta.get("catalog_geo")

if catalog_geo is None:
    print("No catalog geometry available.")
else:
    print("center_lonlat:", catalog_geo.get("center_lonlat"))
    print("corners_lonlat:")
    for c in catalog_geo.get("corners_lonlat", []):
        print(" ", c)
"""))

cells.append(md(r"""
## 21. L0 loading

L0 products can be downloaded/opened too.

This section is disabled by default because L0 loading/conversion can be heavier than L1 loading.
"""))

cells.append(code(r"""
if RUN_L0_EXAMPLES:
    l0_event = client.load_l0(PRODUCT_ID)

    l0_event.show_event_info()
    l0_event.show_all_bands(
        normalization="percentile",
        percentiles=(1, 99),
        figsize=(16, 10),
    )
else:
    print("Skipped. Set RUN_L0_EXAMPLES = True to run this cell.")
"""))

cells.append(md(r"""
## 22. L0 to L1 registration

This is different from display registration between bands.

L0/L1 registration tries to align a L0 product to the space of the corresponding L1 product.
It is useful when comparing raw and processed products.

The exact parameters depend on the acquisition and on the selected master band.
"""))

cells.append(code(r"""
if RUN_L0_L1_REGISTRATION:
    from phiesta.utils.l0_l1_registration import register_l0_to_l1_space

    l0_event = client.load_l0(PRODUCT_ID)
    l1_event = event

    l0_registered = register_l0_to_l1_space(
        l0_event=l0_event,
        l1_event=l1_event,
        master_band=7,          # often NIR
        max_shifts=(300, 300),
    )

    l0_registered.show_rgb(
        bands=("RED", "GREEN", "BLUE"),
        normalization="percentile",
        percentiles=(1, 99),
        per_band=True,
    )
else:
    print("Skipped. Set RUN_L0_L1_REGISTRATION = True to run this cell.")
"""))

cells.append(md(r"""
## 23. Optional: lightweight Sentinel source / crop workflow

This searches for a Sentinel-2B source and can build a Sentinel crop.

This is lighter than the full triplet pipeline, but still requires access to the Sentinel SAFE products mounted on the machine.
"""))

cells.append(code(r"""
if RUN_SENTINEL_LIGHT_WORKFLOW:
    triplet_light = event.build_sentinel_triplet(
        window_days=15,
        max_cloud_cover=20,
        buffer_km=20.0,
        run_sentinel_source=True,
        run_sentinel_crop=True,
        run_simulation=False,
    )

    triplet_light.summary()
else:
    print("Skipped. Set RUN_SENTINEL_LIGHT_WORKFLOW = True to run this cell.")
"""))

cells.append(md(r"""
## 24. Full Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplet

This is the most advanced workflow.

It builds:

1. real ΦSat-2;
2. Sentinel-2B warped to the ΦSat-2 grid;
3. simulated ΦSat-2 warped to the real ΦSat-2 grid.

This can take several minutes.
"""))

cells.append(code(r"""
if RUN_FULL_TRIPLET:
    full_triplet = event.build_full_sentinel_triplet(
        buffer_km=20.0,
        proxy_target_size=(1024, 1024),
        final_margin_pct=0.15,
        verbose=True,
    )

    event.inspect_full_sentinel_triplet(full_triplet)
    event.show_full_sentinel_triplet(full_triplet)

    full_triplet["paths"]
else:
    print("Skipped. Set RUN_FULL_TRIPLET = True to run this cell.")
"""))

cells.append(md(r"""
## 25. Batch full triplet generation

Use this when several acquisitions must be processed.
"""))

cells.append(code(r"""
if RUN_BATCH:
    from phiesta.triplets.batch import build_full_sentinel_triplets_batch

    batch = build_full_sentinel_triplets_batch(
        client=client,
        product_ids=[PRODUCT_ID],
        output_root=OUTPUT_ROOT,
        verbose=True,
    )

    display(pd.DataFrame(batch["rows"]))
else:
    print("Skipped. Set RUN_BATCH = True to run this cell.")
"""))

cells.append(md(r"""
## 26. Load a local L1 product

Remote loading through Insula is not required. You can also load an already downloaded local product.

Edit the path below before running.
"""))

cells.append(code(r"""
# Example only. Update the path before running.
# local_l1 = L1_event.from_path(
#     product_folder="data/l1/PHISAT-2_L1_000005359_20260512183354_20260512183357_236B0FFC",
#     scene_id=0,
#     product_kind="BC",
# )
# local_l1.show_event_info()
"""))

cells.append(md(r"""
## 27. Load a local L0 product

Edit the path below before running.
"""))

cells.append(code(r"""
# Example only. Update the path before running.
# local_l0 = L0_event.from_path("data/l0/PHISAT-2_L0_PRODUCT_FOLDER")
# local_l0.show_event_info()
# local_l0.show_rgb()
"""))

cells.append(md(r"""
## 28. Useful API discovery

If you do not know what an object can do, inspect its public methods.
"""))

cells.append(code(r"""
public_methods = [
    name for name in dir(event)
    if not name.startswith("_") and callable(getattr(event, name))
]

public_methods
"""))

cells.append(md(r"""
## 29. Suggested workflows

### Find products before downloading

```python
results = client.search_l1(page=0, results_per_page=20)
features = results["features"]
```

Each feature can contain acquisition datetime and catalog geometry.

### Open products

```python
event = client.load_l1("5359")
l0 = client.load_l0("5359")
local = L1_event.from_path("path/to/product")
```

### Visual exploration

```python
event.show_event_info()
event.show_all_bands()
event.show_rgb()
event.show_band("NIR")
event.plot_display_diagnostics()
```

### Vegetation / coastal scenes

```python
event.show_rgb(("NIR", "RED", "GREEN"), registered=True)
event.show_rgb(("NIR", "RE1", "RED"), registered=True)
event.compare_display_stretches(("NIR", "RED", "GREEN"))
```

### L0/L1 comparison

```python
from phiesta.utils.l0_l1_registration import register_l0_to_l1_space
l0_registered = register_l0_to_l1_space(l0_event, l1_event)
```

### Triplet generation

```python
triplet = event.build_full_sentinel_triplet()
event.inspect_full_sentinel_triplet(triplet)
event.show_full_sentinel_triplet(triplet)
```

### Batch processing

```python
from phiesta.triplets.batch import build_full_sentinel_triplets_batch
batch = build_full_sentinel_triplets_batch(client, ["5359", "5095"])
```
"""))

cells.append(md(r"""
## 30. Current limitations and future directions

Important limitations:

- catalog georeferencing is useful but not always precise enough for pixel-level overlay;
- raw spectral indices should not be interpreted as physical reflectance indices unless the data are calibrated;
- full triplet generation can be memory-intensive;
- very strict alignment modes are not exposed yet.

Useful future features:

- search acquisitions interactively on a map;
- draw a rectangle and list/download ΦSat-2 products;
- attach WorldCover / OSM / Sentinel-2 context automatically;
- compute WorldCover class fractions for an acquisition or a patch;
- patchify acquisitions into ML-ready datasets;
- support generic image/cube formats, not only ΦSat-2 product folders;
- support stricter multi-stage alignment modes, for example fast/strict/expert.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=2), encoding="utf-8")

print(f"written {OUT}")
print(f"cells: {len(cells)}")
