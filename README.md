# Phiesta

**Python tools for ΦSat-2 L1 loading, visualization, Sentinel-2 alignment, and strict georeferencing.**

> **Documentation**
>
> - **Start here:** [`docs/overview.rst`](docs/overview.rst)
> - **Georeferencing guide:** [`docs/georeferencing.rst`](docs/georeferencing.rst)
> - **Installation:** [`docs/installation.rst`](docs/installation.rst)
> - **API quick reference:** [`docs/api_quick_reference.rst`](docs/api_quick_reference.rst)
> - **Notebook:** [`examples/Phiesta_Quickstart.ipynb`](examples/Phiesta_Quickstart.ipynb)

---

## What Phiesta does

Phiesta is a research-oriented Python toolkit for working with **ΦSat-2 L1 acquisitions**.

It provides a simple API to:

- load local ΦSat-2 L1 products;
- download/load acquisitions from Insula;
- inspect metadata, bands, raster shape, CRS, transforms, and local geolocation files;
- visualize multispectral bands, RGB, false color, and display stretches;
- compute simple band statistics and diagnostics;
- patchify acquisitions into ML-ready arrays;
- search for suitable Sentinel-2 reference acquisitions within a configurable temporal horizon;
- build Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplets;
- refine georeferencing with LightGlue-based alignment;
- return a directly usable georeference object with corners, center, GeoJSON polygon, quality metrics, and output paths.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/malodept/Phiesta.git
cd Phiesta
```

Install the base package:

```bash
python -m pip install -e .
```

For Sentinel-2 triplets and strict georeferencing:

```bash
python -m pip install -e ".[triplets]"
python -m pip install git+https://github.com/cvg/LightGlue.git
```

Quick import check:

```bash
python -c "import phiesta; from phiesta import L1_event, connect_insula; print('Phiesta OK')"
python -c "from lightglue import LightGlue, SuperPoint, SIFT; print('LightGlue OK')"
```

See [`docs/installation.rst`](docs/installation.rst) for more details.

---

## Minimal georeferencing example from an Insula product ID

```python
import json
from pathlib import Path

from phiesta import connect_insula

PRODUCT_ID = "5359"  # replace with the acquisition number

client = connect_insula()
event = client.load_l1(PRODUCT_ID)

georef = event.get_georef(
    sentinel_backend="download",
    source="simulated",
    window_days=60,          # search horizon, not a hard validity threshold
    max_cloud_cover=40.0,
    verbose=True,
)

out = Path(f"georef_{PRODUCT_ID}.json")
out.write_text(json.dumps(georef, indent=2), encoding="utf-8")

print("quality:", georef["quality"])
print("corners_lonlat:", georef["corners_lonlat"])
print("center_lonlat:", georef["center_lonlat"])
print("polygon_geojson:", georef["polygon_geojson"])
print("metrics:", georef["metrics"])
print("saved:", out)
```

The returned `georef` dictionary contains:

```python
georef["quality"]          # alignment quality label
georef["corners_lonlat"]   # acquisition footprint corners as lon/lat
georef["center_lonlat"]    # acquisition center as lon/lat
georef["polygon_geojson"]  # GeoJSON polygon
georef["metrics"]          # matching/alignment metrics
georef["paths"]            # report, preview, warped products, matches
```

---

## Minimal georeferencing example from a local L1 product

```python
import json
from pathlib import Path

from phiesta import L1_event

PRODUCT_PATH = r"/path/to/PHISAT-2_L1_..."
PRODUCT_ID = "local_product"

event = L1_event.from_path(PRODUCT_PATH)

georef = event.get_georef(
    sentinel_backend="download",
    source="simulated",
    verbose=True,
)

out = Path(f"georef_{PRODUCT_ID}.json")
out.write_text(json.dumps(georef, indent=2), encoding="utf-8")

print("quality:", georef["quality"])
print("corners_lonlat:", georef["corners_lonlat"])
print("center_lonlat:", georef["center_lonlat"])
print("polygon_geojson:", georef["polygon_geojson"])
print("metrics:", georef["metrics"])
print("saved:", out)
```

`L1_event.from_path(...)` expects the root folder of a ΦSat-2 L1 product, typically containing:

```text
bands/
geolocation/
session_<id>_metadata.json
processing_config.json
```

---

## Inspect a product

```python
from phiesta import L1_event

event = L1_event.from_path("/path/to/PHISAT-2_L1_...")

event.show_event_info()
```

This reports the product ID, sensing time, raster shape, dtype, CRS, transform, band list, local metadata paths, and useful API calls.

---

## Visualize bands

```python
event.show_all_bands(
    normalization="percentile",
    percentiles=(1, 99),
)

event.show_band(
    "NIR",
    normalization="percentile",
    percentiles=(1, 99),
)

event.show_rgb(
    bands=("RED", "GREEN", "BLUE"),
    per_band=True,
)

event.show_rgb(
    bands=("NIR", "RED", "GREEN"),
    registered=True,
    registration_master="NIR",
)
```

---

## Band statistics and display diagnostics

```python
stats = event.band_stats(
    bands=("BLUE", "GREEN", "RED", "NIR"),
    sample_size=100_000,
    percentiles=(1, 50, 99),
)

event.plot_distribution("NIR")
event.plot_display_diagnostics()
event.compare_display_stretches(bands=("NIR", "RED", "GREEN"))
```

---

## Patchify an acquisition

```python
patch_index = event.build_patch_index(
    patch_size=1024,
    stride=1024,
)

print(patch_index.head())
```

Iterate over patches:

```python
for item in event.iter_patches(
    patch_size=1024,
    stride=1024,
    bands=("RED", "GREEN", "BLUE"),
    normalization="percentile",
    percentiles=(1, 99),
):
    patch = item["patch"]
    meta = item["metadata"]
    print(meta["patch_id"], patch.shape)
    break
```

Export patches:

```python
patch_table = event.export_patches(
    out_dir="outputs/patches_5359",
    patch_size=1024,
    stride=1024,
    bands=("RED", "GREEN", "BLUE"),
    normalization="percentile",
    percentiles=(1, 99),
)
```

---

## Build a Sentinel-2 triplet manually

```python
triplet = event.build_full_sentinel_triplet(
    sentinel_backend="download",
    buffer_km=20,
    proxy_target_size=(1024, 1024),
    verbose=True,
)
```

Then refine the georeference:

```python
strict = event.refine_triplet_georeference_strict(
    triplet,
    source="simulated",
    verbose=True,
)
```

For most users, `event.get_georef(...)` is the recommended high-level entry point.

---

## Local executables and simulator code

The public Python simulator code used by Phiesta is included in:

```text
third_party/orbitalai_phisat2_sim/
```

The repository does **not** distribute local/platform-specific executable binaries. The folder:

```text
third_party/phisat2_exec/
```

contains only a README placeholder. If authorized users have local executable binaries, they can place them there locally or pass their path explicitly through the API.

---

## Repository layout

```text
phiesta/
  l0/          L0 event loading/conversion helpers
  l1/          L1 event loading, visualization, patchify, georef entry points
  georef/      geospatial utilities and feature preparation
  remote/      Insula search/download utilities
  triplets/    Sentinel-2 sourcing, simulation, alignment, strict georef
  utils/       display, patchify, metadata helpers

docs/
  overview.rst
  installation.rst
  georeferencing.rst
  api_quick_reference.rst

examples/
  Phiesta_Quickstart.ipynb
  api_smoke_test.py
```

---

## References and credits

Phiesta builds on public Earth-observation tools, data services, and research code:

- **ΦSat-2 mission:** ESA ΦSat-2 mission material.
- **Sentinel-2 access:** Copernicus Data Space Ecosystem catalogue and OData download API.
- **OrbitalAI / ΦSat-2 simulator:** public OrbitalAI challenge materials and simulator code, vendored in `third_party/orbitalai_phisat2_sim/`.
- **Feature matching:** LightGlue for local feature matching.
- **Land-cover context, when used:** ESA WorldCover 2021 v200.

External services may require their own credentials and terms of use. Phiesta does not distribute proprietary platform-specific executables.

---

## Status

Phiesta is an early research-oriented toolkit. APIs may evolve as the georeferencing workflow, strict alignment strategy, and documentation are improved.

## Product-level inspection API

Phiesta provides product-level utilities for opening, inspecting, screening, and comparing PhiSat-2 products.

### Basic usage

```python
import phiesta

event = phiesta.open_product("6008", level="L1C")

card = phiesta.product_card(event)
manifest = phiesta.file_manifest(event)
families = phiesta.file_family_summary(event)
switches = phiesta.processing_switches(event)
rasters = phiesta.raster_inventory(event)

quality = phiesta.quality_report(event)
```

### Product gallery

```python
import phiesta

df = phiesta.product_gallery(
    ["5978", "5979", "6008", "6025"],
    level="L1C",
    out_path="outputs/l1c_screening_gallery.png",
)
```

### Processing-level comparison

```python
import phiesta

l1a = phiesta.open_product("6008", level="L1A")
l1c = phiesta.open_product("6008", level="L1C")

comparison = phiesta.compare_levels(
    l1a,
    l1c,
    include_shift=True,
)
```

### Quickstart example

Run from the repository root:

```bash
PYTHONPATH="$PWD" python examples/product_inspection_quickstart.py
```

On the ESA PhiLab container environment, use `phipy` instead of `python`.

The example demonstrates:

- product cards and file manifests;
- processing-switch inspection;
- raster inventory extraction;
- heuristic product screening;
- annotated product galleries;
- L1A/L1C comparison with inter-band shift diagnostics.

