# Phiesta

[![CI](https://github.com/PhiSat-2/Phiesta/actions/workflows/ci.yml/badge.svg)](https://github.com/PhiSat-2/Phiesta/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**Mission-aware Python toolkit for ΦSat-2 product access, inspection, validation, comparison, and georeferencing.**

Phiesta provides a single Python interface across raw **L0**, processed **L1A/L1C**, mission metadata, radiometric/geometric diagnostics, and Sentinel-2-assisted georeferencing workflows.

> **Documentation**
>
> - **Start here:** [`docs/overview.rst`](docs/overview.rst)
> - **Georeferencing guide:** [`docs/georeferencing.rst`](docs/georeferencing.rst)
> - **Installation:** [`docs/installation.rst`](docs/installation.rst)
> - **API quick reference:** [`docs/api_quick_reference.rst`](docs/api_quick_reference.rst)
> - **Notebook:** [`examples/Phiesta_Quickstart.ipynb`](examples/Phiesta_Quickstart.ipynb)

---

## What Phiesta does

Phiesta is a research-oriented Python toolkit for working with **ΦSat-2 products from raw L0 through processed L1A/L1C acquisitions**.

It provides a simple API to:

- open local ΦSat-2 L0, L1A, and L1C products;
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
git clone https://github.com/PhiSat-2/Phiesta.git
cd Phiesta
```

For the complete Phiesta workflow, including Sentinel-assisted georeferencing:

```bash
python -m pip install -e ".[triplets]"
```

That single installation command includes the base package and the optional matching dependencies. If you only need local product inspection and no Sentinel-assisted georeferencing, `python -m pip install -e .` is sufficient.

Quick import check:

```bash
python -c "import phiesta; from phiesta import L0_event, L1_event, L1A_event, connect_insula; print('Phiesta OK')"
python -c "from lightglue import LightGlue, SuperPoint, SIFT; print('LightGlue OK')"
```

See [`docs/installation.rst`](docs/installation.rst) for more details.

---

## Minimal georeferenced product from an Insula product ID

```python
from phiesta import connect_insula

PRODUCT_ID = "5359"

client = connect_insula()
event = client.load_l1(PRODUCT_ID)

product = event.georeference(
    sentinel_backend="download",
    # window_days=10,  # optional override; default search horizon is ±60 days
    verbose=True,
)

print(product["path"])
print(product["crs"])
print(product["resolution"])
```

`event.georeference(...)` runs the Sentinel-assisted registration and writes a
standard georeferenced GeoTIFF. The default Sentinel search horizon remains
±60 days; pass `window_days=10` (or another value) when a stricter temporal
constraint is wanted for a particular acquisition.

For access to the intermediate homographies, geographic footprint and matching
metrics without immediately exporting the final raster, use:

```python
georef = event.get_georef(window_days=10)
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

Phiesta's triplet workflow includes the Python orchestration needed to run the ΦSat-2 simulation step directly with the platform-specific simulator executable. No separate simulator-helper checkout or environment variable is required.

Authorized platform-specific simulator executables are versioned under:

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

## ΦSat-2 ecosystem

Phiesta is designed to complement the other public ΦSat-2 resources rather than replace them:

- **Portal-Access** documents how to request and access ΦSat-2 data.
- **Φ-down (`phidown`)** provides general search/download tooling, including ΦSat-2 support.
- **Data-Spec** defines dataset formatting used by ΦSatNet-related workflows.
- **Phiesta** focuses on mission-aware product loading, L0/L1 inspection, validation, cross-level comparison, diagnostics, and georeferencing.

---

## References and credits

Phiesta builds on public Earth-observation tools, data services, and research code:

- **ΦSat-2 mission:** ESA ΦSat-2 mission material.
- **Sentinel-2 access:** Copernicus Data Space Ecosystem catalogue and OData download API.
- **ΦSat-2 simulator:** Phiesta includes the Python orchestration used by the triplet workflow and calls the authorized platform-specific simulator executable; see `THIRD_PARTY_NOTICES.md` for provenance and redistribution terms.
- **Feature matching:** LightGlue for local feature matching.
- **Land-cover context, when used:** ESA WorldCover 2021 v200.

External services and bundled third-party components may have their own terms. The Apache-2.0 license applies to Phiesta itself; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for components that are not covered by that license.

---

## Status

Phiesta is an early research-oriented toolkit. APIs may evolve as the georeferencing workflow, strict alignment strategy, and documentation are improved.

## Product-level inspection API

Phiesta provides product-level utilities for opening, inspecting, screening, comparing, and mission-checking PhiSat-2 products.

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

### Mission-aware specification report

```python
import phiesta

event = phiesta.open_product("6008", level="L1C")

mission_report = phiesta.mission_spec_report(event)
band_table = phiesta.phisat2_band_table()
level_specs = phiesta.phisat2_product_level_specs()
```

The mission-aware report checks observed product metadata against encoded PhiSat-2 product expectations, including image shape, band count, georeferencing presence, CRS metadata, radiance/reflectance level, and expected band-alignment behaviour.

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
- mission-aware product specification checks;
- L1A/L1C comparison with inter-band shift diagnostics.

### Raw L0 inspection

Phiesta can open or download raw L0 product folders without requiring the external rawbin converter.

```python
import phiesta

raw_folder = phiesta.open_raw_l0_product("5090")
report = phiesta.raw_l0_report(raw_folder)
```

A raw L0 folder typically contains `raw.bin`, `metadata.json`, `ancillary.json`, `aocs.json`, and a thumbnail.

Building a prepared `L0_event` from `raw.bin` requires the external Simera/SENSE conversion code, configured through `PHIESTA_SIM_ROOT`. This external converter is not redistributed with Phiesta.

Use:

```python
raw_folder = phiesta.open_raw_l0_product("5090")
```

for public raw-folder access, and:

```python
l0 = phiesta.open_product("5090", level="L0")
```

only when the external converter is configured and a prepared `L0_event` is desired.

### Acquisition-level report

Phiesta can summarize all locally available product levels for the same acquisition.

```python
import phiesta

report = phiesta.acquisition_report("6008")
```

By default, `acquisition_report` is local-only and does not trigger new Insula downloads. It reports available and missing levels, raw L0 availability, L1A/L1C product cards, mission-spec checks, and an L1A/L1C comparison when both processed levels are available.

Use:

```python
report = phiesta.acquisition_report("5090", download_missing=False)
```

to inspect only already available local products, or:

```python
report = phiesta.acquisition_report("5090", download_missing=True)
```

to allow Insula fallback for missing levels.

## Data access behaviour

Phiesta uses a local-first strategy for PhiSat-2 products.

By default, product-opening helpers first look for an already available local product folder. If the requested product is missing locally, Phiesta can fall back to Insula when `download_missing=True`.

For public examples and reproducible local workflows, use `download_missing=False`:

```python
import phiesta

event = phiesta.open_product("6008", level="L1C", download_missing=False)
```

With `download_missing=False`, Phiesta never triggers remote authentication. Missing products raise `FileNotFoundError`.

To allow remote fallback through Insula, use:

```python
event = phiesta.open_product("6008", level="L1C", download_missing=True)
```

In that case, Phiesta tries local resolution first, then requests Insula credentials only if the product is not available locally.

Raw L0 access is separated from prepared L0 decoding:

```python
raw_folder = phiesta.open_raw_l0_product("5090", download_missing=True)
report = phiesta.raw_l0_report(raw_folder)
```

This opens or downloads the raw L0 product folder without decoding `raw.bin`.

A raw L0 folder typically contains `raw.bin`, `metadata.json`, `ancillary.json`, `aocs.json`, and a thumbnail.

Building a prepared `L0_event` from `raw.bin` requires the external Simera/SENSE conversion code, configured through `PHIESTA_SIM_ROOT`. This converter is not redistributed with Phiesta.

Summary:

- `open_product(..., download_missing=False)`: local-only, never asks for Insula credentials.
- `open_product(..., download_missing=True)`: local-first, then Insula fallback if missing.
- `open_raw_l0_product(..., download_missing=True)`: raw L0 access/download without requiring the external converter.
- `open_product(..., level="L0")`: builds a prepared `L0_event`; this requires the external converter when only raw L0 data is present.



---

## License

Phiesta is released under the **Apache License 2.0**. See [`LICENSE`](LICENSE).

Third-party source code and executable components are **not automatically relicensed** under Apache-2.0. Their provenance and licensing status are documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
