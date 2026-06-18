# PyRawPh-Light

PyRawPh-Light is a lightweight Python toolkit for working with ΦSat-2 products.

It provides utilities to:

* load local or remote ΦSat-2 L0/L1 products;
* inspect and visualize multispectral ΦSat-2 scenes;
* access spectral bands by index, wavelength, or semantic alias;
* build RGB composites and simple spectral indices;
* register L0 and L1 products;
* retrieve ΦSat-2 acquisitions from the Insula platform;
* generate pixel-aligned Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplets.

The package is designed primarily for research workflows around ΦSat-2 data, Earth observation, and annotation transfer.

---

## Main features

### ΦSat-2 product handling

PyRawPh-Light supports both L0 and L1 products.

Typical operations include:

* loading a ΦSat-2 product from a local folder;
* loading a ΦSat-2 acquisition directly from Insula;
* accessing bands by index, wavelength, or alias;
* displaying RGB composites;
* exporting arrays to GeoTIFF;
* inspecting scene metadata.

### Flexible band access

Bands can be selected using:

* integer indices, for example `0`, `1`, `3`;
* wavelengths in nanometers, for example `842.0`;
* string selectors such as `"B3"`, `"BAND_3"`, or `"842nm"`;
* semantic aliases such as `"PAN"`, `"BLUE"`, `"GREEN"`, `"RED"`, `"RE1"`, `"RE2"`, `"RE3"`, and `"NIR"`.

Example:

```python
red = event.get_band("RED")
nir = event.get_band("NIR")
pan = event.get_band("PAN")
```

### Full Sentinel-2 / ΦSat-2 triplet generation

The main high-level feature is full triplet generation.

For one ΦSat-2 L1 acquisition, PyRawPh-Light can automatically build:

1. the real ΦSat-2 image;
2. a Sentinel-2B crop warped to the ΦSat-2 grid;
3. a simulated ΦSat-2 image generated from Sentinel-2B and warped to the real ΦSat-2 grid.

The final outputs are pixel-aligned rasters, typically with shape `4096 × 4096`.

---

## Installation

For development usage:

```bash
git clone https://github.com/malodept/PyRawPh.git
cd PyRawPh
python -m pip install -e .
```

For the full triplet pipeline, additional scientific and computer vision dependencies are required, including the packages used by the vendored ΦSat-2 simulator and the LightGlue-based alignment pipeline.

In the ESA / NEOHPC environment, install inside the project container, for example:

```bash
apptainer exec --nv \
  --bind /shared/home:/shared/home,/shared/projects:/shared/projects,/eodata:/eodata \
  --pwd /shared/home/mdepastor/projects/PyRawPh-Light \
  /shared/projects/phisat2/containers/phisat2.sif \
  python -m pip install --user -e .
```

---

## Quick start: load a ΦSat-2 L1 product from Insula

```python
from pyrawph import connect_insula

client = connect_insula()
event = client.load_l1("5359")
```

`connect_insula()` prompts for Insula credentials if no username or password is provided.

---

## Quick start: build a full triplet

```python
from pyrawph import connect_insula

client = connect_insula()
event = client.load_l1("5359")

triplet = event.build_full_sentinel_triplet()

event.inspect_full_sentinel_triplet(triplet)
event.show_full_sentinel_triplet(triplet)

print(triplet["paths"])
```

The resulting `triplet["paths"]` dictionary contains paths to:

```text
real       -> real ΦSat-2 image on the final grid
sentinel   -> Sentinel-2B warped to the ΦSat-2 grid
simulated  -> simulated ΦSat-2 warped to the real ΦSat-2 grid
metadata   -> metadata for the final triplet
report     -> full JSON report for the run
```

---

## Triplet output format

A full triplet contains three main GeoTIFF files.

### Real ΦSat-2

```text
8 bands, 4096 × 4096
band order:
PAN, BLUE, GREEN, RED, RED_EDGE_1, RED_EDGE_2, RED_EDGE_3, NIR
```

### Sentinel-2B warped to ΦSat-2 grid

```text
7 bands, 4096 × 4096
band order:
BLUE, GREEN, RED, NIR_BROAD, RED_EDGE_1, RED_EDGE_2, RED_EDGE_3
```

### Simulated ΦSat-2 warped to real ΦSat-2 grid

```text
8 bands, 4096 × 4096
band order:
BLUE, GREEN, RED, PAN, NIR, RED_EDGE_1, RED_EDGE_2, RED_EDGE_3
```

The final rasters are designed for pixel-aligned machine learning workflows. The main guarantee is pixel alignment between the real ΦSat-2, Sentinel-2B, and simulated ΦSat-2 outputs.

---

## Full triplet pipeline

Internally, the full triplet pipeline performs the following steps:

1. load a real ΦSat-2 L1 acquisition;
2. search for a suitable Sentinel-2B source around the ΦSat-2 acquisition date;
3. create a large Sentinel-2B crop around the ΦSat-2 catalog footprint;
4. create a low-resolution proxy simulation;
5. rectify the proxy simulation using catalog geometry;
6. align the proxy simulation to the real ΦSat-2 image with LightGlue;
7. infer the final Sentinel-2 native crop window;
8. crop Sentinel-2 to the final window;
9. simulate ΦSat-2 at native resolution from that final Sentinel-2 crop;
10. warp Sentinel-2 and simulated ΦSat-2 to the real ΦSat-2 grid.

Most users should not call these internal steps directly. Use:

```python
triplet = event.build_full_sentinel_triplet()
```

---

## Useful triplet parameters

```python
triplet = event.build_full_sentinel_triplet(
    buffer_km=20.0,
    proxy_target_size=(1024, 1024),
    final_margin_pct=0.15,
    window_days=15,
    max_cloud_cover=20.0,
    verbose=True,
)
```

Recommended defaults:

* `buffer_km=20.0`: initial Sentinel-2 search/crop margin;
* `proxy_target_size=(1024, 1024)`: lightweight proxy simulation size for alignment;
* `final_margin_pct=0.15`: safety margin around the final Sentinel-2 crop;
* `window_days=15`: Sentinel search window around the ΦSat-2 acquisition;
* `max_cloud_cover=20.0`: maximum Sentinel cloud cover.

---

## Batch triplet generation

Several acquisitions can be processed with the batch helper:

```python
from pyrawph import connect_insula
from pyrawph.triplets.batch import build_full_sentinel_triplets_batch

client = connect_insula()

batch = build_full_sentinel_triplets_batch(
    client=client,
    product_ids=["5359", "5095"],
    output_root="data/triplets",
)

print(batch["summary_csv"])
print(batch["rows"])
```

The batch helper writes:

* one triplet folder per acquisition;
* a CSV summary;
* a JSON summary;
* a full JSON report with detailed metadata and errors.

---

## Visualizing a triplet

```python
event.show_full_sentinel_triplet(
    triplet,
    save_dir="data/triplets/5359/quicklooks",
)
```

This displays:

* real ΦSat-2 RGB;
* Sentinel-2B warped RGB;
* simulated ΦSat-2 warped RGB;
* PAN overlay with real ΦSat-2 in red and simulated ΦSat-2 in green.

---

## Local L1 product loading

Local products can still be loaded directly:

```python
from pyrawph import L1_event

event = L1_event.from_path(
    product_folder="path/to/PHISAT-2_L1_product",
    scene_id=0,
    product_kind="BC",
)
```

Then:

```python
event.show_event_info()
rgb = event.rgb()
ndvi = event.index("NDVI")
```

---

## Local L0 product loading

```python
from pyrawph import L0_event

event = L0_event.from_path("path/to/PHISAT-2_L0_product")
event.show_event_info()
```

L0 utilities include product loading, raw binary conversion, visualization, and registration helpers for L0/L1 workflows.

---

## Repository structure

```text
pyrawph/
├── l0/           # L0 product loading and conversion
├── l1/           # L1 product loading, visualization, band access
├── remote/       # Insula authentication, download, local cache resolution
├── georef/       # catalog geometry and Sentinel georeferencing utilities
├── triplets/     # Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplet pipeline
└── utils/        # display, export, statistics, L0/L1 registration helpers

third_party/
└── orbitalai_phisat2_sim/  # vendored ΦSat-2 simulation code
```

---

## Notes and limitations

* The full triplet pipeline currently targets research workflows and has mostly been tested in the ESA / NEOHPC environment.
* The final triplet outputs are pixel-aligned rasters intended for machine learning and annotation transfer.
* The CRS/transform metadata of final warped rasters should be treated as grid metadata, not as a guarantee of high-precision geodetic accuracy.
* The simulation step relies on vendored ΦSat-2 simulation code and a bundled ΦSat-2 executable.
* The full triplet pipeline uses LightGlue-based matching and requires GPU support for practical runtime.
* For large or difficult acquisitions, memory usage can become significant during native simulation.

---

## Example script

A complete example is available in:

```text
examples/full_sentinel_triplet_demo.py
```

Run it with:

```bash
python examples/full_sentinel_triplet_demo.py
```

---

## Development checks

Compile the package:

```bash
python -m compileall pyrawph
```

Run the full triplet example:

```bash
python examples/full_sentinel_triplet_demo.py
```

Check git status before committing:

```bash
git status
```
