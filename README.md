# PyRawPh-Light

PyRawPh-Light is a lightweight Python toolkit for working with ΦSat-2 products.

It provides utilities to:

- load local or remote ΦSat-2 L0/L1 products;
- inspect and visualize multispectral ΦSat-2 scenes;
- access spectral bands by index, wavelength, or semantic alias;
- build RGB composites and simple spectral indices;
- register L0 and L1 products;
- retrieve ΦSat-2 acquisitions from the Insula platform;
- generate pixel-aligned Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplets.

The package is designed primarily for research workflows around ΦSat-2 data, Earth observation, and annotation transfer.

---

## Recommended notebook

The easiest way to start is to open:

```text
examples/PyRawPh_Quickstart.ipynb
```

It covers the main workflows:

- connecting to Insula;
- loading a ΦSat-2 L1 acquisition;
- visualizing bands and RGB composites;
- building a full Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplet;
- inspecting and visualizing the final triplet;
- running batch triplet generation.

Most users should start from this notebook rather than from the internal API.

---

## Main high-level workflow

```python
from pyrawph import connect_insula

client = connect_insula()
event = client.load_l1("5359")

triplet = event.build_full_sentinel_triplet()

event.inspect_full_sentinel_triplet(triplet)
event.show_full_sentinel_triplet(triplet)

print(triplet["paths"])
```

The final triplet contains:

- the real ΦSat-2 L1 image;
- a Sentinel-2B crop warped to the ΦSat-2 grid;
- a simulated ΦSat-2 image generated from Sentinel-2B and warped to the real ΦSat-2 grid.

---

## Triplet output format

A full triplet contains three main GeoTIFF files.

### Real ΦSat-2

```text
8 bands, 4096 x 4096
band order:
PAN, BLUE, GREEN, RED, RED_EDGE_1, RED_EDGE_2, RED_EDGE_3, NIR
```

### Sentinel-2B warped to ΦSat-2 grid

```text
7 bands, 4096 x 4096
band order:
BLUE, GREEN, RED, NIR_BROAD, RED_EDGE_1, RED_EDGE_2, RED_EDGE_3
```

### Simulated ΦSat-2 warped to real ΦSat-2 grid

```text
8 bands, 4096 x 4096
band order:
BLUE, GREEN, RED, PAN, NIR, RED_EDGE_1, RED_EDGE_2, RED_EDGE_3
```

The final rasters are designed for pixel-aligned machine learning workflows. The main guarantee is pixel alignment between the real ΦSat-2, Sentinel-2B, and simulated ΦSat-2 outputs.

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

- `buffer_km=20.0`: initial Sentinel-2 search/crop margin;
- `proxy_target_size=(1024, 1024)`: lightweight proxy simulation size for alignment;
- `final_margin_pct=0.15`: safety margin around the final Sentinel-2 crop;
- `window_days=15`: Sentinel search window around the ΦSat-2 acquisition;
- `max_cloud_cover=20.0`: maximum Sentinel cloud cover.

---

## Batch triplet generation

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

- one triplet folder per acquisition;
- a CSV summary;
- a JSON summary;
- a full JSON report with detailed metadata and errors.

---

## Load a ΦSat-2 L1 product from Insula

```python
from pyrawph import connect_insula

client = connect_insula()
event = client.load_l1("5359")
```

`connect_insula()` prompts for Insula credentials if no username or password is provided.

---

## Load a local L1 product

```python
from pyrawph import L1_event

event = L1_event.from_path(
    product_folder="path/to/PHISAT-2_L1_product",
    scene_id=0,
    product_kind="BC",
)

event.show_event_info()
event.show_rgb()
```

---

## Band access

Bands can be selected by index, wavelength, or semantic alias.

```python
pan = event.get_band("PAN")
red = event.get_band("RED")
nir = event.get_band("NIR")
ndvi = event.index("NDVI")
```

Common aliases include:

```text
PAN, BLUE, GREEN, RED, RE1, RE2, RE3, NIR
```

---

## Visualization

```python
event.show_rgb()

event.show_full_sentinel_triplet(
    triplet,
    save_dir="data/triplets/5359/quicklooks",
)
```

The full triplet visualization displays:

- real ΦSat-2 RGB;
- Sentinel-2B warped RGB;
- simulated ΦSat-2 warped RGB;
- PAN overlay with real ΦSat-2 in red and simulated ΦSat-2 in green.

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

- The full triplet pipeline currently targets research workflows and has mostly been tested in the ESA / NEOHPC environment.
- The final triplet outputs are pixel-aligned rasters intended for machine learning and annotation transfer.
- The CRS/transform metadata of final warped rasters should be treated as grid metadata, not as a guarantee of high-precision geodetic accuracy.
- The simulation step relies on vendored ΦSat-2 simulation code and a bundled ΦSat-2 executable.
- The full triplet pipeline uses LightGlue-based matching and requires GPU support for practical runtime.
- For large or difficult acquisitions, memory usage can become significant during native simulation.

---

## Development checks

```bash
python -m compileall pyrawph
python examples/full_sentinel_triplet_demo.py
```
