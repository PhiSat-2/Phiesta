# Phiesta

[![CI](https://github.com/PhiSat-2/Phiesta/actions/workflows/ci.yml/badge.svg)](https://github.com/PhiSat-2/Phiesta/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**Mission-aware Python tools for ΦSat-2 geometry, georeferencing, product inspection, and ML datasets.**

Phiesta works across **L0, L1A, and L1C** and is built around a simple idea:
a satellite product does not have one geometric error. Band-to-band alignment,
processing-level transformations, and absolute geolocation are distinct objects
and should be measurable separately.

## Install

```bash
git clone https://github.com/PhiSat-2/Phiesta.git
cd Phiesta
pip install -e ".[triplets,ml]"
```

Base inspection only:

```bash
pip install -e .
```

## 30-second start

```python
from phiesta import connect_insula

client = connect_insula()
event = client.load_l1("5359")

event.show_rgb()

product = event.georeference()
product.show_rgb()

print(product.meta["path"])
print(product.meta["crs"])
```

`georeference()` performs Sentinel-assisted registration, exports a standard
GeoTIFF, and returns a new `L1_event` backed by the corrected raster.

## Geometry across the processing chain

### Inter-band geometry

```python
from phiesta import interband_shift_table

shifts = interband_shift_table(
    event,
    master_band="RED",
)

print(shifts[[
    "target_band",
    "dx_px",
    "dy_px",
    "shift_px",
    "corr_before",
    "corr_after",
]])
```

For spatially varying residuals:

```python
from phiesta import local_interband_shift_field, plot_shift_map

field = local_interband_shift_field(
    event,
    master_band="RED",
    target_band="NIR",
    window_size=512,
    stride=256,
)

plot_shift_map(field)
```

A red/cyan edge overlay is also available:

```python
from phiesta import edge_overlay

overlay = edge_overlay(
    event,
    band_a="RED",
    band_b="NIR",
    align=False,
)
```

### Inter-level geometry

For paired raw and processed products:

```python
from phiesta import register_l0_to_l1

l0 = client.load_l0("5359")
l1 = client.load_l1("5359")

l0_in_l1 = register_l0_to_l1(
    l0,
    l1,
    master_band="NIR",
)

print(l0_in_l1.meta["registration_info"])
```

The registration record contains the master L0→L1 translation, per-band
residual shifts, and crop/reference-space metadata.

### Absolute geolocation

```python
corrected = l1.georeference()
```

This is deliberately separate from inter-band and inter-level geometry.

See **[Geometry diagnostics](docs/geometry.rst)** and
**[Georeferencing](docs/georeferencing.rst)**.

## Build ML datasets

Selection, construction, splitting, targets, and training adapters are separate
steps:

```python
selection = ["5359", "5360"]

dataset = client.build_l1_dataset(
    selection,
    out_dir="datasets/example",
    patch_size=512,
)

dataset.make_splits(
    train=0.8,
    val=0.1,
    test=0.1,
    seed=42,
)
```

Attach a label already present in the manifest:

```python
from phiesta import column_target

dataset.add_target(
    "class",
    column_target("label"),
)
```

Use it directly with PyTorch:

```python
loader = dataset.to_dataloader(
    split="train",
    targets="class",
    batch_size=16,
)
```

Phiesta also supports raster-aligned targets and WorldCover-based catalog
prefiltering.

See **[Dataset → PyTorch example](examples/dataset_training_quickstart.py)**.

## Product inspection

```python
event.show_event_info()
event.show_all_bands()
event.show_band("NIR")
event.show_rgb(bands=("NIR", "RED", "GREEN"))

stats = event.band_stats(
    bands=("BLUE", "GREEN", "RED", "NIR"),
)
```

Cross-level product metadata and processing differences can be inspected with:

```python
from phiesta import compare_levels

report = compare_levels(l1a, l1c)
```

## Advanced Sentinel / simulator workflow

Phiesta can build aligned:

```text
Sentinel-2
   ↕
simulated ΦSat-2
   ↕
real ΦSat-2
```

For most users, `event.georeference()` is the recommended entry point.
`build_full_sentinel_triplet()` and `get_georef()` expose the advanced
intermediate workflow.

The simulator provenance and third-party rights are documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Documentation

- [Overview](docs/overview.rst)
- [Geometry diagnostics](docs/geometry.rst)
- [Georeferencing](docs/georeferencing.rst)
- [Installation](docs/installation.rst)
- [API quick reference](docs/api_quick_reference.rst)
- [Main notebook](examples/Phiesta_Quickstart.ipynb)
- [Dataset training quickstart](examples/dataset_training_quickstart.py)

## Research reference analysis

The repository contains a publication-oriented geometry audit plan under
[`analysis/geometry_audit/`](analysis/geometry_audit/).

The intended scientific question is:

> **Where does geometric error live in the ΦSat-2 processing chain?**

Phiesta is the reproducibility artifact; the analysis is intended to study
inter-band, inter-level, and absolute geometry rather than merely describe the
software.

## Citation

If you use Phiesta in research, please cite the software and the relevant
ΦSat-2 mission/data publications. Citation metadata are provided in
[`CITATION.cff`](CITATION.cff).

## Contributing and contact

Issues and pull requests are welcome. For questions, use
[GitHub Issues](https://github.com/PhiSat-2/Phiesta/issues).

Maintainer: **Malo de Pastor**.

## License

Phiesta code is released under the Apache-2.0 license. Third-party simulator
assets retain their original rights; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
