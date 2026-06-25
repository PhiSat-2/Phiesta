# Phiesta

Phiesta is a lightweight Python toolkit for working with ΦSat-2 L0/L1 products.

It provides a user-friendly API to:

- search ΦSat-2 products from Insula;
- inspect product metadata, acquisition dates, and catalog footprints;
- download or reuse cached L0/L1 products;
- load local ΦSat-2 products;
- visualize bands, RGB composites, false-color composites, and registered displays;
- analyze raw band distributions and display stretches;
- extract cubes and spatial patches for research or machine learning;
- export search results as tables or GeoJSON;
- build Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplets;
- run triplet generation in batch mode.

The package is designed for research workflows around ΦSat-2, Earth observation, annotation transfer, and rapid exploration of satellite imagery.

---

## Recommended starting point

Open the quickstart notebook:

```text
examples/Phiesta_Quickstart.ipynb

It demonstrates the main API workflows step by step:

connection to Insula;
product search;
L1/L0 loading;
metadata inspection;
band visualization;
display registration;
distribution diagnostics;
cube and patch extraction;
bbox search;
GeoJSON export;
Sentinel triplet generation;
batch workflows.
Installation / environment

In the ESA / NEOHPC environment, Python is usually run inside the project container.

Typical usage:

cd /shared/home/mdepastor/projects/Phiesta
python -m compileall phiesta

If running manually inside the ESA container, use the configured project Python environment.

Quick start: connect to Insula and load one L1 product
from phiesta import connect_insula

client = connect_insula()
event = client.load_l1("5359")

event.show_event_info()

connect_insula() prompts for Insula credentials if no username/password is provided.

load_l1(...) downloads the product if needed, or reuses the local cache.

Search products before downloading

Search L1 products and get a compact table:

df = client.search_l1_table(
    page=0,
    results_per_page=20,
)

df.head()

The table includes product ID, filename, acquisition datetime, center coordinates, and footprint corners when available.

Typical columns:

product_id
filename
start_datetime
completion_datetime
center_lon
center_lat
corner_1_lon
corner_1_lat
corner_2_lon
corner_2_lat
corner_3_lon
corner_3_lat
corner_4_lon
corner_4_lat

Load one product from the table:

event = client.load_l1(df.iloc[0]["product_id"])
Search products by geographic bbox

Search products intersecting a lon/lat bounding box:

bbox = (88.0, 21.4, 90.2, 22.8)  # min_lon, min_lat, max_lon, max_lat

df = client.search_l1_bbox_table(
    bbox_lonlat=bbox,
    pages=40,
    results_per_page=100,
)

df.head()

This uses Insula catalog footprints. It is intended for discovery and approximate filtering, not precise pixel-level georeferencing.

Export search results to GeoJSON
client.export_search_table_geojson(
    df,
    "outputs/search_results.geojson",
)

The output can be opened in QGIS, geojson.io, or map-based notebooks.

Bulk load products from a search table
loaded = client.load_l1_table(
    df.head(3),
    continue_on_error=True,
)

for item in loaded:
    print(item["product_id"], item["status"], item["error"])

For L0 products:

loaded_l0 = client.load_l0_table(df.head(3))
Ways to open products
Remote L1 by acquisition ID
event = client.load_l1("5359")
Remote L1 by full product name
event = client.load_l1(
    "PHISAT-2_L1_000005359_20260512183354_20260512183357_236B0FFC"
)
Remote L0
l0_event = client.load_l0("5359")
Local L1 product
from phiesta import L1_event

event = L1_event.from_path(
    "data/l1/PHISAT-2_L1_000005359_20260512183354_20260512183357_236B0FFC"
)
Local L0 product
from phiesta import L0_event

l0_event = L0_event.from_path(
    "data/l0/PHISAT-2_L0_PRODUCT_FOLDER"
)
Product inspection
event.show_event_info()

This prints:

product ID;
filename;
sensing time;
product type;
raster shape;
dtype;
CRS and transform;
band aliases and wavelengths;
Insula catalog geometry if available;
local/remote paths;
common API calls.

With sampled band statistics:

event.show_event_info(show_stats=True)
Band access

Bands can be selected by:

integer index: 0, 1, 2, ...
wavelength: 842, 842.0;
alias: PAN, BLUE, GREEN, RED, RE1, RE2, RE3, NIR.
pan = event.get_band("PAN")
red = event.get_band("RED")
nir = event.get_band("NIR")
Visualize all bands
event.show_all_bands(
    normalization="percentile",
    percentiles=(1, 99),
    figsize=(16, 10),
)
Natural RGB
event.show_rgb(
    bands=("RED", "GREEN", "BLUE"),
    normalization="percentile",
    percentiles=(1, 99),
    per_band=True,
)

For raw ΦSat-2 DN values, percentile normalization per band is usually necessary for display.

Single-band display
event.show_band(
    "NIR",
    normalization="percentile",
    percentiles=(1, 99),
)
event.show_band(
    "PAN",
    normalization="percentile",
    percentiles=(1, 99),
)
False-color and vegetation composites

Vegetation and coastal scenes are often clearer in false color.

event.show_rgb(
    bands=("NIR", "RED", "GREEN"),
    normalization="percentile",
    percentiles=(1, 99),
    per_band=True,
)
event.show_rgb(
    bands=("NIR", "RE1", "RED"),
    normalization="percentile",
    percentiles=(1, 99),
    per_band=True,
)

In ("NIR", "RED", "GREEN"), vegetation appears red because the NIR band is displayed in the red channel.

Display registration between bands

Some ΦSat-2 bands may be slightly shifted relative to each other.

For visualization, Phiesta can register bands to a master band before building a composite:

event.show_rgb(
    bands=("NIR", "RED", "GREEN"),
    registered=True,
    registration_master="NIR",
    normalization="percentile",
    percentiles=(1, 99),
    per_band=True,
)

Display all bands after display registration:

event.show_all_bands(
    registered=True,
    registration_master="NIR",
    normalization="percentile",
    percentiles=(1, 99),
)

This helps visualization. It does not rewrite the original product on disk.

Display diagnostics
diag = event.plot_display_diagnostics(
    product_id="5359",
    percentiles=(1, 99),
    registered=True,
    registration_master="NIR",
)

diag["stats"]

This shows:

natural RGB;
false-color composite;
vegetation composite;
normalized NIR;
raw NDVI-like diagnostic;
raw NDVI-like distribution;
sampled band statistics.

Important: raw spectral indices computed from raw DN values should not be interpreted as physical reflectance indices unless the data have been calibrated into comparable units.

Compare display stretches
event.compare_display_stretches(
    bands=("NIR", "RED", "GREEN"),
    registered=True,
    registration_master="NIR",
)

This helps compare stretches such as:

(0.5, 99.5)
(1, 99)
(2, 98)
(5, 95)
Band statistics and distributions
stats = event.band_stats(
    bands=("PAN", "BLUE", "GREEN", "RED", "RE1", "RE2", "RE3", "NIR"),
    percentiles=(0, 1, 2, 5, 50, 95, 98, 99, 100),
    sample=1_000_000,
)
event.plot_distribution(
    bands=("BLUE", "GREEN", "RED", "RE1", "NIR"),
    bins=256,
    sample=1_000_000,
    log_y=True,
    percentiles=(1, 2, 5, 50, 95, 98, 99),
    hist_range_percentiles=(0.1, 99.9),
)
Work with cubes

Return the full product as a NumPy cube:

cube = event.to_cube()
print(cube.shape)  # (bands, height, width)

Return selected bands with channels last:

rgb_cube = event.to_cube(
    bands=("RED", "GREEN", "BLUE"),
    band_axis=-1,
)
print(rgb_cube.shape)  # (height, width, 3)
Extract spatial patches
patch = event.get_patch(
    x_min=1000,
    y_min=1000,
    width=512,
    height=512,
    bands=("RED", "GREEN", "BLUE"),
    band_axis=-1,
)

Extract one band:

nir_patch = event.get_patch(
    x_min=1000,
    y_min=1000,
    width=512,
    height=512,
    bands="NIR",
)
Display patches
event.show_patch(
    x_min=1000,
    y_min=1000,
    width=512,
    height=512,
    bands=("NIR", "RED", "GREEN"),
    registered=True,
    registration_master="NIR",
)

Display all bands of a patch:

event.show_patch(
    x_min=1000,
    y_min=1000,
    width=512,
    height=512,
    bands="all",
)
Normalize arrays or cubes
patch_norm = event.normalize(
    patch,
    method="percentile",
    percentiles=(1, 99),
    per_band=True,
    band_axis=-1,
)

This is useful for display and ML preprocessing experiments.

It is not physical radiometric calibration.

L0 / L1 registration

This is different from display registration between bands.

L0/L1 registration attempts to align a L0 product to the space of the corresponding L1 product.

from phiesta.utils.l0_l1_registration import register_l0_to_l1_space

l0_event = client.load_l0("5359")
l1_event = client.load_l1("5359")

l0_registered = register_l0_to_l1_space(
    l0_event=l0_event,
    l1_event=l1_event,
    master_band=7,
    max_shifts=(300, 300),
)
Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplets

Build a full triplet:

triplet = event.build_full_sentinel_triplet(
    buffer_km=20.0,
    proxy_target_size=(1024, 1024),
    final_margin_pct=0.15,
    verbose=True,
)

event.inspect_full_sentinel_triplet(triplet)
event.show_full_sentinel_triplet(triplet)

triplet["paths"]

The final triplet contains:

real ΦSat-2 L1;
Sentinel-2B warped to the real ΦSat-2 grid;
simulated ΦSat-2 warped to the real ΦSat-2 grid.
Batch triplet generation
from phiesta.triplets.batch import build_full_sentinel_triplets_batch

batch = build_full_sentinel_triplets_batch(
    client=client,
    product_ids=["5359", "5095"],
    output_root="data/triplets",
    verbose=True,
)

batch["rows"]

The batch helper writes per-product outputs plus CSV/JSON summaries.

Typical workflows
Search a region, export to GeoJSON, then load products
bbox = (88.0, 21.4, 90.2, 22.8)

df = client.search_l1_bbox_table(
    bbox_lonlat=bbox,
    pages=40,
    results_per_page=100,
)

client.export_search_table_geojson(
    df,
    "outputs/search_results.geojson",
)

events = client.load_l1_table(df.head(3))
Visual exploration
event.show_event_info()
event.show_all_bands()
event.show_rgb()
event.show_band("NIR")
event.plot_display_diagnostics()
Vegetation / coastal scenes
event.show_rgb(("NIR", "RED", "GREEN"), registered=True)
event.show_rgb(("NIR", "RE1", "RED"), registered=True)
event.compare_display_stretches(("NIR", "RED", "GREEN"))
Patch-based ML experiments
cube = event.to_cube()
patch = event.get_patch(1000, 1000, width=512, height=512, bands="all")
patch_norm = event.normalize(patch, percentiles=(1, 99), per_band=True)
Triplet generation
triplet = event.build_full_sentinel_triplet()
event.inspect_full_sentinel_triplet(triplet)
event.show_full_sentinel_triplet(triplet)
Repository structure
phiesta/
├── l0/           # L0 product loading and conversion
├── l1/           # L1 product loading, visualization, band access
├── remote/       # Insula auth, search, download, search tables, GeoJSON export
├── georef/       # catalog geometry and Sentinel georeferencing utilities
├── triplets/     # Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplet pipeline
└── utils/        # display, array ops, event info, registration, stats

third_party/
└── orbitalai_phisat2_sim/  # vendored ΦSat-2 simulation code
Current limitations
Insula catalog georeferencing is useful for search and screening, but not always precise enough for pixel-level overlay.
Raw spectral indices should not be interpreted as physical reflectance indices unless the data are calibrated.
Full triplet generation can be memory-intensive.
Strict multi-stage alignment modes are not yet exposed as public API.
Interactive map search is not implemented yet.
Future directions

Possible next features:

interactive map search: draw a rectangle and list/download ΦSat-2 products;
WorldCover class fractions for an acquisition or patch;
OpenStreetMap / Sentinel-2 / weather context for a scene;
patchify acquisitions into ML-ready datasets;
generic image/cube input support beyond ΦSat-2 product folders;
strict alignment modes for high-quality triplet generation.

---

## Build a patch index

Create a regular grid of pixel windows over an acquisition:

```python
patch_index = event.build_patch_index(
    patch_size=512,
    stride=512,
)

patch_index.head()
```

The patch index contains:

```text
patch_id
row
col
x_min
y_min
x_max
y_max
width
height
is_partial
```

Use overlapping patches:

```python
patch_index = event.build_patch_index(
    patch_size=512,
    stride=256,
)
```

Restrict patching to a sub-window:

```python
patch_index = event.build_patch_index(
    patch_size=512,
    stride=512,
    x_min=1000,
    y_min=1000,
    x_max=3000,
    y_max=3000,
)
```

---

## Iterate over patches

```python
for item in event.iter_patches(
    index=patch_index.head(3),
    bands=("NIR", "RED", "GREEN"),
    band_axis=-1,
):
    print(item["patch_id"], item["patch"].shape)
```

Each yielded item contains:

```text
patch_id
patch
window
row
```

You can also normalize patches while iterating:

```python
for item in event.iter_patches(
    index=patch_index.head(3),
    bands=("RED", "GREEN", "BLUE"),
    band_axis=-1,
    normalize=True,
    normalize_kwargs={
        "percentiles": (1, 99),
        "per_band": True,
        "band_axis": -1,
    },
):
    patch = item["patch"]
```

---

## Export patches for ML experiments

Export patches as `.npy` files:

```python
patch_table = event.export_patches(
    out_dir="outputs/patches/5359_rgb",
    patch_size=512,
    stride=512,
    bands=("RED", "GREEN", "BLUE"),
    band_axis=-1,
    normalize=True,
    normalize_kwargs={
        "percentiles": (1, 99),
        "per_band": True,
        "band_axis": -1,
    },
    prefix="rgb",
    overwrite=True,
)

patch_table.head()
```

`export_patches()` writes:

```text
outputs/patches/5359_rgb/rgb_r0000_c0000.npy
outputs/patches/5359_rgb/rgb_r0000_c0001.npy
...
outputs/patches/5359_rgb/rgb_index.csv
```

`patchify()` is an alias for `export_patches()`:

```python
patch_table = event.patchify(
    out_dir="outputs/patches/5359_all",
    patch_size=512,
    stride=512,
    bands="all",
)
```

This export is intended for simple research and ML workflows. It does not claim georeferenced GeoTIFF output.

