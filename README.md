# PyRawPh https://malodept.github.io/PyRawPh/

PyRawPh is a lightweight Python package for loading, exploring, tiling, visualizing,
and exporting local ΦSat-2 L1 products.

It provides a practical high-level interface for:
- loading local multiband or per-band ΦSat-2 GeoTIFF products,
- accessing spectral bands by index, wavelength, or alias,
- building quick RGB composites and normalized-difference products,
- splitting scenes into tiles,
- plotting scene or tile location,
- exporting processed outputs to GeoTIFF.

## Main features

- **Simple scene loading** with `L1_event.from_path(...)`
- **Flexible band selection** by:
  - integer index, for example `3`
  - wavelength in nanometers, for example `842.0`
  - string selectors such as `"B3"` or `"842nm"`
  - aliases such as `"BLUE"`, `"GREEN"`, `"RED"`, `"RE1"`, `"RE2"`, `"RE3"`, and `"NIR"`
- **Quick processing utilities**
  - RGB composite with `rgb()`
  - NDVI / NDWI with `index()`
- **Tiling utilities**
  - regular grid generation with `to_tiles()` / `make_tiles()`
  - per-tile inspection with `L1_tile`
- **Geospatial helpers**
  - event and tile bounds
  - optional footprint plotting from `GL_scene_<id>.json`
- **GeoTIFF export**
  - export the full event or any derived array with `export_to_tif()`

## Core objects

### `L1_event`

`L1_event` is the main high-level object in PyRawPh.

It represents a full local ΦSat-2 L1 scene and provides methods for:
- loading a product from disk,
- accessing spectral bands,
- computing quick spectral products,
- cropping the scene in pixel coordinates,
- splitting the scene into tiles,
- plotting the scene location,
- exporting arrays to GeoTIFF.

### `L1_tile`

`L1_tile` represents a local tile extracted from an event.

A tile stores:
- a local multiband array,
- its associated metadata,
- helper methods for visualization and geographic inspection.

## Expected product layout

PyRawPh is designed for local ΦSat-2 product folders with a structure similar to:

```text
<product_folder>/
├── bands/
│   ├── scene_0_BC_multiband.tiff
│   ├── scene_0_BC_band_0.tiff
│   ├── scene_0_BC_band_1.tiff
│   └── ...
├── geolocation/
│   └── GL_scene_0.json
├── processing_config.json
├── session_XXXX_metadata.json
└── logs/
    └── session_XXXX_metadata.json
```

The package can read either:
- a single multiband GeoTIFF, or
- one GeoTIFF per band.

When available, sidecar files such as `GL_scene_<id>.json`,
`processing_config.json`, and session metadata files are used to enrich the
scene metadata.

## Quick start

### Load a product

```python
from pyrawph.l1.l1_event import L1_event

ev = L1_event.from_path(
    product_folder="path/to/product_folder",
    scene_id=0,
    product_kind="BC",
)
```

### Inspect the event

```python
ev.show_event_info()
```

### Access one band

```python
nir = ev.get_band("NIR")
red = ev.get_band("RED")
```

### Build an RGB image

```python
rgb = ev.rgb()
```

### Compute NDVI

```python
ndvi = ev.index("NDVI")
```

### Create tiles

```python
tiles = ev.to_tiles(tile_size=512, overlap=0)
print(ev.get_tiles_names())
```

### Inspect one tile

```python
tile = ev.get_tile(0)
tile.show_bands()
```

### Export a derived product

```python
ev.export_to_tif(
    out_path="outputs/ndvi.tif",
    arr=ndvi,
    meta=ev.get_meta(),
)
```

## Band selection

Band selection is intentionally flexible.

Supported selectors include:
- integer indices, for example `0` or `3`
- wavelengths in nanometers, for example `842.0`
- string selectors such as `"3"`, `"B3"`, `"BAND_3"`, or `"842nm"`
- aliases such as `"BLUE"`, `"GREEN"`, `"RED"`, `"RE1"`, `"RE2"`, `"RE3"`, and `"NIR"`

Examples:

```python
ev.get_band(3)
ev.get_band(842.0)
ev.get_band("B3")
ev.get_band("842nm")
ev.get_band("NIR")
```

## Scene and tile metadata

Both `L1_event` and `L1_tile` use a metadata dictionary named `meta`.

Depending on the available files, it may contain fields such as:
- `path`
- `scene_id`
- `product_kind`
- `count`
- `width`
- `height`
- `dtype`
- `crs`
- `transform`
- `bounds`
- `band_wavelength_nm`
- `gl_path`
- `processing_config_path`
- `session_metadata_path`

This metadata is used by:
- band resolution
- crop and tile georeferencing
- plotting
- GeoTIFF export



## Full Sentinel-2 / PhiSat-2 triplet generation

```python
from pyrawph import connect_insula

client = connect_insula()
event = client.load_l1("5359")

triplet = event.build_full_sentinel_triplet()
event.inspect_full_sentinel_triplet(triplet)
event.show_full_sentinel_triplet(triplet)
```

The resulting triplet contains:

real PhiSat-2 L1 image, 8 bands, aligned grid;
Sentinel-2B crop warped to the PhiSat-2 grid;
simulated PhiSat-2 image generated from Sentinel-2B and warped to the PhiSat-2 grid.

Output paths are available with:

triplet["paths"]



## Typical workflow

A common workflow in PyRawPh is:

1. Load a local ΦSat-2 L1 product with `L1_event.from_path(...)`
2. Inspect the scene with `show_event_info()`
3. Access one or more bands with `get_band(...)`
4. Build a quick RGB composite with `rgb()`
5. Compute a normalized-difference product such as NDVI with `index(...)`
6. Split the scene into tiles with `to_tiles(...)`
7. Inspect tiles with `get_tile(...)` and `show_bands()`
8. Export a processed output with `export_to_tif(...)`

## Documentation

If the Sphinx documentation is built locally, open:

```text
docs/build/html/index.html
```

The documentation includes:
- a quickstart guide
- core concepts
- usage examples
- API reference for `L1_event`, `L1_tile`, and utility modules

## Notes

- PyRawPh focuses on **local** ΦSat-2 L1 products.
- Arrays are handled in **channel-first** format `(C, H, W)` for core event and tile data.
- GeoTIFF export requires geospatial metadata, especially `crs` and `transform`.


