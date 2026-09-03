# Phiesta product vision

Phiesta is a Python toolkit for opening, inspecting, validating, and contextualizing PhiSat-2 products.

The goal is not only to download files. The goal is to turn PhiSat-2 product folders into understandable Python objects with explicit metadata, processing, geometry, radiometry, and cross-sensor context.

## Target users

- PhiSat-2 engineers who need to inspect products quickly.
- Earth-observation researchers who need reliable product-level understanding.
- ML researchers who need clean patch datasets and cross-sensor pairs.
- Non-CS users who need simple APIs instead of manual TIFF/JSON parsing.

## Core capabilities

### Access

Open local or remote PhiSat-2 products through a unified Python interface.

### Inspect

Summarize product folders, metadata, bands, rasters, CRS, geolocation files, and processing switches.

### Validate

Report product completeness, geolocation status, raster metadata, inter-band alignment, and radiometric behaviour.

### Compare

Compare products from the same acquisition or different processing levels, such as L1A and L1C.

### Connect

Link PhiSat-2 products to Sentinel-2 context, simulated PhiSat-2 proxies,
alignment workflows, and corrected georeferenced products.

The high-level API keeps the same product abstraction before and after
georeferencing:

```python
event = client.load_l1("5359")
product = event.georeference()
product.show_rgb()
```

## Public API direction

Example usage:

    import phiesta

    event = phiesta.open_product("6008", level="L1C")
    card = phiesta.product_card(event)
    manifest = phiesta.file_manifest(event)
    switches = phiesta.processing_switches(event)

## Publication angle

Phiesta should be presented as a product-level toolkit for PhiSat-2, not as a one-off audit script.

Possible title:

    Phiesta: a Python toolkit for inspecting and validating PhiSat-2 imagery products

Main demos:

1. Load and inspect a PhiSat-2 product.
2. Compare two processing levels from the same acquisition.
3. Build Sentinel-2 / simulated PhiSat-2 / real PhiSat-2 triplets.
