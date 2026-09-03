# Phiesta public API roadmap

## Current public API

### Product opening

    import phiesta

    event = phiesta.open_product("6008", level="L1C")

### Product inspection

    phiesta.product_card(event)
    phiesta.file_manifest(event)
    phiesta.file_family_summary(event)
    phiesta.processing_switches(event)
    phiesta.raster_inventory(event)
    phiesta.compare_product_folders(event_a, event_b)

### Product screening

    phiesta.quality_report(event)
    phiesta.quality_table(events)
    phiesta.product_gallery(product_ids)

### Georeferencing

```python
event = client.load_l1("5359")
product = event.georeference()
product.show_rgb()
```

`georeference()` is the primary high-level georeferencing API and returns a
georeferenced `L1_event`. Use `event.get_georef()` for advanced access to the
intermediate geometry.

## Near-term API priorities

### 1. Level comparison

Goal:

    phiesta.compare_levels(l1a, l1c)

Expected report:

- file-family differences;
- processing switch differences;
- raster inventory differences;
- geolocation / CRS differences;
- optional RC-vs-BC check;
- optional radiometry and geometry diagnostics.

### 2. Inter-band geometry diagnostics

Goal:

    phiesta.interband_shift_table(event, master_band=2)
    phiesta.local_interband_shift_field(event, master_band=2, target_band=6)
    phiesta.edge_overlay(event, band_a=2, band_b=6)

### 3. Radiometry diagnostics

Goal:

    phiesta.radiometry_summary(event_a, event_b)
    phiesta.radiometry_scatter(event_a, event_b)
    phiesta.radiometry_histograms(event_a, event_b)

### 4. Gallery and screening

Goal:

    phiesta.product_gallery(...)
    phiesta.quality_table(...)
    phiesta.select_good_candidates(...)

### 5. Sentinel-2 / triplet workflows

Goal:

    event.build_full_sentinel_triplet(...)
    triplet.summary()
    triplet.inspect()

## Publication framing

Phiesta should be framed as a product-level toolkit for PhiSat-2 imagery:

- access;
- inspection;
- validation;
- comparison;
- quality screening;
- cross-sensor contextualization.

The L1A/L1C audit is a case study demonstrating why product-level tooling is needed.
