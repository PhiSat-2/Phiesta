# L1A/L1C audit analysis scripts

This folder contains one-off analysis scripts used for the PhiSat-2 L1A/L1C product audit.

These scripts are intentionally separated from the public Phiesta API.

## Status

The audit scripts are useful as provenance for figures, tables, and diagnostics, but they are not stable public APIs.

## Candidate functions to promote into Phiesta

### Product comparison

Candidate public module:

    phiesta.products.compare

Potential functions:

    compare_processing_configs(...)
    compare_raster_inventory(...)
    compare_level_pair(...)

### Inter-band geometry

Candidate public module:

    phiesta.geometry.interband

Potential functions:

    interband_shift_table(...)
    local_interband_shift_field(...)
    edge_overlay(...)
    plot_shift_map(...)

### Radiometry diagnostics

Candidate public module:

    phiesta.diagnostics.radiometry

Potential functions:

    radiometry_summary(...)
    radiometry_scatter(...)
    radiometry_histograms(...)
    compare_radiometry(...)
