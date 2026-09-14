# Absolute geolocation protocol v2 amendment

This amendment was frozen after the three-product engineering smoke test and
before the confirmatory 96-acquisition absolute-geolocation batch.

The v1 smoke exposed implementation and proxy-localization limitations. No
successful absolute-geolocation outcome had been obtained.

Changes applied uniformly to every acquisition:

- Sentinel search horizon: +/- 60 days instead of +/- 7 days.
- Proxy simulation target: 2048 x 2048 instead of 1024 x 1024.

Rationale:

The Sentinel temporal window is a source-search horizon, not a geometric
validity criterion. Phiesta's public API uses 60 days by default and applies
temporal distance only as a weak source-selection prior.

The 1024 proxy produced only 3 and 4 LightGlue matches respectively on the
first two visually usable engineering-smoke acquisitions, before the final
strict georeferencing stage. The proxy resolution is therefore increased
uniformly rather than relaxed acquisition by acquisition.

Unchanged:

- maximum Sentinel cloud cover: 40%
- minimum coverage: 0.85
- source temporal weight: 0.05
- source cloud weight: 1.0
- simulation seed: 0
- final simulation target: 2048 x 2048
- SIFT + LightGlue
- all final strict matching / RANSAC thresholds
- failure retention, no replacement, no imputation

All v1 smoke outputs remain preserved as engineering provenance.
