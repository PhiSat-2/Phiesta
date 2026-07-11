# Phiesta TODO

## Alignment quality controls

The current full triplet pipeline produces pixel-aligned triplets using catalog geometry, proxy simulation, and LightGlue-based alignment.

Future improvements to consider:

- expose alignment quality parameters to users;
- save more explicit alignment metrics in the triplet report;
- add optional stricter matching mode for difficult acquisitions;
- add optional second refinement step when overlay quality is not sufficient;
- compute visual/quantitative quality scores beyond inlier ratio and valid fraction;
- document when the final CRS/transform should be considered grid metadata rather than precise geodetic metadata.

This is intentionally not implemented yet. The current priority is to keep the public API simple and stable.

## Source search and validation

The Sentinel-2 time window should be treated as a search horizon, not as a
binary validity threshold. The proper validation signal is the downstream
alignment quality: number of matches, number of inliers, inlier ratio,
reprojection residuals, coverage, correlation/SSIM/edge consistency, and visual
quicklooks.

Next improvements:

- evaluate several Sentinel-2 candidates per ΦSat-2 acquisition and rank by final alignment metrics;
- add a batch georeferencing benchmark over many acquisitions;
- export one CSV with product id, Sentinel-2 id, delta days, cloud, coverage, matches, inliers, inlier ratio, residual errors and quality label;
- keep the default API simple but expose source-search parameters clearly.
