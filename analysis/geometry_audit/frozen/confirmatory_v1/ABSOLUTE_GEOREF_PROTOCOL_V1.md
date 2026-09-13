# Absolute geolocation confirmatory protocol v1

## Population

Use the same 96 acquisitions from the frozen L1C confirmatory probability
sample. No acquisition is replaced.

Primary reporting uses all 96 acquisitions, including technical or matching
failures.

A pre-georeferencing secondary analysis uses the 68 acquisitions labelled U
in acquisition_qc_v1/acquisition_qc_frozen_v1.csv.

## Reference and pipeline

Reference: Sentinel-2 selected automatically by Phiesta.

Frozen source-selection settings:

- Sentinel search horizon: +/- 7 days
- maximum Sentinel cloud cover: 40%
- minimum spatial coverage: 0.85
- no manual Sentinel scene selection

Frozen simulation settings:

- final simulated PhiSat-2 target size: 2048 x 2048
- simulation seed: 0

Frozen strict refinement:

- source: simulated PhiSat-2
- real matching band: PAN
- source matching band: PAN
- features: SIFT + LightGlue
- maximum keypoints: 12000
- matching max-side: 2600
- RANSAC threshold: 3 px
- minimum matches: 80
- minimum inliers: 150
- minimum inlier ratio: 0.12
- clouds masked
- water masked
- low-texture regions masked
- texture percentile: 20
- refinement rounds: 2

## Outcomes

For every acquisition retain:

- success/failure and failure reason
- selected Sentinel satellite/date offset/cloud/coverage
- proxy matches, inliers and inlier ratio
- strict matches, inliers and inlier ratio
- strict reprojection median, p90 and p95 errors
- corrected center and corners
- displacement of corrected center from nominal catalog center
- complete georeference JSON

The displacement between nominal catalog geometry and the Sentinel-refined
geometry is the absolute-geolocation correction estimate.

LightGlue/RANSAC reprojection error is a fit diagnostic and must not be
reported as absolute geolocation error.

## Failure handling

No acquisition is replaced and no failed result is imputed.

Transient network/authentication failures may be retried with identical
parameters.

No acquisition-specific tuning, alternate matching parameters, manual
Sentinel selection or expanded temporal search is permitted in the primary
analysis after outcomes are observed.

Any later alternative configuration is explicitly secondary/sensitivity
analysis.
