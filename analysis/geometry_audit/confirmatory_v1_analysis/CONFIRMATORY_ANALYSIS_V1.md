# L1C confirmatory inter-band geometry analysis v1

## Frozen population and sample

The corrected target population contains 5,339 unique L1C products.
The confirmatory probability sample contains 96 acquisitions drawn from
71 occupied time-geography strata.

The sampling-frame amendment for development product 5359 affects only the
analysis weight of stratum T5_A2_L0. Product 5359 was not selected and no
confirmatory outcome was recomputed or replaced.

## Primary endpoint

The pre-specified primary measurement is global target-to-RED/MS3 translation
at max-side 4096. Max-side 2048 is a pre-specified scale-stability diagnostic.

No failed measurement is imputed.

Primary valid measurements:

- PAN: 91 / 96
- MS1: 96 / 96
- MS2: 96 / 96
- MS4: 96 / 96
- MS5: 96 / 96
- MS6: 96 / 96
- MS7: 96 / 96

The five missing PAN measurements arise from near-zero-variance registration
images.

Design-weighted median primary shift magnitudes are:

- PAN: 1.10 px
- MS1: 2.05 px
- MS2: 0.91 px
- MS4: 1.51 px
- MS5: 1.88 px
- MS6: 2.22 px
- MS7: 1.48 px

These estimates describe a tightly aligned central regime.

## Pre-specified 80-pixel diagnostic tail

At acquisition level, the design-weighted point estimates are:

- at least 1 band >80 px: 17.75%
- at least 2 bands >80 px: 10.01%
- at least 4 bands >80 px: 7.48%
- at least 6 bands >80 px: 7.48%
- all 7 bands >80 px: 4.14%

The >80-pixel event is a diagnostic flag, not by itself evidence of physical
L1C misregistration.

Many extreme estimates occur on acquisitions with very weak registration
response and severe disagreement between the 4096 and 2048 measurements.
Dark, low-texture, corrupted, or otherwise pathological acquisitions can make
global phase-correlation registration non-identifiable.

## Design-based uncertainty

Of 71 occupied strata, 58 have only one sampled acquisition. These strata
represent 46.28% of the corrected target population. Consequently, ordinary
within-stratum bootstrap or variance estimation is not appropriate.

For the >=1-band >80-pixel acquisition-level indicator:

- design-weighted estimate: 17.75%
- conservative 95% Hoeffding interval: 1.63% to 33.87%
- approximate Wald interval using a worst-case binary variance bound:
  6.14% to 29.35%
- Chebyshev interval using the same strict variance upper bound:
  0% to 44.23%

The Hoeffding interval is used as the preferred conservative finite-sample
uncertainty statement. The Wald result is supplementary and approximate.
The Chebyshev result is retained as an audit bound.

## Exploratory failure-mode analysis

Post-hoc analysis of the 23 sampled acquisitions with at least one primary
shift above 80 px identifies:

- 11 acquisition/estimator-failure-like cases
- 10 mixed or isolated-tail cases
- one cross-band coherent-offset candidate: product 4993
- one detector-geometry-like candidate: product 1517

These labels are exploratory and must not be interpreted as pre-specified
population classes.

Product 4993 contains a particularly coherent group across MS2, MS4, MS5 and
MS7, with displacement vectors near (+43, -85) px, sub-pixel-to-few-pixel
cross-scale disagreement, and positive correlation gains.

Product 1517 is an exceptional long-strip product. Several bands have
along-track offsets partially compatible with the known focal-plane detector
line geometry, most strikingly MS6, while other bands are inconsistent or
unstable. It is treated as a mechanistic case study rather than evidence for a
population-wide failure mode.

## Interpretation

The confirmatory result does not support a model in which ordinary PhiSat-2
L1C products are diffusely misregistered by hundreds of pixels.

Instead, the data show:

1. a dominant central regime with residual inter-band shifts of roughly
   1--2 pixels;
2. a heavy diagnostic tail concentrated at the acquisition level;
3. many extreme-tail measurements that are non-identifiable or unstable on
   pathological acquisitions; and
4. a small number of structured anomalies worthy of separate mechanistic
   investigation.

Extreme global registration estimates should therefore not be interpreted as
physical displacement without supporting response, correlation, cross-scale,
and cross-band evidence.
