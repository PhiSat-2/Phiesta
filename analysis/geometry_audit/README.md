# ΦSat-2 geometry audit

This folder defines the reproducible reference analysis for Phiesta's geometry
tooling.

## Scientific question

**Where does geometric error live in the ΦSat-2 processing chain?**

We separate three objects that are often conflated:

1. inter-band geometry inside one product;
2. inter-level geometry from L0 through L1A/L1C;
3. absolute geolocation relative to an external reference.

## Data-access and reproducibility policy

Raw L0 products are access-restricted and must not be assumed to be publicly
redistributable. Phiesta can support L0 analysis for authorized users, but the
public reference analysis and the paper's core reproducible claims should not
require redistribution of raw L0 imagery.

Accordingly:

- public/main claims should be reproducible from the product levels that can be
  legally shared or independently accessed by readers;
- L0 results are an additional authorized-access branch of the processing-chain
  audit, not a prerequisite for reproducing the main public figures;
- no raw L0 pixels, archives, or derived image products are committed to this
  repository;
- aggregate L0-derived measurements (for example displacement summaries) may be
  released only if their redistribution is permitted and they do not expose
  restricted source data;
- all L0 code paths must remain testable with synthetic fixtures so the public
  software does not depend on private mission data.

## Frozen first-pass questions

### RQ1 — Inter-band displacement by processing level

For paired acquisitions, estimate every non-master band relative to one fixed
master band at each available processing level. L1A/L1C form the public core
analysis; L0 is added for authorized-access acquisitions where available.

Primary outputs:

- `dx_px`, `dy_px`, `shift_px`;
- `corr_before`, `corr_after`, `corr_gain`;
- acquisition id and processing level.

Primary comparison: paired change in `shift_px` across publicly usable levels,
with a separate L0 extension for authorized-access data.

### RQ2 — Is residual geometry rigid?

For a pre-specified set of band pairs, estimate local displacement fields on a
fixed grid.

Primary outputs:

- median local shift;
- IQR / spatial standard deviation of local `dx` and `dy`;
- fraction of usable texture windows;
- global-vs-local residual disagreement.

A large spatial spread indicates that a global translation is insufficient.

### RQ3 — L0 → L1 reference-space transformation

For paired L0/L1 products, record:

- master L0→L1 shift;
- per-band residual shifts after master alignment;
- crop strategy / removed rows;
- residual correlation improvement.

### RQ4 — Internal geometry vs absolute geolocation

For the subset with successful Sentinel-assisted georeferencing, compare
internal geometry metrics with absolute correction magnitude and matching
quality.

This analysis is associative unless a causal intervention is explicitly added.

### RQ5 — Practical consequence

Only after RQ1–RQ4 are frozen, evaluate one downstream task where spatial
alignment matters. The task and metric must be chosen before inspecting its
result.

## Statistical unit

The acquisition is the default independent unit. Patches/windows are repeated
measurements within an acquisition and must not be treated as independent
samples.

Where multiple acquisitions share a pass or another dependence unit, inference
should be clustered at the strongest available grouping level.

## Reporting

Report distributions, paired differences, robust medians, and bootstrap
intervals. Avoid turning every band pair/window into an independent hypothesis.

## Provenance outputs

The analysis should write tidy tables rather than only figures:

- `interband_global.parquet`
- `interband_local.parquet`
- `l0_to_l1.parquet`
- `absolute_georef.parquet`
- `cohort.csv`

Every figure in the paper should be reproducible from these frozen tables.

## Scope discipline

Phiesta is the measurement/reproducibility artifact. The scientific paper is
not "we made a Python package"; it is the processing-chain geometry study that
the package makes possible.


## L1C primary probability sample

The first public population-scale analysis uses a pre-specified probability
sample from the full L1C catalogue. The default design draws 96 acquisitions
with a fixed seed (20260910), jointly stratified by six equal-width temporal
bins, three fixed equal-area latitude bands, and four longitude sectors.
Every occupied stratum receives at least one acquisition; remaining allocation
is proportional to stratum population. Sampling is random without replacement
within strata, and inclusion probabilities/design weights are written to the
sample manifest.

The L1C registration reference is RED / MS3 (band index 3), matching the mission
product specification. The encoded mission specification expects fine
band-to-band alignment in L1C, while L1A is expected to precede that fine
alignment. Thus L1C residuals are a processing-output quality measurement; the
future paired L1A→L1C analysis directly tests how much geometry changes across
the alignment stage once L1A download access is available.

The global diagnostic uses anti-aliased area downsampling, sub-pixel phase
correlation, and an explicit native-pixel plausibility bound. Out-of-range
estimates are rejected rather than clipped to the bound.

## Confirmatory L1C protocol v1

The six inspected L1C products used for method development are not part of the
confirmatory sample. The confirmatory global inter-band audit uses MS3/RED as
the reference band. The primary measurement is made on the native 4096-pixel
image; a 2048-pixel area-resampled measurement is retained as a pre-specified
scale-stability diagnostic. Reported displacements are always expressed in
native pixels.

The resolution change is a measurement stress test, not a change to the
scientific product: a genuine global translation should remain broadly
consistent after resampling once the estimate is mapped back to native pixels.
Scale disagreement is therefore recorded rather than hidden.

No response threshold is used to delete observations. Likewise, the historical
80-pixel plausibility value is only a diagnostic flag in the confirmatory run:
extreme numerical estimates are retained. Primary results use the complete
probability sample; response, correlation gain, and 4096-vs-2048 vector
disagreement are reported as measurement-quality diagnostics and sensitivity
analyses.

### Confirmatory development exclusion freeze

Before inspecting any confirmatory geometry outcomes, v1 freezes a development
exclusion manifest in `confirmatory_exclusions_v1.csv`. It contains the six
multiscale pilot products, the three earlier geometry-smoke products, and the
historical L1A/L1C geometry development selection. The confirmatory probability
sample is drawn only after applying this manifest. Metadata inspection alone is
not treated as an outcome inspection.

