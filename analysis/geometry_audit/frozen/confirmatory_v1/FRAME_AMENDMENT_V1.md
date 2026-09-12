# Confirmatory frame amendment v1

## Background

Product `5359` had been inspected during development of the high-level
georeferencing workflow and should therefore have been excluded from the
confirmatory L1C sampling frame.

It was inadvertently retained in the eligible frame used for the frozen
probability sample.

## Effect on the realized sample

`5359` was **not selected** among the 96 confirmatory acquisitions. Therefore,
none of the realized confirmatory image measurements comes from this
development product.

The affected stratum is `T5_A2_L0`.

Executed design:

- stratum population: `N_h = 248`
- stratum sample size: `n_h = 3`
- executed inclusion probability: `3 / 248`
- executed design weight: `248 / 3 = 82.6666666667`

The three realized sampled products in this stratum are:

- `6539`
- `6902`
- `7339`

## Corrected target population

Removing `5359` gives:

- corrected stratum population: `N_h = 247`
- stratum sample size remains: `n_h = 3`
- conditional inclusion probability: `3 / 247`
- analysis design weight: `247 / 3 = 82.3333333333`

Re-running the pre-specified allocation calculation after removing `5359`
changes no stratum allocation: `T5_A2_L0` remains allocated three units and
all other stratum allocations are unchanged.

No acquisition is removed, replaced, redrawn, or recomputed after observing
confirmatory outcomes. The frozen raw sample and raw measurement files remain
unchanged.

For target-population weighted analyses, `analysis_weights_v1.csv` records both
the originally executed weights and the corrected analysis weights.
