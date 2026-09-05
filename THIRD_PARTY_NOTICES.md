# Third-party notices

Phiesta is licensed under the Apache License 2.0. That license applies to the Phiesta-authored source code in this repository; it does **not** automatically relicense third-party material.

## ESA Φ-lab / AI4EO OrbitalAI ΦSat-2 simulator

- **Upstream:** `https://github.com/AI4EO/orbitalAI/tree/main/phisat-2`
- **Context:** the upstream repository is the public starter toolkit released for the ESA Φ-lab / AI4EO OrbitalAI ΦSat-2 Challenge. The challenge explicitly included an Open Science evaluation component.
- **Purpose in Phiesta:** optional ΦSat-2 image simulation used by the Sentinel / simulated ΦSat-2 / real ΦSat-2 triplet workflow.
- **Distribution:** Phiesta does not vendor the former `orbitalai_phisat2_sim` helper package. The current orchestration is Phiesta-maintained code; simulator executables are handled separately below.
- **Attribution:** publications or derived workflows using the simulator should acknowledge the ESA Φ-lab / AI4EO OrbitalAI ΦSat-2 Challenge and link to the upstream repository.
- **Rights:** the upstream repository is public, but Phiesta does not claim that those upstream materials are licensed under Phiesta's Apache-2.0 license. Upstream materials retain their original rights and terms.

## LightGlue feature matching

- **Purpose:** local feature matching for proxy and strict georeferencing.
- **Installation:** the `triplets` extra installs the PyPI package `scm-lightglue`, imported as `scm_lightglue`, for the portable SIFT + LightGlue default. The original `cvg/LightGlue` package is an optional separately installed backend for explicit SuperPoint use.
- **Upstream:** LightGlue originates from `cvg/LightGlue`.
- **Licensing:** LightGlue code, weights, and feature extractors retain their own upstream licenses. SuperPoint, when installed separately through the original backend, is not covered by Phiesta's Apache-2.0 license; users should review the upstream terms for their use case.

## Platform-specific ΦSat-2 simulator executables

- **Location in Phiesta:** `third_party/phisat2_exec/`
- **Upstream provenance:** platform-specific ΦSat-2 simulator packages were publicly distributed with the ESA Φ-lab / AI4EO OrbitalAI ΦSat-2 starter materials.
- **Purpose:** simulator backend used by Phiesta's optional simulation workflow.
- **Packaging:** repository assets; intentionally not included in the Python wheel by the current package-data configuration.
- **Rights:** these executable assets are third-party material and are not relicensed under Phiesta's Apache-2.0 license.
- **Project record:** retain any written authorization, redistribution terms, or later upstream license notice with the project records and reflect it here if/when an explicit notice becomes available.

## Simera / SENSE raw L0 converter

- **Distribution:** not included with Phiesta.
- **Integration:** configured externally through `PHIESTA_SIM_ROOT`.
- **License:** governed independently by the provider of the converter.

## Python dependencies and external services

Phiesta depends on third-party Python packages and may access external data services. Those packages/services retain their own licenses and terms. Installing or using Phiesta does not change those terms.
