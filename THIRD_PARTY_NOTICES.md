# Third-party notices

Phiesta is licensed under the Apache License 2.0. That license applies to the Phiesta-authored source code in this repository; it does **not** automatically relicense third-party material.

## OrbitalAI ΦSat-2 simulator helpers

- **Upstream:** `https://github.com/AI4EO/orbitalAI/tree/main/phisat-2`
- **Purpose:** optional ΦSat-2 image-simulation workflow.
- **Upstream role:** the public OrbitalAI ΦSat-2 materials are retained here as provenance for the simulation workflow.
- **Distribution:** Phiesta does not vendor the former `orbitalai_phisat2_sim` helper package. The current triplet pipeline uses Phiesta-maintained orchestration code and the separately governed platform-specific simulator executable described below.
- **License:** upstream OrbitalAI materials and simulator executables retain their own rights and terms and are not relicensed by Phiesta's Apache-2.0 license.

## LightGlue / SuperPoint

- **Purpose:** local feature matching for proxy and strict georeferencing.
- **Installation:** the `triplets` extra installs the PyPI package `scm-lightglue` so users do not need a second manual Git installation step. The package exposes the `lightglue` Python module used by Phiesta.
- **Upstream:** LightGlue originates from `cvg/LightGlue`.
- **Licensing:** LightGlue code and weights and the feature extractors it exposes retain their own upstream licenses. In particular, SuperPoint is not covered by Phiesta's Apache-2.0 license; users should review the upstream terms for their use case.

## Platform-specific ΦSat-2 simulator executables

- **Location in Phiesta:** `third_party/phisat2_exec/`
- **Purpose:** platform-specific simulator backend used by Phiesta's simulation workflow.
- **Packaging:** repository assets; intentionally not included in the Python wheel by the current package-data configuration.
- **License:** not covered by Phiesta's Apache-2.0 license unless the rights holder explicitly says otherwise.

**Release action:** keep the authorization/redistribution terms for these binaries alongside the project records and, if available, add the exact license or notice here.

## Simera / SENSE raw L0 converter

- **Distribution:** not included with Phiesta.
- **Integration:** configured externally through `PHIESTA_SIM_ROOT`.
- **License:** governed independently by the provider of the converter.

## Python dependencies and external services

Phiesta depends on third-party Python packages and may access external data services. Those packages/services retain their own licenses and terms. Installing or using Phiesta does not change those terms.
