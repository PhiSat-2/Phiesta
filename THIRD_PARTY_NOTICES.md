# Third-party notices

Phiesta is licensed under the Apache License 2.0. That license applies to the Phiesta-authored source code in this repository; it does **not** automatically relicense third-party material.

## OrbitalAI ΦSat-2 simulator helpers

- **Upstream:** `https://github.com/AI4EO/orbitalAI/tree/main/phisat-2`
- **Purpose:** optional ΦSat-2 image-simulation workflow.
- **Distribution:** OrbitalAI helper source code is **not redistributed with Phiesta**.
- **Integration:** `phiesta.triplets.simulation` can load a compatible external `orbitalai_phisat2_sim` package. If it is not already importable, set `PHIESTA_ORBITALAI_ROOT` to its external location.
- **License:** any externally supplied OrbitalAI material remains subject to its own rights and terms and is not covered by Phiesta's Apache-2.0 license.

The upstream repository was publicly accessible when reviewed for the v0.1.0 release, but Phiesta does not rely on public GitHub availability as authorization to redistribute that source code.

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
