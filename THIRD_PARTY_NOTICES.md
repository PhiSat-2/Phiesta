# Third-party notices

Phiesta is licensed under the Apache License 2.0. That license applies to the Phiesta-authored source code in this repository; it does **not** automatically relicense third-party material.

## OrbitalAI ΦSat-2 simulator helpers

- **Location in Phiesta:** `third_party/orbitalai_phisat2_sim/`
- **Upstream:** `https://github.com/AI4EO/orbitalAI/tree/main/phisat-2`
- **Purpose:** helper code derived from the public OrbitalAI / AI4EO ΦSat-2 simulator materials.
- **Local status:** adapted/vendored for Phiesta integration.
- **License status:** the upstream repository did not expose a root `LICENSE` file when checked on 2026-08-25. Public availability on GitHub alone does not establish an open-source redistribution license.

**Release action:** before publishing these vendored files, retain written permission or another authoritative statement establishing that redistribution/modification is permitted. If that authorization is not available, remove this vendored directory from the public release and make the simulator an external dependency instead.

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
