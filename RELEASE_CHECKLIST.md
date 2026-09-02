# Phiesta v0.1.0 public-release checklist

## Required before making the repository public

- [x] Remove the former vendored OrbitalAI helper package from the public release; keep provenance in third-party notices.
- [ ] Record the exact redistribution/license terms for `third_party/phisat2_exec/*.bin`.
- [ ] Confirm the intended copyright holder(s) for Phiesta-authored code (individual contributor(s), ESA, or another arrangement).
- [x] Run the CI workflow successfully on Python 3.10, 3.11, and 3.12.
- [ ] Test at least one local L0, one L1A, and one L1C product with `download_missing=False`.
- [ ] Test one authenticated Insula download using a non-public credential environment.
- [ ] Test one simulator workflow on each platform intended to be supported by the bundled executable assets.
- [ ] Verify no credentials, tokens, local absolute paths, or downloaded mission data are committed.

## GitHub repository settings

- [ ] Repository: `PhiSat-2/Phiesta`
- [ ] Description: `Mission-aware Python toolkit for ΦSat-2 product access, inspection, validation, comparison, and georeferencing.`
- [ ] Topics: `phisat-2`, `earth-observation`, `remote-sensing`, `satellite-imagery`, `multispectral`, `level-0`, `georeferencing`, `sentinel-2`, `python`, `esa`
- [ ] Enable Issues.
- [ ] Enable Discussions only if someone will actively monitor them.
- [ ] Protect `main`: require pull request + passing CI.
- [ ] Create release/tag `v0.1.0` after the checks above pass.
