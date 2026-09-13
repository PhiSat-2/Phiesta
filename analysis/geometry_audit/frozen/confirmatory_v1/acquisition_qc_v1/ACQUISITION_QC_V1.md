# Acquisition quality stratification v1

This file freezes acquisition-level visual quality labels before any
confirmatory absolute-geolocation batch was run.

Sample:
- 96 frozen confirmatory L1C acquisitions
- 68 U: visually usable acquisition
- 28 F: obvious acquisition/sensor corruption relevant to georeferencing

The visual review used RGB, PAN and NIR contact sheets.

F denotes gross acquisition/sensor pathology such as blank or nearly blank
bands, severe striping/grid artefacts, gross corruption, or otherwise
non-usable sensor imagery.

Cloud, water, snow/ice, desert, darkness, or ordinary low-texture content were
not by themselves reasons for an F label.

These labels do not modify the all-96 primary analysis. They define a
secondary georeferencing analysis conditional on visually usable acquisition.

The labels were frozen before absolute-georeferencing outcomes were computed.
They should not be described as strictly blinded or independent of the earlier
inter-band analysis: inter-band results already existed and QC-code/product-ID
correspondences were present in execution logs available during review.

Contact-sheet SHA256:
d458f7b3665b7d774714f77a528dcbb089d1d7eacc17216335781f9c2e131e86

Frozen acquisition-QC CSV SHA256:
ec6ee33ec5ba925844afece6cb38385106dbdd75a374b5cc1953a5b39f2cb9e2
