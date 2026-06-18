# PyRawPh-Light TODO

## Alignment quality controls

The current full triplet pipeline produces pixel-aligned triplets using catalog geometry, proxy simulation, and LightGlue-based alignment.

Future improvements to consider:

- expose alignment quality parameters to users;
- save more explicit alignment metrics in the triplet report;
- add optional stricter matching mode for difficult acquisitions;
- add optional second refinement step when overlay quality is not sufficient;
- compute visual/quantitative quality scores beyond inlier ratio and valid fraction;
- document when the final CRS/transform should be considered grid metadata rather than precise geodetic metadata.

This is intentionally not implemented yet. The current priority is to keep the public API simple and stable.
