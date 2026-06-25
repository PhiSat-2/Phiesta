Overview
========

Phiesta is a research-oriented Python package for working with ΦSat-2
L0/L1 products and building Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplets.

Main features
-------------

- Load local ΦSat-2 L0 and L1 products.
- Load ΦSat-2 L1 acquisitions from Insula.
- Access spectral bands by index, wavelength, or alias.
- Build RGB composites and simple spectral indices.
- Register L0 and L1 products.
- Use catalog geometry to compare ΦSat-2 and Sentinel-2.
- Build full pixel-aligned Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 triplets.
- Run triplet generation in batch mode.

Typical public entry points
---------------------------

- ``phiesta.connect_insula``
- ``phiesta.L1_event``
- ``phiesta.L0_event``
- ``event.build_full_sentinel_triplet()``
- ``event.inspect_full_sentinel_triplet()``
- ``event.show_full_sentinel_triplet()``
- ``phiesta.triplets.batch.build_full_sentinel_triplets_batch``

Minimal triplet example
-----------------------

.. code-block:: python

   from phiesta import connect_insula

   client = connect_insula()
   event = client.load_l1("5359")

   triplet = event.build_full_sentinel_triplet()
   event.inspect_full_sentinel_triplet(triplet)
   event.show_full_sentinel_triplet(triplet)

The final triplet contains:

- real ΦSat-2 L1 image;
- Sentinel-2B warped to the ΦSat-2 grid;
- simulated ΦSat-2 generated from Sentinel-2B and warped to the real ΦSat-2 grid.

Notes
-----

The final triplet rasters are primarily intended for pixel-aligned machine
learning workflows. Their CRS/transform metadata should be treated as grid
metadata unless further geodetic validation is performed.
