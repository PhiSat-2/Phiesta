Geometry diagnostics
====================

Phiesta treats geometry at three distinct scales:

1. **intra-product / inter-band geometry** — whether spectral bands are mutually aligned;
2. **inter-level geometry** — how raw L0 geometry maps into a processed L1 reference space;
3. **absolute geolocation** — where the product lies on Earth, optionally refined against Sentinel-2.

These are different error sources and should not be collapsed into a single
"geolocation error" number.

Inter-band translations
-----------------------

A fast global diagnostic:

.. code-block:: python

   from phiesta import interband_shift_table

   shifts = interband_shift_table(
       event,
       master_band="RED",
   )
   print(shifts[["target_band", "dx_px", "dy_px", "shift_px"]])

The sign convention is explicit: ``dx_px`` / ``dy_px`` are the translations to
apply to the target band so that it aligns with the master band.

The table also reports correlation before and after alignment. This is useful as
a sanity check, especially for spectrally distant bands.

Local displacement fields
-------------------------

A single global translation can hide spatially varying residual geometry:

.. code-block:: python

   from phiesta import local_interband_shift_field, plot_shift_map

   field = local_interband_shift_field(
       event,
       master_band="RED",
       target_band="NIR",
       window_size=512,
       stride=256,
   )

   fig, ax = plot_shift_map(field)

Low-texture windows are reported explicitly rather than assigned arbitrary
translations.

Visual edge overlays
--------------------

.. code-block:: python

   from phiesta import edge_overlay

   before = edge_overlay(
       event,
       band_a="RED",
       band_b="NIR",
       align=False,
   )

   after = edge_overlay(
       event,
       band_a="RED",
       band_b="NIR",
       align=True,
   )

The returned RGB arrays use red for the first band's edges and cyan for the
second. Coincident edges appear approximately white.

Register all bands to one master
--------------------------------

.. code-block:: python

   from phiesta import register_bands

   aligned = register_bands(
       event,
       master_band="NIR",
   )

This returns a new event-like object and leaves the source event unchanged.

L0 into L1 reference space
--------------------------

For paired L0/L1 products:

.. code-block:: python

   from phiesta import register_l0_to_l1

   l0_in_l1 = register_l0_to_l1(
       l0_event,
       l1_event,
       master_band="NIR",
   )

   info = l0_in_l1.meta["registration_info"]

``registration_info`` records the master shift, per-band residual shifts, crop
strategy, source L1 path, and output geometry.

This translation-based registration is an analysis utility, not a replacement
for the mission processing chain.

Absolute geolocation
--------------------

Absolute product positioning is handled separately through Sentinel-assisted
georeferencing:

.. code-block:: python

   corrected = l1_event.georeference()

See :doc:`georeferencing` for the full workflow.

Research use
------------

The geometry utilities are designed to make processing-chain questions
reproducible: which offsets already exist in raw data, which are reduced or
changed by processing, whether residuals are rigid or spatially varying, and how
those internal errors relate to absolute geolocation.
