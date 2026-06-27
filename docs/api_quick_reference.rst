API quick reference
===================

Loading
-------

.. code-block:: python

   from phiesta import connect_insula, L1_event

   client = connect_insula()
   event = client.load_l1("5359")

   event = L1_event.from_path("/path/to/PHISAT-2_L1_...")

Inspection
----------

.. code-block:: python

   event.show_event_info()

Visualization
-------------

.. code-block:: python

   event.show_all_bands(normalization="percentile", percentiles=(1, 99))
   event.show_band("NIR", normalization="percentile", percentiles=(1, 99))
   event.show_rgb(bands=("RED", "GREEN", "BLUE"), per_band=True)
   event.show_rgb(bands=("NIR", "RED", "GREEN"), registered=True)

Statistics
----------

.. code-block:: python

   stats = event.band_stats(
       bands=("BLUE", "GREEN", "RED", "NIR"),
       sample_size=100_000,
       percentiles=(1, 50, 99),
   )

Patchify
--------

.. code-block:: python

   patch_index = event.build_patch_index(patch_size=1024, stride=1024)

   for item in event.iter_patches(patch_size=1024, stride=1024):
       patch = item["patch"]
       meta = item["metadata"]
       break

   table = event.export_patches(
       out_dir="outputs/patches",
       patch_size=1024,
       stride=1024,
   )

Georeferencing
--------------

.. code-block:: python

   georef = event.get_georef(
       sentinel_backend="download",
       source="simulated",
       verbose=True,
   )

Manual triplet workflow
-----------------------

.. code-block:: python

   triplet = event.build_full_sentinel_triplet(
       sentinel_backend="download",
       buffer_km=20,
       proxy_target_size=(1024, 1024),
       verbose=True,
   )

   strict = event.refine_triplet_georeference_strict(
       triplet,
       source="simulated",
       verbose=True,
   )
