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

Recommended high-level API:

.. code-block:: python

   product = event.georeference()
   product.show_rgb()
   product.show_band("NIR")
   print(product.meta["path"])
   print(product.meta["crs"])
   print(product.meta["transform"])

Restrict the Sentinel-2 temporal search:

.. code-block:: python

   product = event.georeference(window_days=7)

Advanced geometry and matching diagnostics:

.. code-block:: python

   georef = event.get_georef()


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


WorldCover dataset prefilter
----------------------------

Find L1 acquisitions whose catalog footprint, buffered by the default 30 km
spatial tolerance, contains a requested ESA WorldCover class:

.. code-block:: python

   candidates = client.search_l1_worldcover("mangrove")

The default ``min_fraction`` is ``1e-6``. Server-side WorldCover statistics use ``statistics_max_size=1024`` by default to keep catalog-wide filtering practical. The operation uses catalog geometry and public Planetary Computer WorldCover statistics; it does not download PhiSat-2 acquisitions, store WorldCover tiles, or run georeferencing.

.. code-block:: python

   events = client.load_l1_table(candidates)

For literal one-pixel presence, use ``min_fraction=0.0``. Candidates still
need at least one target-class pixel to be returned.


WorldCover service failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default search is recall-oriented. If the external WorldCover statistics
service still fails for one acquisition after retries, that acquisition is
retained in the returned candidate table and its ``worldcover_status`` is
``"uncertain"``. This prevents transient service failures from becoming silent
false negatives. Use ``include_uncertain=False`` for fail-fast behavior.


Generic dataset building
------------------------

Selections are independent from dataset construction. A selection may be a
search/filter DataFrame, CSV file, Insula search result, or list of product ids.

.. code-block:: python

   dataset = client.build_l1_dataset(
       ["5359", "5360"],
       out_dir="datasets/example",
   )

Provide ``patch_size`` to export ML-ready ``.npy`` patches. Every selection
column is propagated to acquisition and patch manifests, so arbitrary labels,
groups, split assignments, WorldCover scores, and user annotations are kept.

.. code-block:: python

   dataset = client.build_l1_dataset(
       selection,
       out_dir="datasets/example",
       patch_size=1024,
       stride=1024,
   )

For corrected L1 rasters before patch extraction, set ``georeference=True``.
Builds checkpoint after every acquisition and resume by default. Use
``open_dataset(...)`` to reopen a built dataset.


Leakage-safe dataset splits
---------------------------

.. code-block:: python

   dataset.make_splits(
       train=0.8,
       val=0.1,
       test=0.1,
       seed=42,
   )

Splits are assigned at acquisition/group level and propagated to every patch.
Use ``group_by="pass_id"`` (or any manifest column) to keep related acquisitions
together. Spatial separation is available with:

.. code-block:: python

   dataset.make_splits(
       method="spatial",
       min_distance_km=100,
       seed=42,
       overwrite=True,
   )

Spatial mode guarantees the requested minimum catalog-center distance across
different splits. Use ``dataset.split_summary()`` and
``dataset.get_split("train")`` to inspect the result.
