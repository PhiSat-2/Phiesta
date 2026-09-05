Phiesta overview
================

Phiesta is a Python toolkit for working with PhiSat-2 products.

Load from Insula
----------------

.. code-block:: python

   from phiesta import connect_insula
   client = connect_insula()
   event = client.load_l1("5359")

Georeference
------------

.. code-block:: python

   product = event.georeference()
   product.show_rgb()
   print(product.meta["path"])
   print(product.meta["crs"])
   print(product.meta["transform"])

``product`` is another ``L1_event`` representing the corrected georeferenced
raster, so methods such as ``show_rgb()``, ``show_band()``, ``get_band()`` and
``to_cube()`` continue to work.

The default Sentinel-2 search horizon is +/-60 days. For example:

.. code-block:: python

   product = event.georeference(window_days=7)

Build an ML dataset
-------------------

Dataset selection is independent from construction. A selection can be a list
of product ids, a catalog-search table, a WorldCover-filtered table, a CSV, or a
custom pandas DataFrame.

.. code-block:: python

   selection = ["5359", "5360"]

   dataset = client.build_l1_dataset(
       selection,
       out_dir="datasets/example",
       patch_size=512,
       stride=512,
   )

Create leakage-safe splits at acquisition/group level and propagate them to
every patch:

.. code-block:: python

   dataset.make_splits(
       train=0.8,
       val=0.1,
       test=0.1,
       seed=42,
   )

Targets can be scalar labels or aligned raster masks. With the optional ML
dependency, datasets can be consumed directly by PyTorch:

.. code-block:: python

   loader = dataset.to_dataloader(
       split="train",
       targets="class",
       batch_size=16,
   )

See ``examples/dataset_training_quickstart.py`` for the complete workflow.

Advanced geometry
-----------------

.. code-block:: python

   georef = event.get_georef()
