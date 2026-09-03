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

Advanced geometry
-----------------

.. code-block:: python

   georef = event.get_georef()
