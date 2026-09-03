Georeferencing guide
====================

Recommended workflow
--------------------

.. code-block:: python

   product = event.georeference()

Phiesta searches for suitable Sentinel-2 context, performs the required
simulation and alignment, exports the corrected PhiSat-2 raster as a standard
GeoTIFF, and returns it as a new ``L1_event``.

.. code-block:: python

   product.show_rgb()
   product.show_band("NIR")
   red = product.get_band("RED")
   print(product.meta["path"])
   print(product.meta["crs"])
   print(product.meta["transform"])

From an Insula acquisition
--------------------------

.. code-block:: python

   from phiesta import connect_insula
   client = connect_insula()
   event = client.load_l1("5359")
   product = event.georeference()
   product.show_rgb()

Temporal search window
----------------------

The default Sentinel-2 search horizon is +/-60 days.

.. code-block:: python

   product = event.georeference(window_days=7)

Final simulation size
---------------------

The high-level workflow uses a 2048 x 2048 final simulation by default.
Native-size simulation can be requested with:

.. code-block:: python

   product = event.georeference(final_simulation_target_size=None)

Advanced geometric output
-------------------------

.. code-block:: python

   georef = event.get_georef()

Use ``get_georef()`` for intermediate homographies, footprints, matching
metrics, strict reports, overlays, matches and inliers.
