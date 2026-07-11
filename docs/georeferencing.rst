Georeferencing guide
===================

High-level workflow
-------------------

The recommended entry point is:

.. code-block:: python

   georef = event.get_georef(
       sentinel_backend="download",
       source="simulated",
       verbose=True,
   )

This function builds the required Sentinel-2 context, performs alignment, and returns a Python dictionary with the refined georeference.

From an Insula acquisition ID
-----------------------------

.. code-block:: python

   import json
   from pathlib import Path

   from phiesta import connect_insula

   PRODUCT_ID = "5359"

   client = connect_insula()
   event = client.load_l1(PRODUCT_ID)

   georef = event.get_georef(
       sentinel_backend="download",
       source="simulated",
       verbose=True,
   )

   out = Path(f"georef_{PRODUCT_ID}.json")
   out.write_text(json.dumps(georef, indent=2), encoding="utf-8")

   print("quality:", georef["quality"])
   print("corners_lonlat:", georef["corners_lonlat"])
   print("center_lonlat:", georef["center_lonlat"])
   print("polygon_geojson:", georef["polygon_geojson"])
   print("metrics:", georef["metrics"])
   print("saved:", out)

From a local L1 product
-----------------------

.. code-block:: python

   import json
   from pathlib import Path

   from phiesta import L1_event

   PRODUCT_PATH = r"/path/to/PHISAT-2_L1_..."
   PRODUCT_ID = "local_product"

   event = L1_event.from_path(PRODUCT_PATH)

   georef = event.get_georef(
       sentinel_backend="download",
       source="simulated",
       verbose=True,
   )

   out = Path(f"georef_{PRODUCT_ID}.json")
   out.write_text(json.dumps(georef, indent=2), encoding="utf-8")


Source search strategy
----------------------

The Sentinel-2 temporal window is a configurable search horizon. It is not a
hard georeferencing validity criterion. A closer Sentinel-2 acquisition is
usually preferable, but the final georeference should be judged from alignment
quality: matches, inliers, inlier ratio, residual errors, coverage and visual
inspection. Larger temporal gaps can still be useful for georeferencing stable
scene structures.

Output structure
----------------

The returned dictionary contains:

.. code-block:: python

   georef["quality"]
   georef["corners_lonlat"]
   georef["center_lonlat"]
   georef["polygon_geojson"]
   georef["metrics"]
   georef["paths"]

Typical paths include strict reports, overlay previews, warped Sentinel products, simulated products, match tables and inlier tables.

Notes
-----

``sentinel_backend="download"`` uses CDSE to retrieve Sentinel-2 products. It may ask for CDSE credentials.

``source="simulated"`` means that the strict georeference is refined through the simulated ΦSat-2 representation of Sentinel-2.
