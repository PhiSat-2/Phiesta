Phiesta overview
================

Phiesta is a Python toolkit for working with ΦSat-2 L1 acquisitions.

Its main goal is to make the following workflow easy:

1. load a ΦSat-2 L1 acquisition;
2. inspect and visualize the product;
3. find a nearby low-cloud Sentinel-2 acquisition;
4. simulate Sentinel-2 into the ΦSat-2 spectral domain when needed;
5. align Sentinel-2 / simulated ΦSat-2 / real ΦSat-2 products;
6. return a directly usable georeference dictionary.

Main entry points
-----------------

Load a local L1 product:

.. code-block:: python

   from phiesta import L1_event

   event = L1_event.from_path("/path/to/PHISAT-2_L1_...")

Load from Insula:

.. code-block:: python

   from phiesta import connect_insula

   client = connect_insula()
   event = client.load_l1("5359")

Get a refined georeference:

.. code-block:: python

   georef = event.get_georef(
       sentinel_backend="download",
       source="simulated",
       verbose=True,
   )

Useful output fields
--------------------

``event.get_georef(...)`` returns a dictionary containing:

- ``quality``: quality label derived from alignment metrics;
- ``corners_lonlat``: footprint corners as ``[lon, lat]`` pairs;
- ``center_lonlat``: acquisition center as ``[lon, lat]``;
- ``polygon_geojson``: GeoJSON polygon;
- ``metrics``: matching and strict-alignment metrics;
- ``paths``: report, preview image, warped products, matches and inliers.

Documentation
-------------

- :doc:`installation`
- :doc:`georeferencing`
- :doc:`api_quick_reference`
