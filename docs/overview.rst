Overview
========

PyRawPh-Light is a lightweight Python package for ΦSat-2 local product handling.

Main features
-------------

- Load local L1 events with ``L1_event.from_path()``
- Load local L0 events with ``L0_event.from_path()``
- Access bands by index, wavelength, or alias
- Compute simple products such as RGB and NDVI
- Export arrays to TIFF / GeoTIFF
- Register L0 data into L1 reference space

Typical public entry points
---------------------------

- ``pyrawph.L1_event``
- ``pyrawph.L0_event``
- ``pyrawph.register_l0_to_l1_space``
- ``pyrawph.project_points_l1_to_l0_native``
- ``pyrawph.project_bbox_l1_to_l0_native``