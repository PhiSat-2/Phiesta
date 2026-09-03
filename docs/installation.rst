Installation
============

Clone the repository
--------------------

.. code-block:: bash

   git clone https://github.com/PhiSat-2/Phiesta.git
   cd Phiesta

Complete installation
---------------------

For the complete Phiesta workflow, including Sentinel-2 triplets and strict
georeferencing, use one installation command:

.. code-block:: bash

   python -m pip install -e ".[triplets]"

This includes the base Phiesta package. Users who only need local product
inspection can instead install the smaller base environment with
``python -m pip install -e .``.

Import checks
-------------

.. code-block:: bash

   python -c "import phiesta; from phiesta import L0_event, L1_event, L1A_event, connect_insula; print('Phiesta OK')"
   python -c "from scm_lightglue import LightGlue, SIFT; print('LightGlue OK')"

SIFT + LightGlue is the portable matching default installed by the ``triplets`` extra.
The original SuperPoint backend can still be requested with ``features="superpoint"``
when the original ``cvg/LightGlue`` package is installed separately.

Credentials
-----------

Depending on the workflow, the user may need:

- Insula credentials to load ΦSat-2 products by ID;
- CDSE credentials to download Sentinel-2 data.

Local executable binaries
-------------------------

Phiesta can include authorized platform-specific ΦSat-2 simulator executables under ``third_party/phisat2_exec/``. The triplet extra contains the Python orchestration required to call them, so no separate simulator-helper checkout or environment variable is required.

These third-party components are not automatically covered by Phiesta's Apache-2.0 license. See ``THIRD_PARTY_NOTICES.md`` before redistribution or packaging.

Minimal end-to-end georeferencing
---------------------------------

After installing the ``triplets`` extra:

.. code-block:: python

   from phiesta import connect_insula

   client = connect_insula()
   event = client.load_l1("5359")

   product = event.georeference()
   product.show_rgb()

   print(product.meta["path"])
   print(product.meta["crs"])
   print(product.meta["transform"])

``event.georeference()`` returns a new ``L1_event`` backed by the corrected
georeferenced GeoTIFF.

The Sentinel-2 search horizon defaults to 60 days in each direction. For
example, restrict it to 7 days with ``event.georeference(window_days=7)``.

The high-level workflow uses a 2048 x 2048 final simulation by default.
Native-size simulation can be requested explicitly with
``final_simulation_target_size=None``.
