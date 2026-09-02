Installation
============

Clone the repository
--------------------

.. code-block:: bash

   git clone https://github.com/PhiSat-2/Phiesta.git
   cd Phiesta

Base installation
-----------------

.. code-block:: bash

   python -m pip install -e .

Triplet / strict georeferencing installation
--------------------------------------------

For Sentinel-2 triplets and LightGlue-based strict georeferencing:

.. code-block:: bash

   python -m pip install -e ".[triplets]"

Import checks
-------------

.. code-block:: bash

   python -c "import phiesta; from phiesta import L0_event, L1_event, L1A_event, connect_insula; print('Phiesta OK')"
   python -c "from lightglue import LightGlue, SuperPoint, SIFT; print('LightGlue OK')"

Credentials
-----------

Depending on the workflow, the user may need:

- Insula credentials to load ΦSat-2 products by ID;
- CDSE credentials to download Sentinel-2 data.

Local executable binaries
-------------------------

Phiesta can include authorized platform-specific ΦSat-2 simulator executables under ``third_party/phisat2_exec/``. The triplet extra contains the Python orchestration required to call them, so no separate simulator-helper checkout or environment variable is required.

These third-party components are not automatically covered by Phiesta's Apache-2.0 license. See ``THIRD_PARTY_NOTICES.md`` before redistribution or packaging.
