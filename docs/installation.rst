Installation
============

Clone the repository
--------------------

.. code-block:: bash

   git clone https://github.com/malodept/Phiesta.git
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
   python -m pip install git+https://github.com/cvg/LightGlue.git

Import checks
-------------

.. code-block:: bash

   python -c "import phiesta; from phiesta import L1_event, connect_insula; print('Phiesta OK')"
   python -c "from lightglue import LightGlue, SuperPoint, SIFT; print('LightGlue OK')"

Credentials
-----------

Depending on the workflow, the user may need:

- Insula credentials to load ΦSat-2 products by ID;
- CDSE credentials to download Sentinel-2 data.

Local executable binaries
-------------------------

Phiesta includes public simulator Python code in ``third_party/orbitalai_phisat2_sim/``.

The repository does not distribute local/platform-specific executable binaries. Authorized users can place such files locally in ``third_party/phisat2_exec/`` or pass paths explicitly where supported.
