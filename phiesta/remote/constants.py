from pathlib import Path
from ..sys_cfg import DATA_PATH

PHISAT2_BASE_URL = "https://phisat2.insula.earth"

# Collection IDs observed on the current PHISAT2 Insula instance.
PHISAT2_L0_COLLECTION = "phisat277ab0067a83a48f69a26e03f4963e0e2"
# Historical name kept for backward compatibility.
# In current Phiesta usage, "L1" refers to the available L1C-like products.
PHISAT2_L1C_COLLECTION = "phisat24e55ba83dd304ea9b018b65e9b17a7de"
PHISAT2_L1_COLLECTION = PHISAT2_L1C_COLLECTION

# Fill this once the Insula L1A refDataCollection id is confirmed.
PHISAT2_L1A_COLLECTION = None

DEFAULT_DOWNLOAD_DIR = DATA_PATH
DEFAULT_L0_DOWNLOAD_DIR = DEFAULT_DOWNLOAD_DIR / "l0"
DEFAULT_L1_DOWNLOAD_DIR = DEFAULT_DOWNLOAD_DIR / "l1"

VM_L0_ROOT = Path("/shared/projects/phisat2/data/raw/phisat2/L0")
VM_L1_ROOT = Path("/shared/projects/phisat2/data/raw/phisat2/L1")