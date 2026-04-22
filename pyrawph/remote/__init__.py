from .insula_client import InsulaClient
from .constants import (
    PHISAT2_BASE_URL,
    PHISAT2_L0_COLLECTION,
    PHISAT2_L1_COLLECTION,
    DEFAULT_DOWNLOAD_DIR,
    DEFAULT_L0_DOWNLOAD_DIR,
    DEFAULT_L1_DOWNLOAD_DIR,
    VM_L0_ROOT,
    VM_L1_ROOT,
)
from .auth import connect_insula

__all__ = [
    "InsulaClient",
    "connect_insula",
    "PHISAT2_BASE_URL",
    "PHISAT2_L0_COLLECTION",
    "PHISAT2_L1_COLLECTION",
    "DEFAULT_DOWNLOAD_DIR",
    "DEFAULT_L0_DOWNLOAD_DIR",
    "DEFAULT_L1_DOWNLOAD_DIR",
    "VM_L0_ROOT",
    "VM_L1_ROOT",
]