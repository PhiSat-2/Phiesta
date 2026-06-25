from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = Path(os.environ.get("PHIESTA_DATA_PATH", REPO_ROOT / "data"))