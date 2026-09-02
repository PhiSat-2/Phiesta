from __future__ import annotations

import os
import platform
import stat
from pathlib import Path


def get_default_phisat2_executable() -> Path:
    """
    Return the bundled PhiSat-2 simulation executable for the current OS.
    """
    root = Path(__file__).resolve().parents[2] / "third_party" / "phisat2_exec"

    system = platform.system().lower()
    machine = platform.machine().lower()

    if "windows" in system:
        exe = root / "phisat2_win.bin"
    elif "linux" in system:
        exe = root / "phisat2_unix.bin"
    elif "darwin" in system:
        if "arm" in machine or "aarch64" in machine:
            exe = root / "phisat2_osx-arm64.bin"
        else:
            exe = root / "phisat2_osx-x86_64.bin"
    else:
        raise RuntimeError(f"Unsupported operating system for PhiSat-2 executable: {platform.system()}")

    if not exe.exists():
        raise FileNotFoundError(
            f"Bundled PhiSat-2 executable not found: {exe}"
        )

    return exe


def resolve_phisat2_executable(phisat2_exec_path: str | Path | None = None) -> Path:
    """
    Resolve the PhiSat-2 executable path.

    Priority:
    1. explicit argument
    2. PHIESTA_PHISAT2_EXEC environment variable
    3. bundled executable for current OS
    """
    if phisat2_exec_path is not None:
        path = Path(phisat2_exec_path)
    elif os.environ.get("PHIESTA_PHISAT2_EXEC"):
        path = Path(os.environ["PHIESTA_PHISAT2_EXEC"])
    else:
        path = get_default_phisat2_executable()

    if not path.exists():
        raise FileNotFoundError(f"PhiSat-2 executable not found: {path}")

    # Repository archives and some notebook environments may lose the POSIX
    # executable bit. Make source-checkout installs work without requiring a
    # separate user-facing ``chmod`` step.
    if os.name != "nt" and not os.access(path, os.X_OK):
        try:
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        except OSError as exc:
            raise PermissionError(
                "PhiSat-2 simulator exists but is not executable and Phiesta "
                f"could not enable the executable bit: {path}"
            ) from exc

    return path