"""
Backward-compatible alias for the renamed Phiesta package.

Prefer:

    import phiesta

This compatibility layer may be removed in a future release.
"""

from __future__ import annotations

import importlib
import sys

from phiesta import *  # noqa: F401,F403

try:
    from phiesta import __all__ as __all__  # type: ignore
except Exception:
    __all__ = []

for _submodule in ["georef", "l0", "l1", "remote", "triplets", "utils"]:
    sys.modules[f"{__name__}.{_submodule}"] = importlib.import_module(
        f"phiesta.{_submodule}"
    )
