from __future__ import annotations

from pathlib import Path
from typing import Any


def _find_local_product(identifier: str | int | Path, level: str) -> Path | None:
    p = Path(str(identifier))
    if p.exists():
        return p

    level_u = level.upper()
    s = str(identifier)

    candidates: list[Path] = []

    if s.isdigit():
        pid9 = f"{int(s):09d}"

        if level_u == "L0":
            patterns = [f"data/l0/PHISAT-2_L0_{pid9}_*"]
        elif level_u == "L1A":
            patterns = [f"data/l1a/PHISAT-2_L1A_{pid9}_*"]
        elif level_u in {"L1", "L1C"}:
            patterns = [f"data/l1/PHISAT-2_L1_{pid9}_*"]
        else:
            patterns = []
    else:
        patterns = [
            f"data/l0/{s}*",
            f"data/l1/{s}*",
            f"data/l1a/{s}*",
        ]

    for pat in patterns:
        candidates.extend(sorted(Path(".").glob(pat)))

    return candidates[0] if candidates else None


def open_product(
    identifier: str | int | Path,
    *,
    level: str = "L1C",
    client: Any = None,
    prefer_local: bool = True,
    **kwargs,
):
    """
    Open a PhiSat-2 product from a local folder or from Insula.

    identifier can be:
    - a product id, e.g. "6008";
    - a full product identifier;
    - a local product folder path.

    level is one of: "L0", "L1", "L1C", "L1A".
    """
    from phiesta.remote.auth import connect_insula
    from phiesta.l0.l0_event import L0_event
    from phiesta.l1.l1_event import L1_event
    from phiesta.l1a.l1a_event import L1A_event

    level_u = level.upper()

    if prefer_local:
        local_path = _find_local_product(identifier, level_u)
        if local_path is not None:
            if level_u == "L0":
                return L0_event.from_path(local_path, **kwargs)
            if level_u == "L1A":
                return L1A_event.from_path(local_path, **kwargs)
            if level_u in {"L1", "L1C"}:
                return L1_event.from_path(local_path, **kwargs)

    c = client or connect_insula()

    if level_u == "L0":
        return c.load_l0(str(identifier), **kwargs)
    if level_u == "L1A":
        return c.load_l1a(str(identifier), **kwargs)
    if level_u in {"L1", "L1C"}:
        return c.load_l1c(str(identifier), **kwargs)

    raise ValueError(f"Unsupported level: {level!r}. Expected L0, L1A, L1, or L1C.")
