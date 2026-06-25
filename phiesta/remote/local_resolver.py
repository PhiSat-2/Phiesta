from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence


def _iter_existing_roots(roots: Sequence[str | Path]):
    for root in roots:
        p = Path(root)
        if p.exists() and p.is_dir():
            yield p


def _extract_acquisition_id(name: str) -> Optional[str]:
    m = re.search(r"PHISAT-2_L[01]_(\d{9})_", name, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _normalize_identifier(identifier: str) -> str:
    ident = str(identifier).strip()

    if ident.isdigit():
        return ident.zfill(9)

    m = re.search(r"(\d{9})", ident)
    if m:
        return m.group(1)

    return ident


def find_product_folder_by_identifier(
    identifier: str,
    roots: Sequence[str | Path],
) -> Optional[Path]:
    target = _normalize_identifier(identifier)

    for root in _iter_existing_roots(roots):
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue

            acq_id = _extract_acquisition_id(p.name)
            if acq_id is None:
                continue

            if acq_id == target:
                return p

    return None


def _extract_product_date(name: str) -> Optional[str]:
    m = re.search(r"PHISAT-2_L[01]_\d{9}_(\d{8})\d{6}_", name, flags=re.IGNORECASE)
    if not m:
        return None
    d = m.group(1)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def _extract_product_start_timestamp(name: str) -> str:
    m = re.search(r"PHISAT-2_L[01]_\d{9}_(\d{14})_", name, flags=re.IGNORECASE)
    if not m:
        return ""
    return m.group(1)


def find_product_folders_by_date(
    date_str: str,
    roots: Sequence[str | Path],
) -> List[Path]:
    target = str(date_str).strip()
    out: List[Path] = []

    for root in _iter_existing_roots(roots):
        for p in root.iterdir():
            if not p.is_dir():
                continue

            product_date = _extract_product_date(p.name)
            if product_date == target:
                out.append(p)

    out.sort(key=lambda p: _extract_product_start_timestamp(p.name), reverse=True)
    return out


def resolve_existing_product(
    *,
    identifier: Optional[str] = None,
    roots: Sequence[str | Path],
) -> Optional[Path]:
    if identifier:
        return find_product_folder_by_identifier(identifier, roots)
    return None