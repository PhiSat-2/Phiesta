from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def _as_path(product: Any) -> Path:
    if isinstance(product, (str, Path)):
        return Path(product)

    for attr in ["product_folder", "root", "path"]:
        if hasattr(product, attr):
            value = getattr(product, attr)
            if callable(value):
                value = value()
            if value is not None:
                return Path(value)

    raise TypeError(f"Cannot resolve product path from object of type {type(product)!r}")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _product_id_from_name(name: str) -> str | None:
    match = re.search(r"PHISAT-2_L0_(\d{9})_", name)
    if not match:
        return None
    return str(int(match.group(1)))


def _size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return round(path.stat().st_size / 1_000_000, 3)


def _converter_status() -> dict[str, Any]:
    env = os.environ.get("PHIESTA_SIM_ROOT")
    candidates = []

    if env:
        candidates.append(Path(env))

    candidates.append(
        Path("third_party/simtotiff/SA072-SENSE-Conversion-Code-main/SA072-SENSE-Conversion-Code-main")
    )

    existing = [str(p) for p in candidates if p.exists()]

    return {
        "configured": bool(existing),
        "env_PHIESTA_SIM_ROOT": env,
        "existing_candidates": existing,
        "note": (
            "Prepared L0_event construction from raw.bin requires the external "
            "Simera/SENSE converter. Raw L0 folder inspection does not require it."
        ),
    }


def raw_l0_report(product: Any) -> dict[str, Any]:
    """
    Inspect an unprepared PhiSat-2 L0 raw product folder.

    This does not decode raw.bin and does not require the external Simera/SENSE converter.
    """
    root = _as_path(product)

    metadata = _read_json(root / "metadata.json")
    ancillary = _read_json(root / "ancillary.json")
    aocs = _read_json(root / "aocs.json")

    files = [p for p in root.rglob("*") if p.is_file()]
    total_mb = round(sum(p.stat().st_size for p in files) / 1_000_000, 3)

    raw_bin = root / "raw.bin"
    prepared_tiffs = sorted(str(p.relative_to(root)) for p in root.rglob("*.tif*"))
    json_files = sorted(str(p.relative_to(root)) for p in root.rglob("*.json"))

    other_files = []
    for p in files:
        if p.suffix.lower() not in {".json", ".tif", ".tiff"}:
            other_files.append(str(p.relative_to(root)))
    other_files = sorted(other_files)

    has_prepared_tiffs = bool(prepared_tiffs)

    return {
        "product_id": _product_id_from_name(root.name),
        "level": "L0",
        "folder": str(root),
        "exists": root.exists(),
        "n_files": len(files),
        "total_mb": total_mb,
        "raw": {
            "has_raw_bin": raw_bin.exists(),
            "raw_bin_mb": _size_mb(raw_bin),
            "has_metadata_json": (root / "metadata.json").exists(),
            "metadata_json_mb": _size_mb(root / "metadata.json"),
            "has_ancillary_json": (root / "ancillary.json").exists(),
            "ancillary_json_mb": _size_mb(root / "ancillary.json"),
            "has_aocs_json": (root / "aocs.json").exists(),
            "aocs_json_mb": _size_mb(root / "aocs.json"),
            "has_thumbnail": any(
                (root / f"thumbnail{ext}").exists()
                for ext in [".webp", ".png", ".jpg", ".jpeg"]
            ),
        },
        "prepared": {
            "has_prepared_tiffs": has_prepared_tiffs,
            "n_prepared_tiffs": len(prepared_tiffs),
            "prepared_tiffs": prepared_tiffs[:50],
        },
        "metadata": {
            "metadata_keys": sorted(metadata.keys()) if isinstance(metadata, dict) else [],
            "ancillary_keys": sorted(ancillary.keys()) if isinstance(ancillary, dict) else [],
            "aocs_keys": sorted(aocs.keys()) if isinstance(aocs, dict) else [],
        },
        "files": {
            "json_files": json_files,
            "other_files": other_files[:100],
        },
        "conversion": {
            "needs_external_converter": raw_bin.exists() and not has_prepared_tiffs,
            "converter": _converter_status(),
        },
        "note": (
            "Raw L0 anatomy report. It reports the downloaded raw product folder "
            "without decoding raw.bin."
        ),
    }


def raw_l0_table(products: list[Any]) -> list[dict[str, Any]]:
    rows = []

    for product in products:
        report = raw_l0_report(product)
        rows.append({
            "product_id": report["product_id"],
            "folder": report["folder"],
            "n_files": report["n_files"],
            "total_mb": report["total_mb"],
            "has_raw_bin": report["raw"]["has_raw_bin"],
            "raw_bin_mb": report["raw"]["raw_bin_mb"],
            "has_metadata_json": report["raw"]["has_metadata_json"],
            "has_ancillary_json": report["raw"]["has_ancillary_json"],
            "has_aocs_json": report["raw"]["has_aocs_json"],
            "has_thumbnail": report["raw"]["has_thumbnail"],
            "has_prepared_tiffs": report["prepared"]["has_prepared_tiffs"],
            "needs_external_converter": report["conversion"]["needs_external_converter"],
        })

    return rows
