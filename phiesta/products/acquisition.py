from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from phiesta.l0 import raw_l0_report
from phiesta.products.opening import _find_local_product, open_product, open_raw_l0_product
from phiesta.products.anatomy import product_card
from phiesta.products.compare import compare_levels
from phiesta.specs import mission_spec_report


def _id_for_display(identifier: str | int | Path) -> str:
    s = str(identifier)
    if s.isdigit():
        return str(int(s))
    return s


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _missing_entry(level: str, identifier: str | int | Path) -> dict[str, Any]:
    return {
        "level": level,
        "available": False,
        "status": "missing_local",
        "identifier": _id_for_display(identifier),
        "folder": None,
        "note": "No local product found. Set download_missing=True to allow Insula fallback.",
    }


def _error_entry(level: str, identifier: str | int | Path, error: Exception) -> dict[str, Any]:
    return {
        "level": level,
        "available": False,
        "status": "error",
        "identifier": _id_for_display(identifier),
        "folder": None,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _inspect_l0_raw(
    identifier: str | int | Path,
    *,
    client: Any = None,
    prefer_local: bool = True,
    download_missing: bool = False,
    **kwargs,
) -> dict[str, Any]:
    local_path = _find_local_product(identifier, "L0")

    if local_path is None and not download_missing:
        return _missing_entry("L0", identifier)

    try:
        if local_path is not None and prefer_local:
            folder = Path(local_path)
        else:
            folder = open_raw_l0_product(
                identifier,
                client=client,
                prefer_local=prefer_local,
                **kwargs,
            )

        report = raw_l0_report(folder)
        return {
            "level": "L0",
            "available": True,
            "status": "raw_folder",
            "identifier": _id_for_display(identifier),
            "folder": str(folder),
            "raw_report": report,
        }
    except Exception as e:
        return _error_entry("L0", identifier, e)


def _inspect_processed_level(
    identifier: str | int | Path,
    level: str,
    *,
    client: Any = None,
    prefer_local: bool = True,
    download_missing: bool = False,
    include_mission_specs: bool = True,
    **kwargs,
) -> tuple[dict[str, Any], Any | None]:
    local_path = _find_local_product(identifier, level)

    if local_path is None and not download_missing:
        return _missing_entry(level, identifier), None

    try:
        event = open_product(
            identifier,
            level=level,
            client=client,
            prefer_local=prefer_local,
            **kwargs,
        )

        card = product_card(event)

        entry: dict[str, Any] = {
            "level": level,
            "available": True,
            "status": "opened",
            "identifier": _id_for_display(identifier),
            "folder": card.get("folder"),
            "product_card": card,
        }

        if include_mission_specs:
            ms = mission_spec_report(event)
            entry["mission_spec"] = {
                "overall_ok": ms.get("overall_ok"),
                "spec": ms.get("spec"),
                "checks": ms.get("checks"),
                "note": ms.get("note"),
            }

        return entry, event
    except Exception as e:
        return _error_entry(level, identifier, e), None


def acquisition_report(
    identifier: str | int | Path,
    *,
    client: Any = None,
    prefer_local: bool = True,
    download_missing: bool = False,
    include_l0: bool = True,
    include_l1a: bool = True,
    include_l1c: bool = True,
    include_mission_specs: bool = True,
    include_l1a_l1c_comparison: bool = True,
    include_shift: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    Build an acquisition-level PhiSat-2 report across available product levels.

    By default this is local-only and does not trigger Insula downloads.
    Set download_missing=True to allow online fallback for missing levels.
    """
    levels: dict[str, Any] = {}
    events: dict[str, Any] = {}

    if include_l0:
        levels["L0"] = _inspect_l0_raw(
            identifier,
            client=client,
            prefer_local=prefer_local,
            download_missing=download_missing,
            **kwargs,
        )

    if include_l1a:
        entry, event = _inspect_processed_level(
            identifier,
            "L1A",
            client=client,
            prefer_local=prefer_local,
            download_missing=download_missing,
            include_mission_specs=include_mission_specs,
            **kwargs,
        )
        levels["L1A"] = entry
        if event is not None:
            events["L1A"] = event

    if include_l1c:
        entry, event = _inspect_processed_level(
            identifier,
            "L1C",
            client=client,
            prefer_local=prefer_local,
            download_missing=download_missing,
            include_mission_specs=include_mission_specs,
            **kwargs,
        )
        levels["L1C"] = entry
        if event is not None:
            events["L1C"] = event

    available_levels = [
        level for level, entry in levels.items()
        if isinstance(entry, dict) and entry.get("available") is True
    ]
    missing_levels = [
        level for level, entry in levels.items()
        if isinstance(entry, dict) and entry.get("available") is False
    ]

    warnings = []

    l0_entry = levels.get("L0")
    if isinstance(l0_entry, dict) and l0_entry.get("available"):
        raw_report = l0_entry.get("raw_report", {})
        conversion = raw_report.get("conversion", {})
        if conversion.get("needs_external_converter"):
            warnings.append(
                "Raw L0 is available, but prepared L0_event decoding requires the external Simera/SENSE converter."
            )

    comparison = None
    if include_l1a_l1c_comparison and "L1A" in events and "L1C" in events:
        try:
            comparison = compare_levels(
                events["L1A"],
                events["L1C"],
                include_shift=include_shift,
                include_mission_specs=include_mission_specs,
            )
        except Exception as e:
            comparison = {
                "status": "error",
                "error_type": type(e).__name__,
                "error": str(e),
            }

    report = {
        "identifier": _id_for_display(identifier),
        "download_missing": download_missing,
        "summary": {
            "available_levels": available_levels,
            "missing_or_failed_levels": missing_levels,
            "has_l0_raw": "L0" in available_levels,
            "has_l1a": "L1A" in available_levels,
            "has_l1c": "L1C" in available_levels,
            "has_l1a_l1c_pair": "L1A" in events and "L1C" in events,
            "warnings": warnings,
        },
        "levels": levels,
        "l1a_l1c_comparison": comparison,
        "note": (
            "Acquisition-level report across raw and processed PhiSat-2 products. "
            "By default, missing levels are not downloaded."
        ),
    }

    return _json_safe(report)
