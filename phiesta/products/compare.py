from __future__ import annotations

from typing import Any

import pandas as pd

from .anatomy import (
    product_card,
    compare_product_folders,
    processing_switches,
    raster_inventory,
)
from phiesta.geometry import interband_shift_table


def compare_processing_switches(left: Any, right: Any) -> pd.DataFrame:
    a = processing_switches(left).rename(columns={"value": "left_value", "level": "left_level"})
    b = processing_switches(right).rename(columns={"value": "right_value", "level": "right_level"})

    out = a[["key", "left_level", "left_value"]].merge(
        b[["key", "right_level", "right_value"]],
        on="key",
        how="outer",
    )
    out["same"] = out["left_value"].astype(str) == out["right_value"].astype(str)
    return out


def _shift_summary(event: Any, *, master_band=2, max_side=1024) -> dict:
    try:
        df = interband_shift_table(event, master_band=master_band, max_side=max_side)
        ok = df[df["status"] == "ok"]
        if ok.empty:
            return {"status": "failed", "n": 0}
        return {
            "status": "ok",
            "n": int(len(ok)),
            "master_band": master_band,
            "median_shift_px": float(ok["shift_px"].median()),
            "mean_shift_px": float(ok["shift_px"].mean()),
            "max_shift_px": float(ok["shift_px"].max()),
            "median_response": float(ok["response"].median()),
        }
    except Exception as e:
        return {"status": "failed", "error": f"{type(e).__name__}: {e}"}


def compare_levels(
    left: Any,
    right: Any,
    *,
    include_shift: bool = True,
    include_mission_specs: bool = True,
    master_band=2,
    max_side: int = 1024,
) -> dict:
    """
    Compare two PhiSat-2 products or processing levels.

    Supports L0, L1A, L1/L1C event-like objects or product folders.
    """
    left_card = product_card(left)
    right_card = product_card(right)

    folder_cmp = compare_product_folders(left, right)
    switch_cmp = compare_processing_switches(left, right)

    left_rasters = raster_inventory(left)
    right_rasters = raster_inventory(right)

    report = {
        "left": left_card,
        "right": right_card,
        "same_product_id": left_card.get("product_id") == right_card.get("product_id"),
        "levels": {
            "left": left_card.get("level"),
            "right": right_card.get("level"),
        },
        "file_families": {
            "common": folder_cmp["common_families"],
            "left_only": folder_cmp["left_only_families"],
            "right_only": folder_cmp["right_only_families"],
        },
        "files": {
            "common_count": len(folder_cmp["common_files"]),
            "left_only_count": len(folder_cmp["left_only_files"]),
            "right_only_count": len(folder_cmp["right_only_files"]),
        },
        "geolocation": {
            "left_has_geolocation": left_card.get("has_geolocation"),
            "right_has_geolocation": right_card.get("has_geolocation"),
            "left_crs_values": left_card.get("crs_values"),
            "right_crs_values": right_card.get("crs_values"),
        },
        "rasters": {
            "left_count": int(len(left_rasters)),
            "right_count": int(len(right_rasters)),
            "left_total_mb": round(float(left_rasters["size_mb"].sum()), 3) if not left_rasters.empty else 0.0,
            "right_total_mb": round(float(right_rasters["size_mb"].sum()), 3) if not right_rasters.empty else 0.0,
        },
        "processing_switch_differences": switch_cmp[~switch_cmp["same"]].to_dict(orient="records"),
    }

    if include_shift:
        report["interband_shift"] = {
            "note": "Fast global phase-correlation diagnostic using existing Phiesta registration utilities.",
            "left": _shift_summary(left, master_band=master_band, max_side=max_side),
            "right": _shift_summary(right, master_band=master_band, max_side=max_side),
        }

    if include_mission_specs:
        from phiesta.specs import mission_spec_report

        left_mission = mission_spec_report(left)
        right_mission = mission_spec_report(right)

        report["mission_specs"] = {
            "note": "Mission-aware consistency checks against encoded PhiSat-2 product-level expectations.",
            "left": {
                "overall_ok": left_mission["overall_ok"],
                "level": left_mission["level"],
                "spec": left_mission["spec"],
                "checks": left_mission["checks"],
            },
            "right": {
                "overall_ok": right_mission["overall_ok"],
                "level": right_mission["level"],
                "spec": right_mission["spec"],
                "checks": right_mission["checks"],
            },
        }

    return report
