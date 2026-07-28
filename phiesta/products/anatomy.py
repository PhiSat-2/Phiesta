from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

import pandas as pd

try:
    import rasterio
except Exception:
    rasterio = None


def _as_path(x: Any) -> Path:
    """Accept either a product folder path or an event with product_folder/root/path."""
    if isinstance(x, (str, Path)):
        return Path(x)

    for attr in ("product_folder", "root", "path"):
        if hasattr(x, attr):
            v = getattr(x, attr)
            if callable(v):
                try:
                    v = v()
                except TypeError:
                    pass
            if v:
                return Path(v)

    raise TypeError("Expected a product folder path or a Phiesta event-like object.")


def _product_id_from_name(name: str) -> str | None:
    m = re.search(r"_(\d{9})_", name)
    if m:
        return str(int(m.group(1)))
    return None


def _product_level_from_name(name: str) -> str | None:
    if "PHISAT-2_L1A_" in name:
        return "L1A"
    if "PHISAT-2_L1_" in name:
        return "L1C"
    if "PHISAT-2_L0_" in name:
        return "L0"
    return None


def classify_product_file(relative_path: str | Path) -> str:
    s = str(relative_path)

    if s == "AOCS.json":
        return "AOCS"
    if s == "TilingConfiguration.json":
        return "tiling_config"
    if s == "processing_config.json":
        return "processing_config"
    if "session_" in s and s.endswith("_metadata.json"):
        return "session_metadata"
    if s.startswith("logs/"):
        return "processing_log"
    if s.startswith("geolocation/"):
        return "geolocation"

    if s.startswith("bands/scene_0_RC_band_"):
        return "RC_single_band"
    if s.startswith("bands/scene_0_RC_multiband"):
        return "RC_multiband"
    if s.startswith("bands/scene_0_RC_RGB"):
        return "RC_RGB"

    if s.startswith("bands/scene_0_BC_band_"):
        return "BC_single_band"
    if s.startswith("bands/scene_0_BC_multiband"):
        return "BC_multiband"
    if s.startswith("bands/scene_0_BC_RGB"):
        return "BC_RGB"

    return "other"


def file_manifest(product: Any) -> pd.DataFrame:
    root = _as_path(product)
    rows = []

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue

        rel = p.relative_to(root)
        rows.append({
            "product_id": _product_id_from_name(root.name),
            "level": _product_level_from_name(root.name),
            "product_folder": str(root),
            "relative_path": str(rel),
            "top_dir": str(rel).split("/")[0],
            "file_name": p.name,
            "suffix": p.suffix,
            "size_mb": p.stat().st_size / 1024 / 1024,
            "family": classify_product_file(rel),
        })

    return pd.DataFrame(rows)


def file_family_summary(product: Any) -> pd.DataFrame:
    df = file_manifest(product)
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby(["level", "family"], dropna=False)
        .agg(
            files=("relative_path", "count"),
            total_mb=("size_mb", "sum"),
            examples=("relative_path", lambda x: " ; ".join(list(x)[:4])),
        )
        .reset_index()
    )
    out["total_mb"] = out["total_mb"].round(3)
    return out


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def processing_switches(product: Any) -> pd.DataFrame:
    root = _as_path(product)
    cfg = _load_json(root / "processing_config.json")

    keys = [
        "BandCoregistration",
        "CosmeticFilling",
        "Denoising",
        "AbsoluteCorrection",
        "GeoReferencing",
        "Rad2RefTOA",
        "SaveIntermediateResults",
        "BCOutputGeneratedBandsPath",
        "RCOutputGeneratedBandsPath",
        "GridGeolocationPath",
    ]

    rows = []
    for k in keys:
        rows.append({
            "product_id": _product_id_from_name(root.name),
            "level": _product_level_from_name(root.name),
            "key": k,
            "value": cfg.get(k, None),
        })

    return pd.DataFrame(rows)


def raster_inventory(product: Any) -> pd.DataFrame:
    root = _as_path(product)
    rows = []

    for p in sorted(root.rglob("*.tif*")):
        rel = p.relative_to(root)
        row = {
            "product_id": _product_id_from_name(root.name),
            "level": _product_level_from_name(root.name),
            "relative_path": str(rel),
            "family": classify_product_file(rel),
            "size_mb": p.stat().st_size / 1024 / 1024,
        }

        if rasterio is not None:
            try:
                with rasterio.open(p) as src:
                    row.update({
                        "width": src.width,
                        "height": src.height,
                        "count": src.count,
                        "dtype": "|".join(src.dtypes),
                        "crs": str(src.crs) if src.crs else "",
                        "descriptions": "|".join(str(x) for x in src.descriptions),
                    })
            except Exception as e:
                row["raster_error"] = f"{type(e).__name__}: {e}"

        rows.append(row)

    return pd.DataFrame(rows)


def product_card(product: Any) -> dict:
    root = _as_path(product)
    manifest = file_manifest(root)
    rasters = raster_inventory(root)
    switches = processing_switches(root)

    families = set(manifest["family"].tolist()) if not manifest.empty else set()

    return {
        "product_id": _product_id_from_name(root.name),
        "level": _product_level_from_name(root.name),
        "folder": str(root),
        "n_files": int(len(manifest)),
        "total_mb": round(float(manifest["size_mb"].sum()), 3) if not manifest.empty else 0.0,
        "families": sorted(families),
        "has_bands": (root / "bands").exists(),
        "has_geolocation": (root / "geolocation").exists(),
        "has_processing_config": (root / "processing_config.json").exists(),
        "has_AOCS": (root / "AOCS.json").exists(),
        "has_tiling_config": (root / "TilingConfiguration.json").exists(),
        "n_rasters": int(len(rasters)),
        "crs_values": sorted(set(x for x in rasters.get("crs", pd.Series(dtype=str)).tolist() if x)),
        "processing_switches": {
            str(r["key"]): r["value"]
            for _, r in switches.iterrows()
            if r["value"] is not None
        },
    }


def compare_product_folders(left: Any, right: Any) -> dict:
    left_root = _as_path(left)
    right_root = _as_path(right)

    left_manifest = file_manifest(left_root)
    right_manifest = file_manifest(right_root)

    left_files = set(left_manifest["relative_path"].tolist())
    right_files = set(right_manifest["relative_path"].tolist())

    left_families = set(left_manifest["family"].tolist())
    right_families = set(right_manifest["family"].tolist())

    return {
        "left": product_card(left_root),
        "right": product_card(right_root),
        "common_files": sorted(left_files & right_files),
        "left_only_files": sorted(left_files - right_files),
        "right_only_files": sorted(right_files - left_files),
        "common_families": sorted(left_families & right_families),
        "left_only_families": sorted(left_families - right_families),
        "right_only_families": sorted(right_families - left_families),
    }
