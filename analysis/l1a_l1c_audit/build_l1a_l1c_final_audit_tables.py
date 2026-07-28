from pathlib import Path
import json
import os
import pandas as pd
import rasterio

PRODUCT_IDS = ["6008", "6025", "6038", "6041"]
ROOT = Path("outputs/l1a_l1c_audit_selected")
OUT = ROOT / "final_audit_tables"
OUT.mkdir(parents=True, exist_ok=True)

LEVELS = [
    ("L1A", "data/l1a", "L1A"),
    ("L1C", "data/l1", "L1"),
]

KEYS_OF_INTEREST = [
    "BandCoregistration",
    "CosmeticFilling",
    "Denoising",
    "AbsoluteCorrection",
    "GeoReferencing",
    "Rad2RefTOA",
    "TOAConversion",
    "DarkCurrentCorrection",
    "BadPixelCorrection",
    "RadiometricCorrection",
    "RelativeCorrection",
]

def find_product(root, level_pattern, pid):
    matches = sorted(Path(root).glob(f"PHISAT-2_{level_pattern}_{int(pid):09d}_*"))
    if not matches:
        return None
    return matches[0]

def folder_size_mb(path):
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / 1024 / 1024

def rel_files(path):
    return [str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()]

def tif_meta(path):
    if not path.exists():
        return {}
    with rasterio.open(path) as src:
        return {
            "exists": True,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": ",".join(src.dtypes),
            "crs": str(src.crs) if src.crs else "",
            "descriptions": "|".join([str(x) for x in src.descriptions]),
        }

def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def find_key_recursive(obj, target):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target:
                found.append(v)
            found.extend(find_key_recursive(v, target))
    elif isinstance(obj, list):
        for x in obj:
            found.extend(find_key_recursive(x, target))
    return found

def first_value(obj, key):
    vals = find_key_recursive(obj, key)
    if not vals:
        return ""
    v = vals[0]
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v

structure_rows = []
config_rows = []
raster_rows = []

for pid in PRODUCT_IDS:
    for level, root, pattern in LEVELS:
        product = find_product(root, pattern, pid)
        if product is None:
            structure_rows.append({
                "product_id": pid,
                "level": level,
                "exists": False,
            })
            continue

        files = rel_files(product)
        tifs = [f for f in files if f.lower().endswith((".tif", ".tiff"))]

        structure_rows.append({
            "product_id": pid,
            "level": level,
            "exists": True,
            "product_folder": str(product),
            "size_mb": round(folder_size_mb(product), 2),
            "n_files": len(files),
            "n_tifs": len(tifs),
            "has_bands_dir": (product / "bands").exists(),
            "has_geolocation_dir": (product / "geolocation").exists(),
            "has_logs_dir": (product / "logs").exists(),
            "has_processing_config": (product / "processing_config.json").exists(),
            "has_aocs": (product / "AOCS.json").exists(),
            "has_tiling_config": (product / "TilingConfiguration.json").exists(),
            "has_BC_multiband": (product / "bands" / "scene_0_BC_multiband.tiff").exists(),
            "has_BC_RGB": (product / "bands" / "scene_0_BC_RGB.tiff").exists(),
            "has_RC_multiband": (product / "bands" / "scene_0_RC_multiband.tiff").exists(),
            "has_RC_RGB": (product / "bands" / "scene_0_RC_RGB.tiff").exists(),
            "has_GL_scene_0_json": (product / "geolocation" / "GL_scene_0.json").exists(),
        })

        cfg = load_json(product / "processing_config.json")
        row = {
            "product_id": pid,
            "level": level,
            "processing_config": str(product / "processing_config.json"),
        }
        for k in KEYS_OF_INTEREST:
            row[k] = first_value(cfg, k)
        config_rows.append(row)

        for name, rel in [
            ("BC_multiband", "bands/scene_0_BC_multiband.tiff"),
            ("BC_RGB", "bands/scene_0_BC_RGB.tiff"),
            ("RC_multiband", "bands/scene_0_RC_multiband.tiff"),
            ("RC_RGB", "bands/scene_0_RC_RGB.tiff"),
        ]:
            meta = tif_meta(product / rel)
            r = {
                "product_id": pid,
                "level": level,
                "raster": name,
                "path": str(product / rel),
            }
            r.update(meta)
            raster_rows.append(r)

structure = pd.DataFrame(structure_rows)
config = pd.DataFrame(config_rows)
raster = pd.DataFrame(raster_rows)

structure.to_csv(OUT / "product_structure_summary.csv", index=False)
config.to_csv(OUT / "processing_config_summary.csv", index=False)
raster.to_csv(OUT / "raster_metadata_summary.csv", index=False)

print("\n===== PRODUCT STRUCTURE =====")
print(structure.to_string(index=False))

print("\n===== PROCESSING CONFIG =====")
print(config.to_string(index=False))

print("\n===== RASTER METADATA =====")
print(raster.to_string(index=False))

# Geometry summaries already generated
geometry_files = [
    ROOT / "final_clean_raw_local_shift_summary.csv",
    ROOT / "final_clean_local_residual_after_global_shift_summary.csv",
    ROOT / "final_per_product_shift_vs_residual.csv",
]

for p in geometry_files:
    if p.exists():
        target = OUT / p.name
        target.write_text(p.read_text())

# Simple markdown synthesis
md = OUT / "AUDIT_SYNTHESIS.md"
md.write_text(
"""# ΦSat-2 L1A vs L1C audit synthesis

## A. Product structure

L1A and L1C are not organized as equivalent products.

Main checks are in:

- `product_structure_summary.csv`
- `raster_metadata_summary.csv`

Expected pattern:

- L1A contains both RC and BC raster families.
- L1C focuses on BC outputs and includes geolocation products.
- L1C products expose georeferenced raster metadata, while L1A rasters are not directly georeferenced in the same way.

## B. Processing configuration

Main checks are in:

- `processing_config_summary.csv`

Expected pattern:

- L1A disables several post-processing steps.
- L1C activates core processing steps such as band coregistration, absolute correction, georeferencing, and TOA reflectance conversion when present in the product config.

This means L1C is not just a relabelled L1A product.

## C. Geometry / inter-band registration

Main checks are in:

- `final_clean_raw_local_shift_summary.csv`
- `final_clean_local_residual_after_global_shift_summary.csv`
- `final_per_product_shift_vs_residual.csv`

Clean set: 6008, 6025, 6038, 6041.

Main result:

- L1A has large raw inter-band shifts.
- On clean scenes, most of the L1A shift is explained by a near-constant band-dependent translation.
- L1C removes this dominant offset and brings residual inter-band displacement close to pixel level.

For band 6 relative to band 2, the patch maps show L1A offsets around 67–73 px in clean scenes, while L1C is typically around 1–2 px.

## D. Practical implication

L1A can be exposed and loaded in Phiesta, but should not be treated as an analysis-ready multispectral stack equivalent to L1C. L1C should remain the default product for multispectral operations requiring band-to-band geometric consistency.
""",
encoding="utf-8"
)

print("\nwrote", OUT)
print("wrote", md)
