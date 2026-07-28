from __future__ import annotations

from typing import Any
import pandas as pd

from phiesta.products.anatomy import product_card, raster_inventory, processing_switches


PHISAT2_IMAGE_WIDTH_PX = 4096
PHISAT2_IMAGE_HEIGHT_PX = 4096
PHISAT2_N_BANDS = 8
PHISAT2_GSD_M = 4.75
PHISAT2_SWATH_KM = 19.4

# Zero-based order used by Phiesta product stacks:
# [PAN, MS1, MS2, MS3, MS4, MS5, MS6, MS7]
PHISAT2_BANDS = [
    {"band_index": 0, "name": "PAN", "central_wavelength_nm": 625, "bandwidth_nm": 250, "role": "panchromatic"},
    {"band_index": 1, "name": "MS1", "central_wavelength_nm": 490, "bandwidth_nm": 65, "role": "blue"},
    {"band_index": 2, "name": "MS2", "central_wavelength_nm": 560, "bandwidth_nm": 35, "role": "green"},
    {"band_index": 3, "name": "MS3", "central_wavelength_nm": 665, "bandwidth_nm": 30, "role": "red"},
    {"band_index": 4, "name": "MS4", "central_wavelength_nm": 705, "bandwidth_nm": 15, "role": "red-edge"},
    {"band_index": 5, "name": "MS5", "central_wavelength_nm": 740, "bandwidth_nm": 15, "role": "red-edge"},
    {"band_index": 6, "name": "MS6", "central_wavelength_nm": 783, "bandwidth_nm": 20, "role": "red-edge/nir"},
    {"band_index": 7, "name": "MS7", "central_wavelength_nm": 842, "bandwidth_nm": 115, "role": "nir"},
]


PHISAT2_PRODUCT_LEVELS = {
    "L1A": {
        "radiometry": "toa_radiance",
        "geometry": "sensor_geometry",
        "expected_georeferenced": False,
        "expected_band_alignment": False,
        "expected_orthorectified": False,
        "expected_rmse_m": 400,
        "description": "Top of Atmosphere radiance in sensor geometry; no fine georeferencing; no band-to-band alignment.",
    },
    "L1B": {
        "radiometry": "toa_radiance",
        "geometry": "sensor_geometry",
        "expected_georeferenced": True,
        "expected_band_alignment": True,
        "expected_orthorectified": False,
        "expected_rmse_m": 10,
        "alignment_reference_band": "MS3",
        "alignment_reference_band_index": 3,
        "description": "Top of Atmosphere radiance in sensor geometry; fine georeferenced; fine band-to-band alignment.",
    },
    "L1C": {
        "radiometry": "toa_reflectance",
        "geometry": "sensor_geometry",
        "expected_georeferenced": True,
        "expected_band_alignment": True,
        "expected_orthorectified": False,
        "expected_rmse_m": 10,
        "alignment_reference_band": "MS3",
        "alignment_reference_band_index": 3,
        "description": "Top of Atmosphere reflectance in sensor geometry; fine georeferenced; fine band-to-band alignment; not orthorectified.",
    },
}


def phisat2_band_table() -> pd.DataFrame:
    return pd.DataFrame(PHISAT2_BANDS)


def phisat2_product_level_specs() -> pd.DataFrame:
    rows = []
    for level, spec in PHISAT2_PRODUCT_LEVELS.items():
        row = {"level": level}
        row.update(spec)
        rows.append(row)
    return pd.DataFrame(rows)


def _main_multiband_rows(rasters: pd.DataFrame) -> pd.DataFrame:
    if rasters.empty or "family" not in rasters:
        return pd.DataFrame()
    return rasters[rasters["family"].isin(["BC_multiband", "RC_multiband"])].copy()


def _switch_dict(product: Any) -> dict[str, Any]:
    try:
        df = processing_switches(product)
        return {str(r["key"]): r["value"] for _, r in df.iterrows()}
    except Exception:
        return {}


def _observed_raster_summary(rasters: pd.DataFrame) -> dict[str, Any]:
    if rasters.empty:
        return {
            "n_rasters": 0,
            "multiband_shapes": [],
            "max_band_count": None,
            "crs_values": [],
        }

    shapes = []
    mb = _main_multiband_rows(rasters)

    for _, r in mb.iterrows():
        if all(k in r for k in ["width", "height", "count"]):
            shapes.append({
                "family": r.get("family"),
                "relative_path": r.get("relative_path"),
                "width": None if pd.isna(r.get("width")) else int(r.get("width")),
                "height": None if pd.isna(r.get("height")) else int(r.get("height")),
                "count": None if pd.isna(r.get("count")) else int(r.get("count")),
                "crs": r.get("crs", ""),
            })

    crs_values = []
    if "crs" in rasters:
        crs_values = sorted(set(str(x) for x in rasters["crs"].tolist() if str(x)))

    max_band_count = None
    if "count" in rasters:
        vals = [int(x) for x in rasters["count"].tolist() if pd.notna(x)]
        max_band_count = max(vals) if vals else None

    return {
        "n_rasters": int(len(rasters)),
        "multiband_shapes": shapes,
        "max_band_count": max_band_count,
        "crs_values": crs_values,
    }


def mission_spec_report(product: Any) -> dict[str, Any]:
    """
    Compare an observed PhiSat-2 product against mission-level product expectations.

    This is a mission-aware consistency report, not a certification tool.
    It uses public mission/product definitions and observed product metadata.
    """
    card = product_card(product)
    rasters = raster_inventory(product)
    switches = _switch_dict(product)

    level = card.get("level")
    spec = PHISAT2_PRODUCT_LEVELS.get(level, None)

    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool | None, observed: Any, expected: Any, note: str = ""):
        checks.append({
            "name": name,
            "ok": ok,
            "observed": observed,
            "expected": expected,
            "note": note,
        })

    observed_rasters = _observed_raster_summary(rasters)

    if spec is None:
        add_check(
            "known_product_level",
            False,
            level,
            sorted(PHISAT2_PRODUCT_LEVELS),
            "No mission-level specification encoded for this level.",
        )
    else:
        add_check(
            "georeferencing_presence",
            bool(card.get("has_geolocation")) == bool(spec["expected_georeferenced"]),
            bool(card.get("has_geolocation")),
            bool(spec["expected_georeferenced"]),
            "Checks presence of geolocation folder/output, not geolocation accuracy.",
        )

        add_check(
            "crs_presence",
            bool(card.get("crs_values")) == bool(spec["expected_georeferenced"]),
            card.get("crs_values"),
            "present" if spec["expected_georeferenced"] else "absent",
            "Checks whether rasters expose CRS metadata.",
        )

        if level == "L1C":
            add_check(
                "radiance_to_reflectance_switch",
                switches.get("Rad2RefTOA") is True,
                switches.get("Rad2RefTOA"),
                True,
                "L1C is expected to be Top-of-Atmosphere reflectance.",
            )

            add_check(
                "not_orthorectified",
                True,
                "sensor_geometry",
                "not orthorectified",
                "PhiSat-2 L1C is defined in sensor geometry, unlike Sentinel-2 L1C nomenclature.",
            )

        if level == "L1A":
            add_check(
                "no_fine_band_alignment_expected",
                switches.get("BandCoregistration") in [0, False, None],
                switches.get("BandCoregistration"),
                "disabled or absent",
                "L1A is expected to have no fine band-to-band alignment.",
            )

        if level in {"L1B", "L1C"}:
            add_check(
                "fine_band_alignment_expected",
                switches.get("BandCoregistration") not in [0, False, None],
                switches.get("BandCoregistration"),
                "enabled",
                "L1B/L1C are expected to have fine band-to-band alignment.",
            )

    multiband_shapes = observed_rasters["multiband_shapes"]
    if multiband_shapes:
        full_image_ok = any(
            s["width"] == PHISAT2_IMAGE_WIDTH_PX and
            s["height"] == PHISAT2_IMAGE_HEIGHT_PX
            for s in multiband_shapes
        )
        eight_band_ok = any(s["count"] == PHISAT2_N_BANDS for s in multiband_shapes)

        add_check(
            "full_image_shape",
            full_image_ok,
            multiband_shapes,
            {"width": PHISAT2_IMAGE_WIDTH_PX, "height": PHISAT2_IMAGE_HEIGHT_PX},
            "If false, the product may be a crop, tile, or non-standard raster.",
        )

        add_check(
            "eight_band_stack",
            eight_band_ok,
            [s["count"] for s in multiband_shapes],
            PHISAT2_N_BANDS,
            "Expected PAN + 7 multispectral bands.",
        )
    else:
        add_check(
            "multiband_raster_present",
            False,
            [],
            "BC_multiband or RC_multiband",
            "No multiband raster was found.",
        )

    return {
        "product_id": card.get("product_id"),
        "level": level,
        "folder": card.get("folder"),
        "mission_constants": {
            "image_width_px": PHISAT2_IMAGE_WIDTH_PX,
            "image_height_px": PHISAT2_IMAGE_HEIGHT_PX,
            "n_bands": PHISAT2_N_BANDS,
            "gsd_m": PHISAT2_GSD_M,
            "nominal_square_image_width_km": PHISAT2_SWATH_KM,
        },
        "spec": spec,
        "observed": {
            "product_card": card,
            "raster_summary": observed_rasters,
            "processing_switches": switches,
        },
        "checks": checks,
        "overall_ok": all(c["ok"] is True for c in checks if c["ok"] is not None),
        "note": "Mission-aware consistency report; not a certified product-quality or geolocation-accuracy assessment.",
    }
