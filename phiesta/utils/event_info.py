from __future__ import annotations

import re
from pathlib import Path
from typing import Any


BAND_ALIASES = {
    0: "PAN",
    1: "BLUE",
    2: "GREEN",
    3: "RED",
    4: "RE1",
    5: "RE2",
    6: "RE3",
    7: "NIR",
}


def _safe_meta(event) -> dict:
    if hasattr(event, "meta"):
        try:
            return event.meta or {}
        except Exception:
            pass
    return getattr(event, "_meta", {}) or {}


def _safe_array(event):
    for name in ("arr", "_arr", "data", "_data", "cube", "_cube"):
        if hasattr(event, name):
            try:
                value = getattr(event, name)
                if hasattr(value, "shape"):
                    return value
            except Exception:
                pass
    return None


def _short(value: Any, max_len: int = 120) -> str:
    if value is None or value == "":
        return "n/a"
    s = str(value)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _product_id_from_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value)

    match = re.search(r"PHISAT-2_L[01]_0*(\d+)_", text)
    if match:
        return str(int(match.group(1)))

    match = re.search(r"\b0*(\d{3,8})\b", text)
    if match:
        return str(int(match.group(1)))

    return None


def _guess_product_id(meta: dict, event) -> str:
    candidates = [
        meta.get("identifier"),
        meta.get("insula_product_identifier"),
        meta.get("insula_filename"),
        meta.get("filename"),
        meta.get("resolved_product_folder"),
        meta.get("path"),
        getattr(event, "_product_folder", None),
    ]

    for value in candidates:
        product_id = _product_id_from_text(value)
        if product_id:
            return product_id

    return "n/a"


def _guess_filename(meta: dict, event) -> str:
    candidates = [
        meta.get("insula_filename"),
        meta.get("filename"),
        meta.get("insula_product_identifier"),
        meta.get("resolved_product_folder"),
        meta.get("path"),
        getattr(event, "_product_folder", None),
    ]

    for value in candidates:
        if value:
            name = Path(str(value)).name
            if name:
                return name

    return "n/a"


def _shape_info(event, meta: dict) -> tuple[Any, Any, Any]:
    arr = _safe_array(event)

    count = meta.get("count")
    height = meta.get("height")
    width = meta.get("width")

    if arr is not None and hasattr(arr, "shape"):
        if len(arr.shape) == 3:
            count = count or arr.shape[0]
            height = height or arr.shape[1]
            width = width or arr.shape[2]
        elif len(arr.shape) == 2:
            height = height or arr.shape[0]
            width = width or arr.shape[1]

    return count, height, width


def _array_memory(event) -> str:
    arr = _safe_array(event)
    if arr is None:
        return "n/a"

    try:
        return f"{arr.nbytes / 1024 / 1024:.1f} MB"
    except Exception:
        return "n/a"


def _format_transform(transform: Any) -> str:
    if transform is None:
        return "n/a"

    try:
        values = tuple(transform)[:6]
        return "(" + ", ".join(f"{float(v):.6g}" for v in values) + ")"
    except Exception:
        return _short(transform, max_len=160)


def _print_section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def _print_kv(key: str, value: Any, indent: int = 2, max_len: int = 120) -> None:
    print(" " * indent + f"{key:<24}: {_short(value, max_len=max_len)}")


def _print_band_table(event, meta: dict) -> None:
    count, _, _ = _shape_info(event, meta)
    wavelengths = meta.get("band_wavelength_nm") or []

    try:
        count = int(count)
    except Exception:
        count = len(wavelengths)

    print("  index  alias        wavelength_nm")
    print("  -----  -----------  -------------")

    for idx in range(count):
        alias = BAND_ALIASES.get(idx, f"BAND_{idx}")
        wavelength = wavelengths[idx] if idx < len(wavelengths) else "n/a"
        print(f"  {idx:<5}  {alias:<11}  {wavelength}")


def _print_optional_stats(event, show_stats: bool, stats_sample: int | None) -> None:
    if not show_stats:
        return

    if not hasattr(event, "band_stats"):
        print("  band_stats() not available for this event.")
        return

    print()
    print("  Display stats, sampled:")
    try:
        stats = event.band_stats(
            bands=tuple(BAND_ALIASES.values()),
            percentiles=(1, 50, 99),
            sample=stats_sample,
        )

        for band_name, values in stats.items():
            p1 = values.get("p1", "n/a")
            p50 = values.get("p50", "n/a")
            p99 = values.get("p99", "n/a")
            print(f"  {band_name:<24} p1={p1:<8} p50={p50:<8} p99={p99:<8}")

    except Exception as exc:
        print(f"  Could not compute stats: {type(exc).__name__}: {exc}")


def show_event_info(
    event,
    *,
    show_stats: bool = False,
    stats_sample: int | None = 1_000_000,
) -> None:
    """
    Print a useful overview of a Phiesta event.

    This user-facing helper shows product identity, raster metadata, band
    information, Insula catalog geometry, local/remote paths, and common API
    calls.
    """
    meta = _safe_meta(event)
    arr = _safe_array(event)

    klass = type(event).__name__
    product_id = _guess_product_id(meta, event)
    filename = _guess_filename(meta, event)
    count, height, width = _shape_info(event, meta)

    _print_section("Phiesta product overview")
    _print_kv("object", klass)
    _print_kv("product_id", product_id)
    _print_kv("filename", filename)
    _print_kv("product_kind", meta.get("product_kind") or getattr(event, "_product_kind", None))
    _print_kv("scene_id", meta.get("scene_id") or getattr(event, "_scene_id", None))
    _print_kv("source", meta.get("source"))
    _print_kv("sensing_time", meta.get("sensing_time"))
    _print_kv("creation_time", meta.get("creation_time"))

    _print_section("Raster")
    _print_kv("shape", f"({count}, {height}, {width})")
    _print_kv("dtype", meta.get("dtype") or getattr(arr, "dtype", None))
    _print_kv("array_memory", _array_memory(event))
    _print_kv("crs", meta.get("crs"))
    _print_kv("transform", _format_transform(meta.get("transform")), max_len=180)

    _print_section("Bands")
    _print_band_table(event, meta)
    _print_optional_stats(event, show_stats=show_stats, stats_sample=stats_sample)

    _print_section("Catalog geometry from Insula")
    catalog_geo = meta.get("catalog_geo")

    if not isinstance(catalog_geo, dict):
        print("  No catalog geometry found in metadata.")
    else:
        _print_kv("center_lonlat", catalog_geo.get("center_lonlat"), max_len=180)
        _print_kv("center_latlon", catalog_geo.get("center_latlon"), max_len=180)
        _print_kv("geometry_type", catalog_geo.get("geometry_type"))
        _print_kv("start_datetime", catalog_geo.get("start_datetime"))
        _print_kv("completion_datetime", catalog_geo.get("completion_datetime"))

        corners = catalog_geo.get("corners_lonlat") or []
        if corners:
            print("  corners_lonlat:")
            for idx, corner in enumerate(corners, start=1):
                print(f"    {idx}: {corner}")
        else:
            print("  corners_lonlat        : n/a")

    _print_section("Local / remote paths")
    printed_any_path = False
    for key in [
        "resolved_product_folder",
        "path",
        "session_metadata_path",
        "gl_path",
        "processing_config_path",
        "insula_platform_url",
        "insula_download_url",
    ]:
        value = meta.get(key)
        if value:
            printed_any_path = True
            _print_kv(key, value, max_len=160)

    if not printed_any_path:
        print("  No local or remote path found in metadata.")

    _print_section("Useful calls")
    if klass.startswith("L1"):
        print('  event.show_all_bands(normalization="percentile", percentiles=(1, 99))')
        print('  event.show_rgb(bands=("RED", "GREEN", "BLUE"), per_band=True)')
        print('  event.show_rgb(bands=("NIR", "RED", "GREEN"), registered=True, registration_master="NIR")')
        print('  event.show_band("NIR", normalization="percentile", percentiles=(1, 99))')
        print("  event.band_stats(...)")
        print("  event.plot_distribution(...)")
        print("  event.plot_display_diagnostics()")
        print('  event.compare_display_stretches(bands=("NIR", "RED", "GREEN"))')
        print("  event.build_full_sentinel_triplet()")
    elif klass.startswith("L0"):
        print('  event.show_all_bands(normalization="percentile", percentiles=(1, 99))')
        print('  event.show_rgb(bands=("RED", "GREEN", "BLUE"), per_band=True)')
        print('  event.show_band("NIR", normalization="percentile", percentiles=(1, 99))')
    else:
        print("  dir(event)")

    print()
