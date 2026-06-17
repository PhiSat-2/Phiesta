from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


def _extract_product_id(value: Any) -> str:
    if value is None:
        return "unknown"

    s = str(value).strip()

    # Common case: 5359
    if s.isdigit():
        return str(int(s))

    # Common case: PHISAT-2_L1_000005359_...
    m = re.search(r"PHISAT-2_L[01]_0*(\d+)_", s)
    if m:
        return str(int(m.group(1)))

    return s


def crop_sentinel_window(
    sentinel_big_crop_path: str | Path,
    metadata_path: str | Path,
    window_native: dict[str, int],
    output_dir: str | Path,
    product_id: str | int | None = None,
    overwrite: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Crop the big Sentinel-2 7-band crop to the final native Sentinel window.

    The output crop stays at Sentinel-2 resolution, usually 10 m, and keeps
    the 7-band order:
        BLUE, GREEN, RED, NIR_BROAD, RED_EDGE_1, RED_EDGE_2, RED_EDGE_3
    """
    sentinel_big_crop_path = Path(sentinel_big_crop_path)
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sentinel_big_crop_path.exists():
        raise FileNotFoundError(f"Sentinel big crop not found: {sentinel_big_crop_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Sentinel metadata not found: {metadata_path}")

    pid = _extract_product_id(product_id)

    out_tif = output_dir / f"{pid}_s2b_final_crop_7bands.tif"
    out_json = output_dir / f"{pid}_s2b_final_metadata.json"

    if out_tif.exists() and out_json.exists() and not overwrite:
        if verbose:
            print(f"[PyRawPh] Reusing final Sentinel crop: {out_tif}")

        return {
            "status": "ALREADY_EXISTS",
            "crop_path": str(out_tif),
            "metadata_path": str(out_json),
            "product_id": pid,
            "window_native": dict(window_native),
        }

    x_min = int(window_native["x_min"])
    y_min = int(window_native["y_min"])
    width = int(window_native["width"])
    height = int(window_native["height"])

    with rasterio.open(sentinel_big_crop_path) as src:
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        width = min(width, src.width - x_min)
        height = min(height, src.height - y_min)

        if width <= 0 or height <= 0:
            raise ValueError(
                "Invalid final crop window after clamping: "
                f"x_min={x_min}, y_min={y_min}, width={width}, height={height}"
            )

        window = Window(x_min, y_min, width, height)

        profile = src.profile.copy()
        profile.update(
            height=height,
            width=width,
            transform=window_transform(window, src.transform),
            compress=profile.get("compress", "deflate"),
            bigtiff="if_safer",
        )

        data = src.read(window=window)
        descriptions = src.descriptions

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(data)
        if descriptions:
            dst.descriptions = descriptions

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    final_window = {
        "x_min": int(x_min),
        "y_min": int(y_min),
        "width": int(width),
        "height": int(height),
        "x_max": int(x_min + width),
        "y_max": int(y_min + height),
    }

    meta.update(
        {
            "product_id": pid,
            "source_big_crop_path": str(sentinel_big_crop_path),
            "final_crop_path": str(out_tif),
            "final_metadata_path": str(out_json),
            "window_native": final_window,
            "final_crop_shape": {
                "height": int(height),
                "width": int(width),
                "count": int(data.shape[0]),
            },
            "band_names": list(descriptions) if descriptions else None,
        }
    )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print("[PyRawPh] Final Sentinel crop saved")
        print(f"[PyRawPh] source: {sentinel_big_crop_path}")
        print(f"[PyRawPh] output: {out_tif}")
        print(f"[PyRawPh] metadata: {out_json}")
        print(f"[PyRawPh] shape: count={data.shape[0]}, height={height}, width={width}")

    return {
        "status": "SUCCESS",
        "crop_path": str(out_tif),
        "metadata_path": str(out_json),
        "product_id": pid,
        "window_native": final_window,
        "shape": {
            "count": int(data.shape[0]),
            "height": int(height),
            "width": int(width),
        },
    }
