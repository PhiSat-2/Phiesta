from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyproj
import rasterio
from rasterio.transform import rowcol

from ..remote.catalog_geometry import get_catalog_corners


def _event_shape(event: Any) -> tuple[int, int]:
    arr = getattr(event, "_arr", None)
    if arr is None:
        raise AttributeError("Event has no _arr attribute.")
    return int(arr.shape[1]), int(arr.shape[2])  # H, W


def _order_quad_tl_tr_br_bl(points_xy: np.ndarray) -> np.ndarray:
    """
    Order 4 points in image coordinates as TL, TR, BR, BL.
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected 4 points, got {pts.shape}")

    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _catalog_corners_to_raster_pixels(event: Any, raster_crs, raster_transform) -> np.ndarray:
    """
    Convert event catalog lon/lat corners to pixel coordinates in a raster.
    """
    corners_lonlat = get_catalog_corners(event, order="lonlat")
    if not corners_lonlat or len(corners_lonlat) != 4:
        raise ValueError("Could not retrieve 4 catalog corners from event.catalog_geo.")

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",
        raster_crs,
        always_xy=True,
    )

    pixels = []
    for lon, lat in corners_lonlat:
        x, y = transformer.transform(lon, lat)
        r, c = rowcol(raster_transform, x, y)
        pixels.append([float(c), float(r)])  # x=col, y=row

    return _order_quad_tl_tr_br_bl(np.array(pixels, dtype=np.float32))


def rectify_simulated_catalog_crop(
    event: Any,
    simulated_path: str | Path,
    output_dir: str | Path,
    output_shape: tuple[int, int] | None = None,
    flip_horizontal: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
    overwrite: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Rectify the simulated PhiSat-2 crop to the catalog footprint frame.

    Input:
        simulated_path: big simulated crop built from Sentinel-2 + buffer.

    Output:
        an 8-band GeoTIFF with same pixel shape as real PhiSat-2, typically 4096x4096.
    """
    simulated_path = Path(simulated_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_shape is None:
        out_h, out_w = _event_shape(event)
    else:
        out_h, out_w = int(output_shape[0]), int(output_shape[1])

    output_path = output_dir / f"{simulated_path.stem}_catalog_rectified.tif"


    with rasterio.open(simulated_path) as src:
        sim = src.read().astype(np.float32)
        profile = src.profile.copy()
        descriptions = src.descriptions

        src_quad = _catalog_corners_to_raster_pixels(
            event=event,
            raster_crs=src.crs,
            raster_transform=src.transform,
        )

    if flip_horizontal:
        dst_quad = np.array(
            [
                [out_w - 1, 0],
                [0, 0],
                [0, out_h - 1],
                [out_w - 1, out_h - 1],
            ],
            dtype=np.float32,
        )
    else:
        dst_quad = np.array(
            [
                [0, 0],
                [out_w - 1, 0],
                [out_w - 1, out_h - 1],
                [0, out_h - 1],
            ],
            dtype=np.float32,
        )

    H = cv2.getPerspectiveTransform(src_quad, dst_quad)

    rectified = np.zeros((sim.shape[0], out_h, out_w), dtype=np.float32)

    for b in range(sim.shape[0]):
        rectified[b] = cv2.warpPerspective(
            sim[b],
            H,
            dsize=(out_w, out_h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # This output is primarily in real PhiSat-2 pixel frame.
    # Reuse event georeference if available, but the important part is pixel alignment.
    event_meta = getattr(event, "meta", getattr(event, "_meta", {}))
    profile.update(
        driver="GTiff",
        height=out_h,
        width=out_w,
        count=rectified.shape[0],
        dtype="float32",
        crs=event_meta.get("crs", profile.get("crs")),
        transform=event_meta.get("transform", profile.get("transform")),
        compress="deflate",
        bigtiff="if_safer",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(rectified)
        if descriptions:
            dst.descriptions = descriptions

    if verbose:
        print("[Phiesta] Rectified simulated catalog crop")
        print(f"[Phiesta] source: {simulated_path}")
        print(f"[Phiesta] output: {output_path}")
        print(f"[Phiesta] output_shape: {(out_h, out_w)}")
        print(f"[Phiesta] flip_horizontal: {flip_horizontal}")
        print(f"[Phiesta] src_quad_px:\n{src_quad}")

    return {
        "status": "SUCCESS",
        "rectified_path": str(output_path),
        "src_quad_px": src_quad.tolist(),
        "homography": H.tolist(),
        "flip_horizontal": flip_horizontal,
        "output_shape": (out_h, out_w),
    }