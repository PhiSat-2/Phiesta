from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine


_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}


def _as_h3x3(value: Any, *, name: str) -> np.ndarray:
    H = np.asarray(value, dtype=np.float64)
    if H.shape == (2, 3):
        H = np.vstack([H, [0.0, 0.0, 1.0]])
    if H.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {H.shape}.")
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def _affine_matrix(transform: Affine) -> np.ndarray:
    return np.array(
        [
            [transform.a, transform.b, transform.c],
            [transform.d, transform.e, transform.f],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _project_points(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    homog = np.concatenate(
        [points_xy, np.ones((len(points_xy), 1), dtype=np.float64)], axis=1
    )
    out = (H @ homog.T).T
    denom = out[:, 2:3]
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError("Homography maps one or more points to infinity.")
    return out[:, :2] / denom


def _estimate_native_resolution(
    H_real_to_map: np.ndarray,
    width: int,
    height: int,
) -> float:
    """Estimate local ground sampling distance near the image centre."""
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    pts = np.array(
        [
            [cx, cy],
            [cx + 1.0, cy],
            [cx, cy + 1.0],
        ],
        dtype=np.float64,
    )
    mapped = _project_points(H_real_to_map, pts)
    dx = float(np.linalg.norm(mapped[1] - mapped[0]))
    dy = float(np.linalg.norm(mapped[2] - mapped[0]))
    values = [v for v in (dx, dy) if np.isfinite(v) and v > 0]
    if not values:
        raise ValueError("Could not estimate output resolution from the georeference.")
    return float(np.mean(values))


def _default_output_path(georef: dict, real_path: Path) -> Path:
    product_id = georef.get("product_id")
    if product_id is None:
        stem = real_path.stem
        product_id = stem.replace("phisat2_real_", "")

    # Standard Phiesta layout:
    # <product>/final_triplet/phisat2_real_4096.tif
    product_root = real_path.parent.parent
    out_dir = product_root / "georeferenced"
    return out_dir / f"phisat2_{product_id}_georeferenced.tif"


def export_georeferenced_tif(
    georef: dict,
    output_path: str | Path | None = None,
    *,
    resolution: float | None = None,
    resampling: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    compress: str = "deflate",
    overwrite: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Export the real PhiSat-2 acquisition as a georeferenced GeoTIFF.

    ``get_georef()`` returns a projective mapping between the real PhiSat-2
    pixel grid and the final georeferenced Sentinel-2 crop. A standard GeoTIFF
    GeoTransform is affine, so the projective correction cannot be represented
    exactly by only replacing the transform on the original raster. This
    function therefore resamples the real PhiSat-2 image onto a regular map
    grid in the Sentinel-2 crop CRS and writes a conventional GeoTIFF that GIS
    software can place directly.

    Parameters
    ----------
    georef:
        Dictionary returned by ``event.get_georef(...)``.
    output_path:
        Destination GeoTIFF. When omitted, Phiesta writes
        ``<product>/georeferenced/phisat2_<id>_georeferenced.tif``.
    resolution:
        Output pixel size in units of the Sentinel CRS. If omitted, Phiesta
        estimates the local PhiSat-2 ground sampling distance from the strict
        projective mapping, preserving approximately native resolution.
    resampling:
        ``"nearest"``, ``"bilinear"`` or ``"cubic"``.
    compress:
        GDAL GeoTIFF compression name, e.g. ``"deflate"``.
    overwrite:
        Replace an existing file when True.

    Returns
    -------
    dict
        Output path, CRS, transform, dimensions, resolution and footprint.
    """
    if not isinstance(georef, dict):
        raise TypeError("georef must be the dict returned by event.get_georef(...).")

    if resampling not in _INTERPOLATION:
        raise ValueError(
            f"Unsupported resampling={resampling!r}. "
            f"Choose one of {sorted(_INTERPOLATION)}."
        )

    paths = georef.get("paths") or {}
    real_value = paths.get("real")
    sentinel_value = paths.get("final_sentinel_crop")
    if real_value is None or sentinel_value is None:
        raise ValueError(
            "georef['paths'] must contain both 'real' and 'final_sentinel_crop'."
        )

    real_path = Path(str(real_value))
    sentinel_path = Path(str(sentinel_value))
    if not real_path.exists():
        raise FileNotFoundError(f"Real PhiSat-2 raster not found: {real_path}")
    if not sentinel_path.exists():
        raise FileNotFoundError(f"Final Sentinel crop not found: {sentinel_path}")

    H_real_to_s2_value = georef.get("H_real_to_s2")
    if H_real_to_s2_value is None:
        H_s2_to_real_value = georef.get("H_s2_to_real")
        if H_s2_to_real_value is None:
            raise ValueError("georef does not contain H_real_to_s2 or H_s2_to_real.")
        H_real_to_s2 = np.linalg.inv(
            _as_h3x3(H_s2_to_real_value, name="H_s2_to_real")
        )
    else:
        H_real_to_s2 = _as_h3x3(H_real_to_s2_value, name="H_real_to_s2")
    H_real_to_s2 = H_real_to_s2 / H_real_to_s2[2, 2]

    with rasterio.open(sentinel_path) as s2_src:
        s2_transform = s2_src.transform
        s2_crs = s2_src.crs

    if s2_crs is None:
        raise ValueError("Final Sentinel crop has no CRS; cannot export a GIS raster.")
    if resolution is None and not s2_crs.is_projected:
        raise ValueError(
            "Automatic native-resolution estimation requires a projected Sentinel CRS. "
            "Pass resolution=... explicitly for a geographic CRS."
        )

    with rasterio.open(real_path) as real_src:
        real_w = int(real_src.width)
        real_h = int(real_src.height)
        count = int(real_src.count)
        dtype = real_src.dtypes[0]
        descriptions = tuple(real_src.descriptions)

        # OpenCV homographies use integer coordinates at pixel centres, while
        # rasterio/GDAL affine transforms use integer coordinates at pixel
        # corners. Insert the half-pixel translation explicitly so the two
        # conventions compose without a systematic 0.5-pixel shift.
        C = np.array(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        A_s2_center = _affine_matrix(s2_transform) @ C
        H_real_to_map = A_s2_center @ H_real_to_s2

        # Use outer image edges rather than only pixel centres so the complete
        # source raster is covered by the destination grid.
        real_edges = np.array(
            [
                [-0.5, -0.5],
                [real_w - 0.5, -0.5],
                [real_w - 0.5, real_h - 0.5],
                [-0.5, real_h - 0.5],
            ],
            dtype=np.float64,
        )
        footprint_map = _project_points(H_real_to_map, real_edges)

        if resolution is None:
            resolution = _estimate_native_resolution(
                H_real_to_map,
                width=real_w,
                height=real_h,
            )
        resolution = float(resolution)
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be a finite positive number.")

        min_x = float(np.min(footprint_map[:, 0]))
        max_x = float(np.max(footprint_map[:, 0]))
        min_y = float(np.min(footprint_map[:, 1]))
        max_y = float(np.max(footprint_map[:, 1]))

        out_w = max(1, int(math.ceil((max_x - min_x) / resolution)))
        out_h = max(1, int(math.ceil((max_y - min_y) / resolution)))
        out_transform = Affine(resolution, 0.0, min_x, 0.0, -resolution, max_y)

        # real pixel centre -> map -> output pixel centre
        A_out_center = _affine_matrix(out_transform) @ C
        A_out_center_inv = np.linalg.inv(A_out_center)
        H_real_to_out = A_out_center_inv @ H_real_to_map
        H_real_to_out = H_real_to_out / H_real_to_out[2, 2]

        if output_path is None:
            out_path = _default_output_path(georef, real_path)
        else:
            out_path = Path(output_path)
        out_path = out_path.expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {out_path}. Pass overwrite=True to replace it."
            )

        profile = real_src.profile.copy()
        profile.update(
            driver="GTiff",
            width=out_w,
            height=out_h,
            count=count,
            dtype=dtype,
            crs=s2_crs,
            transform=out_transform,
            compress=compress,
            BIGTIFF="IF_SAFER",
        )
        # Tiny rasters used in tests cannot use arbitrary inherited TIFF block
        # sizes. Enable tiling only when a valid 16-pixel block can be formed.
        if out_w >= 16 and out_h >= 16:
            profile["tiled"] = True
            profile["blockxsize"] = min(256, max(16, (out_w // 16) * 16))
            profile["blockysize"] = min(256, max(16, (out_h // 16) * 16))
        else:
            profile.pop("tiled", None)
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        profile.pop("nodata", None)

        interpolation = _INTERPOLATION[resampling]

        with rasterio.open(out_path, "w", **profile) as dst:
            for band_index in range(1, count + 1):
                band = real_src.read(band_index)
                warped = cv2.warpPerspective(
                    band,
                    H_real_to_out,
                    (out_w, out_h),
                    flags=interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                dst.write(warped, band_index)
                description = descriptions[band_index - 1]
                if description:
                    dst.set_band_description(band_index, description)

            # Keep valid-data semantics independent from pixel value 0.
            src_mask = np.full((real_h, real_w), 255, dtype=np.uint8)
            dst_mask = cv2.warpPerspective(
                src_mask,
                H_real_to_out,
                (out_w, out_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            dst.write_mask(dst_mask)
            dst.update_tags(
                PHIESTA_GEOREF_METHOD=str(georef.get("method", "sentinel_strict")),
                PHIESTA_GEOREF_QUALITY=str(georef.get("quality", "unknown")),
                PHIESTA_PRODUCT_ID=str(georef.get("product_id", "")),
                PHIESTA_SOURCE_REAL=str(real_path),
                PHIESTA_SOURCE_SENTINEL=str(sentinel_path),
            )

    result = {
        "status": "SUCCESS",
        "path": str(out_path),
        "crs": str(s2_crs),
        "transform": tuple(out_transform),
        "width": int(out_w),
        "height": int(out_h),
        "count": int(count),
        "dtype": str(dtype),
        "resolution": float(resolution),
        "resampling": resampling,
        "footprint_projected": footprint_map.tolist(),
        "H_real_to_output": H_real_to_out.tolist(),
    }

    if verbose:
        print("[Phiesta] Exported corrected PhiSat-2 GeoTIFF")
        print(f"[Phiesta] path={out_path}")
        print(f"[Phiesta] crs={s2_crs}")
        print(f"[Phiesta] resolution={resolution:.4f}")
        print(f"[Phiesta] shape=({out_h}, {out_w})")

    return result
