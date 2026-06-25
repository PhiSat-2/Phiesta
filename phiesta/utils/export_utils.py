from __future__ import annotations

from typing import Any, Dict, Optional
import os
import numpy as np
import rasterio
import tifffile


def export_plain_tif(arr: np.ndarray, out_path: str) -> None:
    """
    Write an array to a plain TIFF file without geospatial metadata.

    Args:
        arr: Array to write.
        out_path: Destination path of the TIFF file.

    Returns:
        None.
    """
    tifffile.imwrite(out_path, arr)
    
def export_to_tif(
    out_path: str,
    arr: np.ndarray,
    meta: Dict[str, Any],
    dtype: Optional[str] = None,
    nodata: Optional[float] = None,
    compress: str = "deflate",
    bigtiff: str = "if_safer",
) -> str:
    """
    Export an array to GeoTIFF using geospatial metadata.

    This function writes a NumPy-compatible array to a GeoTIFF file using
    `meta["crs"]` and `meta["transform"]` as the spatial reference.

    Accepted input shapes are:
    - `(H, W)` for a single-band image,
    - `(C, H, W)` for a multiband image,
    - `(H, W, 3)` or `(H, W, 4)` for RGB or RGBA data, automatically converted
    to channel-first `(C, H, W)`.

    The output directory is created automatically if it does not already exist.
    If `dtype` is provided, the data is cast before writing. If `nodata` is
    provided for floating-point output, NaN values are replaced with that nodata
    value before export.

    Args:
        out_path: Destination path of the GeoTIFF file to create.
        arr: Array to export. Any array-like input is converted to `np.ndarray`.
        meta: Metadata dictionary containing at least `"crs"` and `"transform"`.
        dtype: Optional output dtype. If `None`, the input array dtype is kept.
        nodata: Optional nodata value to store in the GeoTIFF profile. For
            floating-point outputs, NaN values are replaced by this value.
        compress: Compression method passed to the GeoTIFF profile.
        bigtiff: BigTIFF policy passed to the GeoTIFF profile.

    Returns:
        The output path of the written GeoTIFF file.

    Raises:
        ValueError: If `arr` is not 2D or 3D.
        KeyError: If `meta["crs"]` or `meta["transform"]` is missing.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if arr.ndim == 2:
        data = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[-1] in (3, 4) and arr.shape[0] not in (3, 4):
            data = np.moveaxis(arr, -1, 0)
        else:
            data = arr
    else:
        raise ValueError(f"arr must be 2D or 3D, got shape={arr.shape}")

    C, H, W = map(int, data.shape)

    crs = meta.get("crs")
    transform = meta.get("transform")
    if crs is None:
        raise KeyError("meta['crs'] is missing.")
    if transform is None:
        raise KeyError("meta['transform'] is missing.")

    out_dtype = np.dtype(dtype).name if dtype is not None else data.dtype.name
    out_data = data.astype(out_dtype, copy=False) if out_dtype != data.dtype.name else data

    if nodata is not None and np.issubdtype(out_data.dtype, np.floating):
        out_data = out_data.copy()
        out_data[np.isnan(out_data)] = nodata

    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": C,
        "dtype": out_dtype,
        "crs": crs,
        "transform": transform,
        "compress": compress,
        "bigtiff": bigtiff,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_data)

    return out_path