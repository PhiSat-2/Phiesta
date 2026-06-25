from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


def _as_pair(value, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)

    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])

    raise ValueError(f"{name} must be an int or a length-2 tuple/list, got {value!r}")


def _event_height_width(event) -> tuple[int, int]:
    meta = getattr(event, "meta", None) or getattr(event, "_meta", {}) or {}

    height = meta.get("height")
    width = meta.get("width")

    if height is not None and width is not None:
        return int(height), int(width)

    band0 = np.asarray(event.get_band(0))
    return int(band0.shape[0]), int(band0.shape[1])


def _starts(start: int, stop: int, size: int, stride: int, include_partial: bool) -> list[int]:
    start = int(start)
    stop = int(stop)
    size = int(size)
    stride = int(stride)

    if stop <= start:
        return []

    if stop - start < size:
        return [start] if include_partial else []

    values = list(range(start, stop - size + 1, stride))

    if include_partial:
        last = stop - size
        if not values or values[-1] != last:
            values.append(last)

    return sorted(set(values))


def build_patch_index(
    event,
    patch_size: int | tuple[int, int] = 512,
    stride: int | tuple[int, int] | None = None,
    x_min: int = 0,
    y_min: int = 0,
    x_max: int | None = None,
    y_max: int | None = None,
    include_partial: bool = False,
) -> pd.DataFrame:
    """
    Build a pixel-window index for regular patches over an event.

    Args:
        event: L0/L1 event-like object.
        patch_size: int or (height, width).
        stride: int or (y_stride, x_stride). Defaults to patch_size.
        x_min, y_min: top-left area origin.
        x_max, y_max: exclusive area end. Defaults to image bounds.
        include_partial: if True, include edge patches clipped to the image.

    Returns:
        DataFrame with patch_id, row, col, x/y windows, width, height.
    """
    image_height, image_width = _event_height_width(event)

    patch_h, patch_w = _as_pair(patch_size, "patch_size")

    if stride is None:
        stride_h, stride_w = patch_h, patch_w
    else:
        stride_h, stride_w = _as_pair(stride, "stride")

    x_max = image_width if x_max is None else int(x_max)
    y_max = image_height if y_max is None else int(y_max)

    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(image_width, x_max)
    y_max = min(image_height, y_max)

    xs = _starts(x_min, x_max, patch_w, stride_w, include_partial)
    ys = _starts(y_min, y_max, patch_h, stride_h, include_partial)

    rows = []

    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            x1 = min(x0 + patch_w, x_max)
            y1 = min(y0 + patch_h, y_max)

            width = x1 - x0
            height = y1 - y0

            if width <= 0 or height <= 0:
                continue

            is_partial = width != patch_w or height != patch_h

            rows.append({
                "patch_id": f"r{r:04d}_c{c:04d}",
                "row": r,
                "col": c,
                "x_min": x0,
                "y_min": y0,
                "x_max": x1,
                "y_max": y1,
                "width": width,
                "height": height,
                "is_partial": bool(is_partial),
            })

    return pd.DataFrame(rows)


def iter_patches(
    event,
    index: pd.DataFrame | None = None,
    patch_size: int | tuple[int, int] = 512,
    stride: int | tuple[int, int] | None = None,
    bands: Any = "all",
    band_axis: int = 0,
    squeeze: bool = True,
    normalize: bool = False,
    normalize_kwargs: dict | None = None,
    include_metadata: bool = True,
    limit: int | None = None,
    **index_kwargs,
) -> Iterator[dict | np.ndarray]:
    """
    Iterate over patches.

    If include_metadata=True, yields dictionaries:
        {"patch_id", "patch", "window", "row"}

    Otherwise yields patch arrays directly.
    """
    if index is None:
        index = build_patch_index(
            event,
            patch_size=patch_size,
            stride=stride,
            **index_kwargs,
        )

    if limit is not None:
        index = index.head(int(limit))

    normalize_kwargs = normalize_kwargs or {}

    for _, row in index.iterrows():
        patch = event.get_patch(
            x_min=int(row["x_min"]),
            y_min=int(row["y_min"]),
            x_max=int(row["x_max"]),
            y_max=int(row["y_max"]),
            bands=bands,
            band_axis=band_axis,
            squeeze=squeeze,
            clip=True,
        )

        if normalize:
            norm_kwargs = dict(normalize_kwargs)
            norm_kwargs.setdefault("band_axis", band_axis)
            patch = event.normalize(
                patch,
                **norm_kwargs,
            )

        if not include_metadata:
            yield patch
            continue

        yield {
            "patch_id": row["patch_id"],
            "patch": patch,
            "window": {
                "x_min": int(row["x_min"]),
                "y_min": int(row["y_min"]),
                "x_max": int(row["x_max"]),
                "y_max": int(row["y_max"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            },
            "row": row.to_dict(),
        }


def export_patches(
    event,
    out_dir: str | Path,
    index: pd.DataFrame | None = None,
    patch_size: int | tuple[int, int] = 512,
    stride: int | tuple[int, int] | None = None,
    bands: Any = "all",
    band_axis: int = 0,
    normalize: bool = False,
    normalize_kwargs: dict | None = None,
    dtype: Any | None = None,
    prefix: str = "patch",
    overwrite: bool = False,
    limit: int | None = None,
    save_index: bool = True,
    **index_kwargs,
) -> pd.DataFrame:
    """
    Export patches as .npy files and return an index DataFrame.

    This is a simple ML/research export format. It intentionally does not claim
    georeferenced GeoTIFF export.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if index is None:
        index = build_patch_index(
            event,
            patch_size=patch_size,
            stride=stride,
            **index_kwargs,
        )

    if limit is not None:
        index = index.head(int(limit)).copy()
    else:
        index = index.copy()

    records = []
    normalize_kwargs = normalize_kwargs or {}

    for item in iter_patches(
        event,
        index=index,
        bands=bands,
        band_axis=band_axis,
        squeeze=False,
        normalize=normalize,
        normalize_kwargs=normalize_kwargs,
        include_metadata=True,
    ):
        patch_id = item["patch_id"]
        patch = item["patch"]

        if dtype is not None:
            patch = patch.astype(dtype, copy=False)

        path = out_dir / f"{prefix}_{patch_id}.npy"

        status = "written"
        if path.exists() and not overwrite:
            status = "exists"
        else:
            np.save(path, patch)

        record = dict(item["row"])
        record["patch_path"] = str(path)
        record["status"] = status
        record["bands"] = str(bands)
        record["band_axis"] = band_axis
        record["normalized"] = bool(normalize)
        records.append(record)

    out_index = pd.DataFrame(records)

    if save_index:
        out_index.to_csv(out_dir / f"{prefix}_index.csv", index=False)

    return out_index
