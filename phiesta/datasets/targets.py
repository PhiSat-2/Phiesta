from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import json

import numpy as np
import pandas as pd


@dataclass
class TargetContext:
    """Context passed to a target provider for one dataset row."""

    dataset: Any
    name: str
    level: str
    output_dir: Path
    row_index: int
    item_id: str


@dataclass
class TargetResult:
    """Explicit result object for target providers."""

    value: Any = None
    array: np.ndarray | None = None
    path: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _safe_name(value: str) -> str:
    chars = []
    for ch in str(value):
        chars.append(ch if (ch.isalnum() or ch in "-_.") else "_")
    text = "".join(chars).strip("._")
    return text or "target"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _normalize_result(result: Any) -> TargetResult:
    if isinstance(result, TargetResult):
        return result
    if isinstance(result, np.ndarray):
        return TargetResult(array=result)
    if isinstance(result, Path):
        return TargetResult(path=result)
    if isinstance(result, dict):
        special = {"value", "array", "path", "metadata"}
        if special.intersection(result):
            return TargetResult(
                value=result.get("value"),
                array=result.get("array"),
                path=result.get("path"),
                metadata=dict(result.get("metadata") or {}),
            )
        return TargetResult(metadata=dict(result))
    if result is None or isinstance(
        result,
        (str, bytes, bool, int, float, np.integer, np.floating, np.bool_),
    ):
        return TargetResult(value=result)
    raise TypeError(
        "Target provider must return TargetResult, numpy.ndarray, scalar, "
        "Path, or dict."
    )


def column_target(column: str) -> Callable:
    """Create a provider that copies one manifest column into a target."""

    column = str(column)

    def provider(row, *, context):
        if column not in row:
            raise KeyError(f"Target source column {column!r} is missing.")
        return TargetResult(
            value=row[column],
            metadata={"source_column": column},
        )

    provider.__name__ = f"column_target_{_safe_name(column)}"
    return provider


class RasterTarget:
    """
    Align a local georeferenced label raster to each image/patch grid.

    ``source`` may be a fixed path or ``source(row) -> path``.
    """

    def __init__(
        self,
        source,
        *,
        band: int = 1,
        resampling: str = "nearest",
        dtype: Any | None = None,
        src_nodata: Any | None = None,
        dst_nodata: Any = 0,
    ):
        self.source = source
        self.band = int(band)
        self.resampling = str(resampling)
        self.dtype = dtype
        self.src_nodata = src_nodata
        self.dst_nodata = dst_nodata

    def _source_path(self, row) -> Path:
        value = self.source(row) if callable(self.source) else self.source
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def __call__(self, row, *, context):
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window
        from rasterio.warp import reproject

        image_path = row.get("raster_path")
        if not image_path:
            raise ValueError("Raster targets require a raster_path in the manifest.")
        image_path = Path(str(image_path))
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        source_path = self._source_path(row)

        try:
            resampling = getattr(Resampling, self.resampling)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown raster resampling mode {self.resampling!r}."
            ) from exc

        with rasterio.open(image_path) as image:
            if image.crs is None:
                raise ValueError(
                    "Dataset image is not georeferenced. Build the L1 dataset "
                    "with georeference=True before adding raster targets."
                )

            if context.level == "patches":
                needed = ("x_min", "y_min", "x_max", "y_max")
                missing = [key for key in needed if key not in row]
                if missing:
                    raise ValueError(
                        f"Patch target is missing window columns: {missing}"
                    )
                x0 = int(row["x_min"])
                y0 = int(row["y_min"])
                x1 = int(row["x_max"])
                y1 = int(row["y_max"])
                width = x1 - x0
                height = y1 - y0
                if width <= 0 or height <= 0:
                    raise ValueError("Invalid patch window.")
                window = Window(x0, y0, width, height)
                dst_transform = image.window_transform(window)
            else:
                width = int(image.width)
                height = int(image.height)
                dst_transform = image.transform

            dst_crs = image.crs

        with rasterio.open(source_path) as src:
            out_dtype = np.dtype(self.dtype or src.dtypes[self.band - 1])
            destination = np.full(
                (height, width),
                self.dst_nodata,
                dtype=out_dtype,
            )
            reproject(
                source=rasterio.band(src, self.band),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=(
                    self.src_nodata
                    if self.src_nodata is not None
                    else src.nodata
                ),
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=self.dst_nodata,
                resampling=resampling,
            )

        return TargetResult(
            array=destination,
            metadata={
                "source_raster": str(source_path),
                "band": self.band,
                "resampling": self.resampling,
                "dst_nodata": _jsonable(self.dst_nodata),
            },
        )


def raster_target(source, **kwargs) -> RasterTarget:
    """Create a local georeferenced-raster target provider."""
    return RasterTarget(source, **kwargs)


def worldcover_target(source, **kwargs) -> RasterTarget:
    """
    Create a categorical ESA WorldCover target provider from a local raster.

    Nearest-neighbour resampling preserves integer class codes.
    """
    kwargs.setdefault("band", 1)
    kwargs.setdefault("resampling", "nearest")
    kwargs.setdefault("dtype", np.uint8)
    kwargs.setdefault("dst_nodata", 0)
    return RasterTarget(source, **kwargs)


def _select_table(dataset, level: str):
    level = str(level).lower()
    if level == "auto":
        level = "patches" if not dataset.patches.empty else "acquisitions"
    if level == "patches":
        if dataset.patches.empty:
            raise ValueError(
                "Dataset has no patches. Use level='acquisitions' or build "
                "with patch_size=..."
            )
        return level, dataset.patches.copy()
    if level == "acquisitions":
        if dataset.acquisitions.empty:
            raise ValueError("Dataset has no acquisitions.")
        return level, dataset.acquisitions.copy()
    raise ValueError("level must be 'auto', 'patches', or 'acquisitions'.")


def list_targets(dataset) -> pd.DataFrame:
    """Return target definitions stored in dataset metadata."""
    rows = []
    for name, info in dataset.metadata.get("targets", {}).items():
        row = {"name": name}
        if isinstance(info, dict):
            row.update(info)
        rows.append(row)
    return pd.DataFrame(rows)


def add_target(
    dataset,
    name: str,
    provider,
    *,
    level: str = "auto",
    overwrite: bool = False,
    continue_on_error: bool = True,
    provider_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Add one target to every acquisition or patch.

    Provider signature::

        provider(row_dict, context=context)

    Accepted outputs are scalar values, NumPy arrays, Paths, dictionaries, or
    TargetResult. Array outputs are saved as ``.npy`` under
    ``targets/<name>/``. Successful rows are skipped on subsequent calls unless
    ``overwrite=True``.
    """
    if not callable(provider):
        raise TypeError("provider must be callable.")

    safe = _safe_name(name)
    provider_kwargs = dict(provider_kwargs or {})
    level, table = _select_table(dataset, level)

    prefix = f"target_{safe}"
    status_col = f"{prefix}_status"
    value_col = f"{prefix}_value"
    path_col = f"{prefix}_path"
    error_col = f"{prefix}_error"

    output_dir = dataset.root / "targets" / safe
    output_dir.mkdir(parents=True, exist_ok=True)

    if level == "patches":
        if "dataset_patch_id" in table.columns:
            ids = table["dataset_patch_id"].astype(str).tolist()
        elif "patch_id" in table.columns:
            ids = table["patch_id"].astype(str).tolist()
        else:
            ids = [f"patch_{i:06d}" for i in range(len(table))]
    else:
        if "product_id" in table.columns:
            ids = table["product_id"].astype(str).tolist()
        else:
            ids = [f"acquisition_{i:06d}" for i in range(len(table))]

    from .builder import _write_state

    def checkpoint():
        if level == "patches":
            dataset.patches = table.copy()
        else:
            dataset.acquisitions = table.copy()
        _write_state(
            dataset.root,
            dataset.acquisitions,
            dataset.patches,
            dataset.metadata,
        )

    success = 0
    failed = 0

    for pos, (index, row) in enumerate(table.iterrows()):
        if (
            not overwrite
            and status_col in table.columns
            and str(table.at[index, status_col]) == "SUCCESS"
        ):
            continue

        item_id = _safe_name(ids[pos])
        context = TargetContext(
            dataset=dataset,
            name=safe,
            level=level,
            output_dir=output_dir,
            row_index=pos,
            item_id=item_id,
        )

        try:
            raw = provider(
                row.to_dict(),
                context=context,
                **provider_kwargs,
            )
            result = _normalize_result(raw)

            target_path = None
            if result.array is not None:
                target_path = output_dir / f"{item_id}.npy"
                if overwrite or not target_path.exists():
                    np.save(target_path, np.asarray(result.array))
            elif result.path is not None:
                target_path = Path(result.path)

            table.at[index, status_col] = "SUCCESS"
            table.at[index, error_col] = ""

            if result.value is not None:
                table.at[index, value_col] = _jsonable(result.value)

            if target_path is not None:
                table.at[index, path_col] = str(target_path)

            for key, value in result.metadata.items():
                table.at[index, f"{prefix}_{_safe_name(key)}"] = _jsonable(value)

            if result.array is not None:
                arr = np.asarray(result.array)
                table.at[index, f"{prefix}_shape"] = str(tuple(arr.shape))
                table.at[index, f"{prefix}_dtype"] = str(arr.dtype)

            success += 1
            if verbose:
                print(f"[Phiesta target] SUCCESS {safe} {item_id}")

        except Exception as exc:
            table.at[index, status_col] = "FAILED"
            table.at[index, error_col] = f"{type(exc).__name__}: {exc}"
            failed += 1
            checkpoint()
            if verbose:
                print(
                    f"[Phiesta target] FAILED {safe} {item_id}: "
                    f"{table.at[index, error_col]}"
                )
            if not continue_on_error:
                raise
            continue

        checkpoint()

    dataset.metadata.setdefault("targets", {})
    dataset.metadata["targets"][safe] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "provider": getattr(provider, "__name__", provider.__class__.__name__),
        "status_column": status_col,
        "value_column": value_col,
        "path_column": path_col,
        "success_this_run": int(success),
        "failed_this_run": int(failed),
    }
    checkpoint()

    if verbose:
        print(
            f"[Phiesta target] done name={safe}, level={level}, "
            f"success={success}, failed={failed}"
        )

    return table
