from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass
class PhiestaDataset:
    root: Path
    acquisitions: pd.DataFrame
    patches: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def acquisition_manifest_path(self) -> Path:
        return self.root / "acquisitions.csv"

    @property
    def patch_manifest_path(self) -> Path:
        return self.root / "patches.csv"

    @property
    def selection_path(self) -> Path:
        return self.root / "selection.csv"

    def __len__(self) -> int:
        return len(self.patches) if not self.patches.empty else len(self.acquisitions)

    def __repr__(self) -> str:
        ok = failed = 0
        if "build_status" in self.acquisitions.columns:
            ok = int((self.acquisitions["build_status"] == "SUCCESS").sum())
            failed = int((self.acquisitions["build_status"] == "FAILED").sum())
        return (
            f"PhiestaDataset(root={str(self.root)!r}, acquisitions={len(self.acquisitions)}, "
            f"success={ok}, failed={failed}, patches={len(self.patches)})"
        )

    @property
    def split_manifest_path(self) -> Path:
        return self.root / "splits.csv"

    def make_splits(self, **kwargs):
        from .splits import make_splits
        return make_splits(self, **kwargs)

    def split_summary(self, **kwargs):
        from .splits import split_summary
        return split_summary(self, **kwargs)

    def get_split(self, name, **kwargs):
        from .splits import get_split
        return get_split(self, name, **kwargs)


def _normalize_product_id(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if not s:
        return None
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    from phiesta.remote.catalog_geometry import extract_phisat_acquisition_id
    return extract_phisat_acquisition_id(s) or s


def normalize_dataset_selection(selection: Any, *, product_id_col: str = "product_id") -> pd.DataFrame:
    """Normalize DataFrame/CSV/search-result/id-list selections into one table."""
    from phiesta.remote.search_table import search_result_to_dataframe

    if isinstance(selection, pd.DataFrame):
        table = selection.copy()
    elif isinstance(selection, pd.Series):
        table = selection.to_frame() if selection.name == product_id_col else pd.DataFrame({product_id_col: selection.tolist()})
    elif isinstance(selection, (str, Path)):
        path = Path(selection)
        if path.exists():
            if path.suffix.lower() != ".csv":
                raise ValueError("Dataset selection files currently support CSV only.")
            table = pd.read_csv(path)
        else:
            table = pd.DataFrame({product_id_col: [str(selection)]})
    elif isinstance(selection, dict):
        table = search_result_to_dataframe(selection) if ("features" in selection or "content" in selection) else pd.DataFrame([selection])
    elif isinstance(selection, Iterable):
        values = list(selection)
        if not values:
            table = pd.DataFrame(columns=[product_id_col])
        elif isinstance(values[0], dict):
            first = values[0]
            table = search_result_to_dataframe(values) if ("geometry" in first and "properties" in first) else pd.DataFrame(values)
        else:
            table = pd.DataFrame({product_id_col: values})
    else:
        raise TypeError("Unsupported dataset selection type.")

    if product_id_col not in table.columns:
        raise ValueError(f"Dataset selection must contain column {product_id_col!r}.")

    table = table.copy()
    table[product_id_col] = table[product_id_col].map(_normalize_product_id)
    table = table[table[product_id_col].notna()].copy()
    table = table.drop_duplicates(subset=[product_id_col], keep="first")
    return table.reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _upsert(table: pd.DataFrame, record: dict[str, Any], product_id_col: str) -> pd.DataFrame:
    pid = str(record[product_id_col])
    if not table.empty and product_id_col in table.columns:
        table = table[table[product_id_col].astype(str) != pid].copy()
    return pd.concat([table, pd.DataFrame([record])], ignore_index=True)


def _write_state(root: Path, acquisitions: pd.DataFrame, patches: pd.DataFrame, metadata: dict[str, Any]):
    acquisitions.to_csv(root / "acquisitions.csv", index=False)
    patches.to_csv(root / "patches.csv", index=False)
    safe = {}
    for k, v in metadata.items():
        try:
            json.dumps(v)
            safe[k] = v
        except TypeError:
            safe[k] = str(v)
    (root / "dataset.json").write_text(json.dumps(safe, indent=2, ensure_ascii=False), encoding="utf-8")


def open_dataset(root: str | Path) -> PhiestaDataset:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    acquisitions = _read_csv(root / "acquisitions.csv")
    patches = _read_csv(root / "patches.csv")
    meta_path = root / "dataset.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return PhiestaDataset(root, acquisitions, patches, metadata)


def build_dataset(
    client,
    selection: Any,
    *,
    out_dir: str | Path,
    level: str = "L1",
    product_id_col: str = "product_id",
    limit: int | None = None,
    georeference: bool = False,
    georeference_kwargs: dict[str, Any] | None = None,
    patch_size: int | tuple[int, int] | None = None,
    stride: int | tuple[int, int] | None = None,
    patch_limit: int | None = None,
    bands: Any = "all",
    normalize: bool = False,
    normalize_kwargs: dict[str, Any] | None = None,
    dtype: Any | None = None,
    include_partial: bool = False,
    patch_kwargs: dict[str, Any] | None = None,
    load_kwargs: dict[str, Any] | None = None,
    continue_on_error: bool = True,
    resume: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> PhiestaDataset:
    """
    Build a dataset from any acquisition selection.

    Selection is intentionally independent of construction: WorldCover, bbox/date
    search, custom pandas filtering, CSVs and explicit product-id lists all work.
    All selection columns are propagated to acquisition/patch manifests.

    patch_size=None builds only the acquisition manifest. Set patch_size to export
    ML-ready .npy patches. L1 datasets may set georeference=True before patching.
    Progress is checkpointed after every acquisition and resume=True skips prior
    SUCCESS rows.
    """
    level = str(level).upper()
    if level not in {"L1", "L0"}:
        raise ValueError("level must be 'L1' or 'L0'.")
    if georeference and level != "L1":
        raise ValueError("georeference=True is currently supported for L1 only.")

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    selection_table = normalize_dataset_selection(selection, product_id_col=product_id_col)
    if limit is not None:
        selection_table = selection_table.head(int(limit)).copy()
    selection_table.to_csv(root / "selection.csv", index=False)

    acq_path, patch_path = root / "acquisitions.csv", root / "patches.csv"
    if resume:
        acquisitions, patches = _read_csv(acq_path), _read_csv(patch_path)
    else:
        if not overwrite and (acq_path.exists() or patch_path.exists()):
            raise FileExistsError(f"Dataset manifests already exist in {root}.")
        acquisitions, patches = pd.DataFrame(), pd.DataFrame()

    load_kwargs = dict(load_kwargs or {})
    georeference_kwargs = dict(georeference_kwargs or {})
    normalize_kwargs = dict(normalize_kwargs or {})
    patch_kwargs = dict(patch_kwargs or {})

    completed = set()
    if not acquisitions.empty and product_id_col in acquisitions.columns and "build_status" in acquisitions.columns:
        ok = acquisitions["build_status"].astype(str) == "SUCCESS"
        completed = set(acquisitions.loc[ok, product_id_col].dropna().astype(str))

    metadata = {
        "format": "phiesta-dataset-v1",
        "created_or_updated_utc": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "selection_count": int(len(selection_table)),
        "georeference": bool(georeference),
        "patch_size": patch_size,
        "stride": stride,
        "patch_limit": patch_limit,
        "bands": str(bands),
        "normalize": bool(normalize),
        "dtype": None if dtype is None else str(dtype),
        "include_partial": bool(include_partial),
    }
    _write_state(root, acquisitions, patches, metadata)

    def checkpoint():
        metadata["created_or_updated_utc"] = datetime.now(timezone.utc).isoformat()
        _write_state(root, acquisitions, patches, metadata)

    try:
        for _, row in selection_table.iterrows():
            source = row.to_dict()
            pid = _normalize_product_id(source.get(product_id_col))
            if pid is None:
                continue
            if resume and pid in completed:
                if verbose:
                    print(f"[Phiesta dataset] skip {pid}: already SUCCESS")
                continue

            record = dict(source)
            record[product_id_col] = pid
            record.update({
                "dataset_level": level,
                "build_status": "PROCESSING",
                "build_error": "",
                "source_product_folder": "",
                "raster_path": "",
                "georeferenced": False,
                "patch_count": 0,
            })

            try:
                if verbose:
                    print(f"[Phiesta dataset] {level} {pid}")
                event = client.load_l1(pid, **load_kwargs) if level == "L1" else client.load_l0(pid, **load_kwargs)
                record["source_product_folder"] = str(getattr(event, "product_folder", "") or "")
                meta = getattr(event, "meta", {}) or {}

                # Keep catalog center metadata even when the selection was only
                # a list of ids, so spatial splitting remains available.
                catalog_geo = meta.get("catalog_geo")
                if isinstance(catalog_geo, dict):
                    center = catalog_geo.get("center_lonlat")
                    if center is not None and len(center) >= 2:
                        if record.get("center_lon") in (None, ""):
                            record["center_lon"] = float(center[0])
                        if record.get("center_lat") in (None, ""):
                            record["center_lat"] = float(center[1])
                    if record.get("start_datetime") in (None, ""):
                        value = catalog_geo.get("start_datetime")
                        if value is not None:
                            record["start_datetime"] = value

                raster_path = meta.get("raster_path") or meta.get("path") or ""
                work_event = event

                if georeference:
                    georef_dir = root / "georeferenced"
                    georef_dir.mkdir(parents=True, exist_ok=True)
                    kwargs = dict(georeference_kwargs)
                    kwargs.setdefault("output_path", str(georef_dir / f"phisat2_{pid}_georeferenced.tif"))
                    work_event = event.georeference(**kwargs)
                    wmeta = getattr(work_event, "meta", {}) or {}
                    raster_path = wmeta.get("raster_path") or wmeta.get("path") or kwargs["output_path"]
                    record["georeferenced"] = True

                record["raster_path"] = str(raster_path or "")

                if patch_size is not None:
                    product_patch_dir = root / "patches" / pid
                    args = dict(
                        out_dir=product_patch_dir,
                        patch_size=patch_size,
                        stride=stride,
                        bands=bands,
                        normalize=normalize,
                        normalize_kwargs=normalize_kwargs,
                        dtype=dtype,
                        prefix=pid,
                        overwrite=overwrite,
                        limit=patch_limit,
                        save_index=False,
                        include_partial=include_partial,
                    )
                    args.update(patch_kwargs)
                    pt = work_event.export_patches(**args).copy()
                    if "status" in pt.columns:
                        pt = pt.rename(columns={"status": "patch_status"})
                    pt["product_id"] = pid
                    pt["dataset_level"] = level
                    pt["source_product_folder"] = record["source_product_folder"]
                    pt["raster_path"] = record["raster_path"]
                    pt["georeferenced"] = bool(record["georeferenced"])
                    if "patch_id" in pt.columns:
                        pt["dataset_patch_id"] = pid + "_" + pt["patch_id"].astype(str)
                    for key, value in source.items():
                        if key not in pt.columns:
                            pt[key] = value
                    if not patches.empty and "product_id" in patches.columns:
                        patches = patches[patches["product_id"].astype(str) != pid].copy()
                    patches = pd.concat([patches, pt], ignore_index=True)
                    record["patch_count"] = int(len(pt))

                record["build_status"] = "SUCCESS"
                acquisitions = _upsert(acquisitions, record, product_id_col)
                checkpoint()
                if verbose:
                    suffix = f" patches={record['patch_count']}" if patch_size is not None else ""
                    print(f"[Phiesta dataset] SUCCESS {pid}{suffix}")

            except Exception as exc:
                record["build_status"] = "FAILED"
                record["build_error"] = f"{type(exc).__name__}: {exc}"
                acquisitions = _upsert(acquisitions, record, product_id_col)
                checkpoint()
                if verbose:
                    print(f"[Phiesta dataset] FAILED {pid}: {record['build_error']}")
                if not continue_on_error:
                    raise

    except KeyboardInterrupt:
        metadata["interrupted"] = True
        checkpoint()
        raise

    metadata["interrupted"] = False
    metadata["acquisition_rows"] = int(len(acquisitions))
    metadata["successful_acquisitions"] = int((acquisitions.get("build_status", pd.Series(dtype=str)) == "SUCCESS").sum())
    metadata["failed_acquisitions"] = int((acquisitions.get("build_status", pd.Series(dtype=str)) == "FAILED").sum())
    metadata["patch_rows"] = int(len(patches))
    checkpoint()

    result = PhiestaDataset(root, acquisitions.reset_index(drop=True), patches.reset_index(drop=True), metadata)
    if verbose:
        print(result)
    return result
