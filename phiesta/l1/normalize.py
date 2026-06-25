from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import rasterio


def _list_tifs(bands_dir: Path) -> List[Path]:
    return sorted(
        [p for p in bands_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    )


def _is_rgb_preview(path: Path) -> bool:
    return "rgb" in path.stem.lower()


def _canonical_band_path(bands_dir: Path, scene_id: int, product_kind: str, band_idx: int) -> Path:
    return bands_dir / f"scene_{scene_id}_{product_kind}_band_{band_idx}.tiff"


def _canonical_multiband_path(bands_dir: Path, scene_id: int, product_kind: str) -> Path:
    return bands_dir / f"scene_{scene_id}_{product_kind}_multiband.tiff"


def _extract_band_index_from_name(path: Path) -> Optional[int]:
    stem = path.stem.lower()

    m = re.search(r"_band_(\d+)$", stem)
    if m:
        idx = int(m.group(1))
        return idx if 0 <= idx <= 7 else None

    nums = re.findall(r"(\d+)", stem)
    if not nums:
        return None

    idx = int(nums[-1])
    return idx if 0 <= idx <= 7 else None


def _find_source_multiband(tif_paths: List[Path]) -> Optional[Path]:
    candidates = []

    for p in tif_paths:
        if _is_rgb_preview(p):
            continue
        try:
            with rasterio.open(p) as ds:
                if ds.count >= 8:
                    score = ds.count
                    stem = p.stem.lower()
                    if "multiband" in stem:
                        score += 100
                    if "session" in stem:
                        score += 50
                    if "scene_" in stem:
                        score += 25
                    candidates.append((score, p))
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _find_source_single_bands(tif_paths: List[Path]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}

    for p in tif_paths:
        if _is_rgb_preview(p):
            continue
        try:
            with rasterio.open(p) as ds:
                if ds.count != 1:
                    continue
        except Exception:
            continue

        idx = _extract_band_index_from_name(p)
        if idx is not None and idx not in out:
            out[idx] = p

    return out


def _write_single_band_copy(src_path: Path, dst_path: Path, overwrite: bool = False) -> None:
    if dst_path.exists() and not overwrite:
        return

    with rasterio.open(src_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        profile.update(count=1)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data, 1)


def _write_single_band_from_multiband(
    src_multiband_path: Path,
    band_idx: int,
    dst_path: Path,
    overwrite: bool = False,
) -> None:
    if dst_path.exists() and not overwrite:
        return

    with rasterio.open(src_multiband_path) as src:
        if src.count < 8:
            raise ValueError(
                f"Multiband source has only {src.count} band(s), expected at least 8: {src_multiband_path}"
            )
        data = src.read(band_idx + 1)
        profile = src.profile.copy()
        profile.update(count=1)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data, 1)


def _write_canonical_multiband_from_source(
    src_multiband_path: Path,
    dst_path: Path,
    overwrite: bool = False,
) -> None:
    if dst_path.exists() and not overwrite:
        return

    with rasterio.open(src_multiband_path) as src:
        if src.count < 8:
            raise ValueError(
                f"Multiband source has only {src.count} band(s), expected at least 8: {src_multiband_path}"
            )

        # canonical normalized multiband = first 8 useful bands only
        data = src.read(list(range(1, 9)))
        profile = src.profile.copy()
        profile.update(count=8)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)


def _write_canonical_multiband_from_band_files(
    src_band_paths: List[Path],
    dst_path: Path,
    overwrite: bool = False,
) -> None:
    if dst_path.exists() and not overwrite:
        return

    if len(src_band_paths) != 8:
        raise ValueError(f"Expected 8 band files, got {len(src_band_paths)}")

    data_stack = []
    profile = None

    for p in src_band_paths:
        with rasterio.open(p) as src:
            data_stack.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()

    assert profile is not None
    profile.update(count=8)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        for i, band in enumerate(data_stack, start=1):
            dst.write(band, i)


def normalize_l1_product_layout(
    product_folder: str | Path,
    scene_id: int = 0,
    product_kind: str = "BC",
    overwrite: bool = False,
) -> Path:
    """
    Normalize a legacy or non-canonical PHISAT-2 L1 product layout into the
    canonical PyRawPh layout.

    The canonical layout is:

        bands/scene_<scene_id>_<product_kind>_band_0.tiff
        ...
        bands/scene_<scene_id>_<product_kind>_band_7.tiff
        bands/scene_<scene_id>_<product_kind>_multiband.tiff

    This function is designed to make old or differently named products readable
    by the standard L1 reader. In particular, it:

    - ignores RGB preview files,
    - creates canonical single-band TIFF aliases when needed,
    - creates a canonical multiband TIFF when needed,
    - truncates legacy 9-band multiband products to the first 8 physical bands.

    Args:
        product_folder: Path to the local L1 product folder.
        scene_id: Scene identifier used to build canonical file names.
        product_kind: Product variant, typically `"BC"`.
        overwrite: If True, overwrite existing canonical outputs.

    Returns:
        The input product folder path as a `Path`, after normalization.
    """
    product_kind = product_kind.upper()
    product_folder = Path(product_folder)
    bands_dir = product_folder / "bands"

    if not bands_dir.is_dir():
        raise FileNotFoundError(f"Missing bands directory: {bands_dir}")

    tif_paths = _list_tifs(bands_dir)
    if not tif_paths:
        raise FileNotFoundError(f"No TIFF files found in {bands_dir}")

    canonical_multiband = _canonical_multiband_path(bands_dir, scene_id, product_kind)
    canonical_band_paths = [
        _canonical_band_path(bands_dir, scene_id, product_kind, i) for i in range(8)
    ]

    source_multiband = _find_source_multiband(tif_paths)
    source_single_bands = _find_source_single_bands(tif_paths)

    # 1) create canonical single-band files
    for i, dst_path in enumerate(canonical_band_paths):
        if dst_path.exists() and not overwrite:
            continue

        if i in source_single_bands:
            _write_single_band_copy(source_single_bands[i], dst_path, overwrite=True)
        elif source_multiband is not None:
            _write_single_band_from_multiband(source_multiband, i, dst_path, overwrite=True)
        else:
            raise FileNotFoundError(
                f"Could not create canonical band {i} in {bands_dir}: "
                f"no usable source single-band file and no usable multiband file found."
            )

    # 2) create canonical multiband
    if not canonical_multiband.exists() or overwrite:
        if source_multiband is not None and source_multiband.resolve() != canonical_multiband.resolve():
            _write_canonical_multiband_from_source(
                source_multiband,
                canonical_multiband,
                overwrite=True,
            )
        else:
            _write_canonical_multiband_from_band_files(
                canonical_band_paths,
                canonical_multiband,
                overwrite=True,
            )

    return product_folder