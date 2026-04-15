from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None


RAW_BAND_RE = re.compile(r"Band(\d+)", re.IGNORECASE)


def _read_json(path: Path) -> Dict[str, Any]:
    """
    Read a JSON file and return its contents as a dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_first_existing(product_folder: Path, candidates: Sequence[str]) -> Optional[Path]:
    """
    Return the first existing path among several candidate file names.

    Args:
        product_folder: Base folder in which candidates are searched.
        candidates: Candidate file names, in priority order.

    Returns:
        The first existing path, or `None` if none exists.
    """
    for name in candidates:
        p = product_folder / name
        if p.exists():
            return p
    return None


def _read_tiff(path: Path) -> np.ndarray:
    """
    Read one 2D TIFF image as a NumPy array.

    The function prefers `tifffile` when available and falls back to `rasterio`
    otherwise.

    Args:
        path: Path to the TIFF file.

    Returns:
        A 2D NumPy array.

    Raises:
        ImportError: If neither `tifffile` nor `rasterio` is available.
        ValueError: If the TIFF does not contain a 2D image.
    """
    if tifffile is not None:
        arr = tifffile.imread(path)
    elif rasterio is not None:
        with rasterio.open(path) as src:
            arr = src.read(1)
    else:  # pragma: no cover
        raise ImportError("Need tifffile or rasterio to read raw TIFF bands.")

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D TIFF for {path.name}, got shape {arr.shape}.")
    return arr


def discover_raw_band_paths(product_folder: str | Path, scene_id: int = 0) -> List[Path]:
    """
    Find raw TIFF files for bands 0..7 inside product_folder/raw.
    Keeps only the full-resolution TIFFs, not thumbnails.
    """
    product_folder = Path(product_folder)
    raw_dir = product_folder / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")

    tif_paths = sorted(
        [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    )

    band_to_path: Dict[int, Path] = {}
    for p in tif_paths:
        name_lower = p.name.lower()

        # Exclude thumbnails / TN products
        if "_tn_" in name_lower or "thumbnail" in name_lower:
            continue

        m = RAW_BAND_RE.search(p.name)
        if m is None:
            continue

        band_idx = int(m.group(1))
        if 0 <= band_idx <= 7:
            band_to_path[band_idx] = p

    missing = [b for b in range(8) if b not in band_to_path]
    if missing:
        raise FileNotFoundError(
            f"Could not find full-res TIFFs for bands {missing} in {raw_dir}."
        )

    return [band_to_path[b] for b in range(8)]


def load_l0_stack(
    product_folder: str | Path,
    scene_id: int = 0,
    bands: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Read the 8 full-resolution raw TIFFs and build an array of shape (C, H, W).
    """
    product_folder = Path(product_folder)

    metadata_path = _find_first_existing(product_folder, ["metadata", "metadata.json"])
    ancillary_path = _find_first_existing(product_folder, ["ancillary", "ancillary.json"])
    aocs_path = _find_first_existing(product_folder, ["aocs", "aocs.json"])
    raw_bin_path = _find_first_existing(product_folder, ["raw.bin"])

    if metadata_path is None:
        raise FileNotFoundError(
            f"Could not find metadata file in {product_folder} "
            f"(expected metadata or metadata.json)."
        )

    metadata_json = _read_json(metadata_path)
    ancillary_json = _read_json(ancillary_path) if ancillary_path else {}
    aocs_json = _read_json(aocs_path) if aocs_path else {}

    session_ids = list(metadata_json.get("Sessions", {}).keys())
    if not session_ids:
        raise ValueError("metadata JSON does not contain any session.")

    session_id = session_ids[0]
    session = metadata_json["Sessions"][session_id]

    scene = session["Scenes"][str(scene_id)]
    imager_cfg = session["ImagerConfiguration"]

    band_paths = discover_raw_band_paths(product_folder, scene_id=scene_id)
    full_stack = np.stack([_read_tiff(p) for p in band_paths], axis=0)  # (C, H, W)

    if bands is not None:
        bands = list(bands)
        arr = full_stack[bands]
        selected_paths = [band_paths[b] for b in bands]
        selected_wavelengths = [imager_cfg["BandCWL"][b] for b in bands]
        selected_start_rows = [imager_cfg["BandStartRow"][b] for b in bands]
    else:
        bands = list(range(full_stack.shape[0]))
        arr = full_stack
        selected_paths = band_paths
        selected_wavelengths = list(imager_cfg["BandCWL"])
        selected_start_rows = list(imager_cfg["BandStartRow"])

    meta: Dict[str, Any] = {
        "path": str(product_folder),
        "product_folder": str(product_folder),
        "scene_id": scene_id,
        "session_id": session_id,
        "count": int(arr.shape[0]),
        "height": int(arr.shape[1]),
        "width": int(arr.shape[2]),
        "dtype": str(arr.dtype),
        "band_indices": list(bands),
        "band_paths": [str(p) for p in selected_paths],
        "band_wavelength_nm": selected_wavelengths,
        "band_start_row": selected_start_rows,
        "line_period": imager_cfg.get("LinePeriod"),
        "band_setup": imager_cfg.get("BandSetup"),
        "thumbnail_factor": imager_cfg.get("ThumbnailFactor"),
        "scan_direction": imager_cfg.get("ScanDirection"),
        "spectral_bands_total": imager_cfg.get("SpectralBands"),
        "scene_type": scene.get("Type"),
        "scene_width": scene.get("Width"),
        "scene_height": scene.get("Height"),
        "metadata_path": str(metadata_path),
        "ancillary_path": str(ancillary_path) if ancillary_path else None,
        "aocs_path": str(aocs_path) if aocs_path else None,
        "raw_bin_path": str(raw_bin_path) if raw_bin_path else None,
        "metadata_json": metadata_json,
        "ancillary_json": ancillary_json,
        "aocs_json": aocs_json,
        "native_space": "L0_native",
    }

    return arr, meta