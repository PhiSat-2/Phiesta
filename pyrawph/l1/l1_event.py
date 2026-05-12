from __future__ import annotations

import os
import re
from datetime import datetime
from sys import path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except Exception:  
    torch = None 

from pathlib import Path


try:
    from termcolor import colored
except Exception:  
    def colored(x, *_args, **_kwargs): 
        return x

from ..utils.l1_utils import read_L1_event_from_folder_phisat2
from ..utils.processing_utils import make_rgb, normalized_difference
from ..utils.export_utils import export_to_tif as _export_to_tif

from rasterio.windows import Window, bounds as window_bounds, transform as window_transform

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..remote.insula_client import InsulaClient

from ..remote.constants import DEFAULT_L1_DOWNLOAD_DIR, VM_L1_ROOT
from ..remote.local_resolver import resolve_existing_product
from ..remote.catalog_geometry import (
    enrich_meta_with_insula_feature,
    get_catalog_corners as _get_catalog_corners,
    get_catalog_center as _get_catalog_center,
    get_catalog_polygon as _get_catalog_polygon,
    format_catalog_geo as _format_catalog_geo,
    print_catalog_geo as _print_catalog_geo,
)

from ..georef.catalog_overlay import (
    ensure_nearest_valid_cdse_mosaic_for_catalog as _ensure_nearest_valid_cdse_mosaic_for_catalog,
    show_catalog_geo_in_sentinel as _show_catalog_geo_in_sentinel,
    show_coordinates_in_sentinel as _show_coordinates_in_sentinel,
    compare_catalog_rectified as _compare_catalog_rectified,
)

from ..utils.display import (
    prepare_event_display_image as _prepare_event_display_image,
    show_prepared_display as _show_prepared_display,
    format_event_display_title as _format_event_display_title,
)
from ..utils.l0_l1_registration import (
    register_event_bands_to_master as _register_event_bands_to_master,
)

from .normalize import normalize_l1_product_layout

from ..utils.stats import (
    compute_band_stats as _compute_band_stats,
    plot_value_distribution as _plot_value_distribution,
)

from ..triplets.builder import build_sentinel_triplet as _build_sentinel_triplet


BandSpec = Union[int, str, float]





def _try_parse_product_times(product_folder: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to parse two timestamps from the product folder name:
      ..._<YYYYMMDDhhmmss>_<YYYYMMDDhhmmss>_...
    Returns ISO-like strings (or None).
    """
    base = os.path.basename(product_folder.rstrip("\\/"))
    m = re.search(r"_(\d{14})_(\d{14})_", base)
    if not m:
        return None, None

    t0, t1 = m.group(1), m.group(2)

    def _fmt(s: str) -> str:
        dt = datetime.strptime(s, "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    try:
        return _fmt(t0), _fmt(t1)
    except Exception:
        return None, None
    



class L1_event:
    """
    Represent one local ΦSat-2 L1 scene as a multispectral cube with shape
    (C, H, W) and associated metadata.

    This class provides the core high-level interface for:
    - loading a local L1 product,
    - accessing bands by index, wavelength, or alias,
    - building RGB composites,
    - computing simple normalized-difference indices,
    - cropping in pixel coordinates,
    - exporting the result to GeoTIFF,
    - inspecting basic event metadata.

    The class intentionally stays lightweight and does not manage tiles or higher-
    level pairing logic. Cross-space registration between L0 and L1 is handled by
    separate utilities.
    """

    # alias wavelengths (nm) -> resolve by closest wavelength
    _ALIAS_NM: Dict[str, int] = {
        "BLUE": 490, "B": 490,
        "GREEN": 560, "G": 560,
        "RED": 665, "R": 665,
        "REDEDGE1": 705, "RE1": 705,
        "REDEDGE2": 740, "RE2": 740,
        "REDEDGE3": 783, "RE3": 783,
        "NIR": 842,
    }

    @property
    def meta(self) -> Dict[str, Any]:
        return self._meta

    @property
    def product_folder(self) -> str:
        return self._product_folder

    @property
    def scene_id(self) -> int:
        return self._scene_id

    @property
    def device(self) -> str:
        return self._device
    
    @property
    def product_kind(self) -> str:
        return self._product_kind
    
    def build_sentinel_triplet(self, **kwargs):
        """
        Build a Sentinel-2 / simulated PhiSat-2 / real PhiSat-2 triplet.

        This is the high-level user-facing entry point. The current implementation
        initializes the triplet workspace and QC metadata; Sentinel sourcing,
        simulation, and alignment are progressively added in the triplet pipeline.
        """
        return _build_sentinel_triplet(self, **kwargs)
    
    def get_catalog_corners(self, order: str = "latlon"):
        return _get_catalog_corners(self, order=order)

    def get_catalog_center(self, order: str = "latlon"):
        return _get_catalog_center(self, order=order)

    def get_catalog_polygon(self, order: str = "latlon", closed: bool = True):
        return _get_catalog_polygon(self, order=order, closed=closed)

    def format_catalog_geo(self, order: str = "latlon", decimals: int = 6) -> str:
        return _format_catalog_geo(self, order=order, decimals=decimals)

    def print_catalog_geo(self, order: str = "latlon", decimals: int = 6) -> None:
        _print_catalog_geo(self, order=order, decimals=decimals)

    def show_catalog_geo_in_sentinel(self, cache_dir: str | Path = "georef_cache", **kwargs):
        return _show_catalog_geo_in_sentinel(self, cache_dir=cache_dir, **kwargs)

    def show_coordinates_in_sentinel(
        self,
        cache_dir: str | Path = "georef_cache",
        **kwargs,
    ):
        """
        Show the Insula catalog footprint on a Sentinel-2 mosaic.

        If no mosaic_path is provided, PyRawPh automatically builds or reuses one.
        """
        if "mosaic_path" not in kwargs:
            cached = self._meta.get("sentinel_mosaic_path")
            if cached is not None:
                kwargs["mosaic_path"] = cached

        result = _show_coordinates_in_sentinel(
            self,
            cache_dir=cache_dir,
            **kwargs,
        )

        mosaic = result.get("mosaic")
        if mosaic is not None and getattr(mosaic, "source_path", None) is not None:
            self._meta["sentinel_mosaic_path"] = str(mosaic.source_path)

        return result
    
    def compare_catalog_to_phisat(self, cache_dir: str | Path = "georef_cache", **kwargs):
        """
        Legacy compatibility alias.

        Prefer compare_catalog_rectified(...) for new code.
        """
        return _compare_catalog_rectified(self, cache_dir=cache_dir, **kwargs)
    
    def compare_catalog_rectified(
        self,
        cache_dir: str | Path = "georef_cache",
        **kwargs,
    ):
        """
        Compare PhiSat-2 with a rectified Sentinel-2 crop from the catalog footprint.

        If no mosaic_path is provided, PyRawPh automatically builds or reuses one.
        """
        if "mosaic_path" not in kwargs:
            cached = self._meta.get("sentinel_mosaic_path")
            if cached is not None:
                kwargs["mosaic_path"] = cached

        result = _compare_catalog_rectified(
            self,
            cache_dir=cache_dir,
            **kwargs,
        )

        mosaic_path = result.get("mosaic_path")
        if mosaic_path is not None:
            self._meta["sentinel_mosaic_path"] = str(mosaic_path)

        return result
    

    def band_stats(
        self,
        bands="all",
        percentiles=(0, 1, 2, 5, 50, 95, 98, 99, 100),
        sample=500_000,
        random_state=0,
    ):
        """
        Compute raw-value statistics for selected bands.
        """
        return _compute_band_stats(
            self,
            bands=bands,
            percentiles=percentiles,
            sample=sample,
            random_state=random_state,
        )


    def plot_distribution(
        self,
        bands="all",
        bins=256,
        sample=500_000,
        log_y=True,
        percentiles=(1, 2, 50, 98, 99),
        hist_range_percentiles=(0.1, 99.9),
        random_state=0,
        figsize=(12, 7),
        title=None,
        out_png=None,
    ):
        """
        Plot raw-value distributions for selected bands.
        """
        return _plot_value_distribution(
            self,
            bands=bands,
            bins=bins,
            sample=sample,
            log_y=log_y,
            percentiles=percentiles,
            hist_range_percentiles=hist_range_percentiles,
            random_state=random_state,
            figsize=figsize,
            title=title,
            out_png=out_png,
        )

    def _resolve_band_index_for_registration(self, band):
        """
        Resolve a band selector to an integer index for registration purposes.
        """
        return self._resolve_band(band)


    def _registered_for_display(
        self,
        master_band="RED",
        max_shifts=(80, 80),
        force: bool = False,
    ):
        """
        Return a band-registered copy of this event for display.
        """
        if not hasattr(self, "_display_registered_cache"):
            self._display_registered_cache = {}

        cache_key = f"master={master_band}|max_shifts={tuple(max_shifts)}"

        if not force and cache_key in self._display_registered_cache:
            return self._display_registered_cache[cache_key]

        master_idx = self._resolve_band_index_for_registration(master_band)

        registered = _register_event_bands_to_master(
            self,
            master_band=master_idx,
            max_shifts=max_shifts,
        )

        self._display_registered_cache[cache_key] = registered
        return registered
    
    def with_array(
        self,
        arr: np.ndarray,
        meta_updates: dict | None = None,
    ) -> "L1_event":
        """
        Return a new L1_event with the same metadata but a different array.

        Useful for derived products such as display-registered cubes.
        The original event is not modified.
        """
        arr = np.asarray(arr)

        meta = dict(self._meta)
        if meta_updates:
            meta.update(meta_updates)

        if arr.ndim == 3:
            meta["count"] = int(arr.shape[0])
            meta["height"] = int(arr.shape[1])
            meta["width"] = int(arr.shape[2])
        elif arr.ndim == 2:
            meta["count"] = 1
            meta["height"] = int(arr.shape[0])
            meta["width"] = int(arr.shape[1])
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

        meta["dtype"] = str(arr.dtype)

        product_folder = getattr(self, "product_folder", meta.get("path", None))
        scene_id = getattr(self, "scene_id", meta.get("scene_id", 0))
        product_kind = getattr(
            self,
            "product_kind",
            meta.get("product_kind", meta.get("kind", "BC")),
        )

        return L1_event(
            arr=arr,
            meta=meta,
            product_folder=str(product_folder) if product_folder is not None else "",
            scene_id=scene_id,
            product_kind=product_kind,
        )


    def show(
        self,
        bands=("RED", "GREEN", "BLUE"),
        normalize: bool = True,
        normalization: str = "percentile",
        percentiles: tuple[float, float] = (2, 98),
        per_band: bool = True,
        registered: bool = False,
        registration_master="RED",
        max_shifts=(80, 80),
        force_registration: bool = False,
        interpolation: str = "nearest",
        figsize=(8, 8),
        title: str | None = None,
        out_png: str | None = None,
    ):
        """
        Display a PhiSat-2 event.

        Args:
            bands:
                - one selector, e.g. "NIR" or 7 -> grayscale
                - three selectors, e.g. ("RED", "GREEN", "BLUE") -> RGB
                - "all" -> all bands separately
            normalize: Whether to normalize values for display.
            normalization: "percentile", "minmax", "zscore", or "none".
            percentiles: Percentiles used for percentile normalization.
            per_band: If True, normalize each displayed band/channel independently.
            registered: If True, register all bands to `registration_master` before display.
            registration_master: Master band used for display registration.
            max_shifts: Maximum allowed band-to-band registration shift.
            force_registration: Recompute registration even if cached.
            interpolation: Matplotlib display interpolation.
            figsize: Figure size.
            title: Optional title.
            out_png: Optional output PNG path.
        """
        ev = self
        if registered:
            ev = self._registered_for_display(
                master_band=registration_master,
                max_shifts=max_shifts,
                force=force_registration,
            )

        prepared = _prepare_event_display_image(
            ev,
            bands=bands,
            normalize=normalize,
            normalization=normalization,
            percentiles=percentiles,
            per_band=per_band,
        )

        if title is None:
            title = _format_event_display_title(
                level="L1",
                prepared=prepared,
                registered=registered,
                normalize=normalize,
                normalization=normalization,
                percentiles=percentiles,
                per_band=per_band,
                registration_master=registration_master,
            )

        return _show_prepared_display(
            prepared,
            figsize=figsize,
            title=title,
            interpolation=interpolation,
            out_png=out_png,
        )


    def show_rgb(self, bands=("RED", "GREEN", "BLUE"), **kwargs):
        """
        Display an RGB composite.
        """
        return self.show(bands=bands, **kwargs)


    def show_band(self, band, **kwargs):
        """
        Display one band in grayscale.
        """
        return self.show(bands=band, **kwargs)


    def show_all_bands(self, **kwargs):
        """
        Display all bands separately.
        """
        return self.show(bands="all", **kwargs)
    
    def ensure_sentinel_mosaic(
        self,
        cache_dir: str | Path = "georef_cache",
        preset: str = "balanced",
        **kwargs,
    ):
        """
        Build or reuse a valid Sentinel-2 mosaic around the catalog footprint.
        """
        path = _ensure_nearest_valid_cdse_mosaic_for_catalog(
            self,
            cache_dir=cache_dir,
            preset=preset,
            **kwargs,
        )
        self._meta["sentinel_mosaic_path"] = str(path)
        return path

    def __init__(
        self,
        arr: np.ndarray,
        meta: Dict[str, Any],
        product_folder: str,
        scene_id: int,
        product_kind: str,
        device: str = "cpu",
    ):
        """
        Initialize an :class:`L1_event` from an array and metadata.

        Args:
            arr: Multispectral array with shape `(C, H, W)`.
            meta: Metadata dictionary associated with the scene.
            product_folder: Path to the source product folder.
            scene_id: Scene identifier inside the product.
            product_kind: Product variant, for example `"BC"`.
            device: Target device used when converting the event to a torch tensor.

        Raises:
            ValueError: If the provided array is not three-dimensional.
        """
        self._arr = arr
        self._meta = meta
        self._product_folder = product_folder
        self._scene_id = int(scene_id)
        self._product_kind = product_kind.upper()
        self._device = device

        # ensure times exist 
        st, ct = _try_parse_product_times(product_folder)
        if "sensing_time" not in self._meta:
            self._meta["sensing_time"] = st
        if "creation_time" not in self._meta:
            self._meta["creation_time"] = ct
        
        if self._arr.ndim != 3:
            raise ValueError(f"L1_event expects (C,H,W), got {self._arr.shape}.")

        

    # constructors
    @classmethod
    def from_path(
        cls,
        product_folder: str,
        scene_id: int = 0,
        product_kind: str = "BC",
        multiband: bool = True,
        bands: Optional[List[int]] = None,
        as_float32: bool = True,
        verbose: bool = True,
        device: str = "cpu",
        normalize_layout: bool = True,
    ) -> "L1_event":
        if normalize_layout:
            normalize_l1_product_layout(
                product_folder=product_folder,
                scene_id=scene_id,
                product_kind=product_kind,
                overwrite=False,
            )

        if verbose:
            print("[PyRawPh] Loading ΦSat-2 L1 from:", product_folder)

        arr, meta = read_L1_event_from_folder_phisat2(
            product_folder=product_folder,
            scene_id=scene_id,
            product_kind=product_kind,
            multiband=multiband,
            bands=bands,
            as_float32=as_float32,
        )
        return cls(
            arr=arr,
            meta=meta,
            product_folder=product_folder,
            scene_id=scene_id,
            product_kind=product_kind,
            device=device,
        )
    
    @classmethod
    def from_insula_search(
        cls,
        client: "InsulaClient",
        ref_data_collection: str,
        page: int = 0,
        results_per_page: int = 20,
        feature_index: int = 0,
        keep_zip: bool = False,
        skip_existing: bool = True,
        force_redownload: bool = False,
        scene_id: int = 0,
        product_kind: str = "BC",
        multiband: bool = True,
        bands: Optional[List[int]] = None,
        as_float32: bool = True,
        verbose: bool = True,
        device: str = "cpu",
        dest_dir: str | Path | None = None,
        **search_filters,
    ) -> "L1_event":
        """
        Search remote PHISAT-2 L1 products on Insula, download one result, and load it.

        This is a low-level constructor. For date-based workflows, prefer:
        1) `client.search_l1(date=...)`
        2) `client.load_l1(identifier=...)`

        Args:
            client: Connected Insula client.
            ref_data_collection: Insula REF_DATA collection id.
            page: Search page index.
            results_per_page: Number of results requested from Insula.
            feature_index: Index inside the returned page.
            keep_zip: If True, keep the downloaded zip archive on disk.
            skip_existing: If True, reuse an already extracted local copy when possible.
            force_redownload: If True, ignore any local copy and download again.
            scene_id: Scene id to load from the downloaded product.
            product_kind: Product variant, typically `"BC"`.
            multiband: If True, read the canonical multiband TIFF.
            bands: Optional subset of L1 band indices to load.
            as_float32: If True, cast the array to float32.
            verbose: If True, print a loading message.
            device: Device used by `as_tensor()`.
            dest_dir: Optional destination directory for downloads.
            **search_filters: Additional Insula search parameters.

        Returns:
            A loaded `L1_event`.

        Raises:
            ValueError: If no result is found.
            IndexError: If `feature_index` is out of range.
        """
        data = client.search_ref_data(
            ref_data_collection=ref_data_collection,
            page=page,
            results_per_page=results_per_page,
            **search_filters,
        )

        features = data.get("features", [])
        if not features:
            raise ValueError("No feature found for this Insula search.")

        if not (0 <= feature_index < len(features)):
            raise IndexError(
                f"feature_index={feature_index} out of range for {len(features)} result(s)."
            )

        feature = features[feature_index]

        product_folder = client.download_feature(
            feature,
            dest_dir=dest_dir or DEFAULT_L1_DOWNLOAD_DIR,
            extract=True,
            keep_zip=keep_zip,
            skip_existing=skip_existing,
            force_redownload=force_redownload,
        )
        event = cls.from_path(
            product_folder=str(product_folder),
            scene_id=scene_id,
            product_kind=product_kind,
            multiband=multiband,
            bands=bands,
            as_float32=as_float32,
            verbose=verbose,
            device=device,
        )

        event._meta["source"] = "insula"
        event._meta["resolved_product_folder"] = str(product_folder)

        enrich_meta_with_insula_feature(
            meta=event._meta,
            feature=feature,
            ref_data_collection=ref_data_collection,
        )

        return event

    @classmethod
    def from_insula_identifier(
        cls,
        client: InsulaClient,
        ref_data_collection: str,
        identifier: str,
        keep_zip: bool = False,
        skip_existing: bool = True,
        force_redownload: bool = False,
        scene_id: int = 0,
        product_kind: str = "BC",
        multiband: bool = True,
        bands: Optional[List[int]] = None,
        as_float32: bool = True,
        verbose: bool = True,
        device: str = "cpu",
        dest_dir: str | Path | None = None,
        local_fallback: bool = True,
        vm_fallback: bool = False,
        local_roots: Optional[List[str | Path]] = None,
        attach_catalog_geo: bool = True,
    ) -> "L1_event":
        """
        Resolution order:
        1) local/project data dir
        2) VM shared dir (if vm_fallback=True)
        3) Insula download
        """
        search_roots = []

        if local_fallback:
            search_roots.append(dest_dir or DEFAULT_L1_DOWNLOAD_DIR)

        if vm_fallback:
            search_roots.append(VM_L1_ROOT)

        if local_roots:
            search_roots.extend(local_roots)

        existing = resolve_existing_product(
            identifier=identifier,
            roots=search_roots,
        )

        feature = None

        if existing is not None:
            product_folder = existing
            feature_props = None

            if attach_catalog_geo:
                feature = client.get_feature_by_identifier(
                    ref_data_collection=ref_data_collection,
                    identifier=identifier,
                )
                feature_props = feature["properties"]
        else:
            feature = client.get_feature_by_identifier(
                ref_data_collection=ref_data_collection,
                identifier=identifier,
            )

            product_folder = client.download_feature(
                feature,
                dest_dir=dest_dir or DEFAULT_L1_DOWNLOAD_DIR,
                extract=True,
                keep_zip=keep_zip,
                skip_existing=skip_existing,
                force_redownload=force_redownload,
            )
            feature_props = feature["properties"]

        event = cls.from_path(
            product_folder=str(product_folder),
            scene_id=scene_id,
            product_kind=product_kind,
            multiband=multiband,
            bands=bands,
            as_float32=as_float32,
            verbose=verbose,
            device=device,
        )

        event._meta["source"] = "local_or_vm" if feature_props is None else "insula"
        event._meta["resolved_product_folder"] = str(product_folder)

        if feature_props is not None:
            enrich_meta_with_insula_feature(
                meta=event._meta,
                feature=feature,
                ref_data_collection=ref_data_collection,
            )

        return event

    # basic getters
    def as_numpy(self) -> np.ndarray:
        """
        Return the event data as a NumPy array.

        The returned array is the internal scene array stored by the event and is
        expected to have shape `(C, H, W)`.

        Returns:
            The event data as a NumPy array of shape `(C, H, W)`.
        """
        return self._arr

    def as_tensor(self, as_float32: bool = True):
        """
        Return the event data as a PyTorch tensor on the configured device.

        Args:
            as_float32: If `True`, cast the tensor to `torch.float32` before moving
                it to the target device.

        Returns:
            A PyTorch tensor containing the event data, typically with shape
            `(C, H, W)`.

        Raises:
            ImportError: If PyTorch is not available in the current environment.
        """

        if torch is None:
            raise ImportError("torch is not available")
        t = torch.from_numpy(self._arr)
        if as_float32 and t.dtype != torch.float32:
            t = t.float()
        return t.to(self._device)

    def get_meta(self) -> Dict[str, Any]:
        """
        Return the metadata dictionary associated with the event.

        The metadata may contain fields such as CRS, affine transform, bounds,
        wavelengths, product paths, sensing time, and creation time.

        Returns:
            The event metadata dictionary.
        """
        return self._meta

    def get_wavelengths(self) -> List[Optional[int]]:
        """
        Return the list of band center wavelengths in nanometers.
        """
        for key in ("band_wavelength_nm", "wavelengths_nm", "band_wavelengths_nm"):
            w = self._meta.get(key, None)
            if isinstance(w, (list, tuple)):
                return list(w)
        return []

    def _resolve_band(self, band: BandSpec) -> int:
        """
        Resolve a band selector to a zero-based local channel index in the currently
        loaded array.

        Supported selectors include:
        - integer local channel indices,
        - float wavelengths in nanometers,
        - strings such as `"842nm"`, `"3"`, `"B3"`, or `"BAND_3"`,
        - aliases such as `"BLUE"`, `"GREEN"`, `"RED"`, `"RE1"`, `"RE2"`, `"RE3"`,
        and `"NIR"`.

        If the metadata contains `picked_bands`, selectors such as `"B3"` are first
        interpreted as original physical band identifiers and then mapped to the local
        channel index. Otherwise, they fall back to local indexing.

        Args:
            band: Band selector to resolve.

        Returns:
            The resolved local channel index.

        Raises:
            ValueError: If the selector cannot be resolved.
        """
        C = int(self._arr.shape[0])
        picked = list(self._meta.get("picked_bands", list(range(C))))
        wls = self.get_wavelengths()

        def _closest_wavelength_index(target_nm: int) -> int:
            if not wls:
                raise ValueError("No wavelengths in metadata; cannot resolve by wavelength.")
            valid = [(i, v) for i, v in enumerate(wls) if v is not None]
            if not valid:
                raise ValueError("No valid wavelengths in metadata; cannot resolve by wavelength.")
            return int(min(valid, key=lambda iv: abs(int(iv[1]) - target_nm))[0])

        if isinstance(band, int):
            if not (0 <= band < C):
                raise ValueError(f"Band index out of range: {band} (C={C})")
            return band

        if isinstance(band, float):
            return _closest_wavelength_index(int(round(band)))

        s = str(band).strip().upper()
        s_clean = s.replace(" ", "").replace("_", "")

        if s_clean.endswith("NM") and s_clean[:-2].isdigit():
            target_nm = int(s_clean[:-2])
            return _closest_wavelength_index(target_nm)

        if s.isdigit():
            raw_idx = int(s)
            if raw_idx in picked:
                return picked.index(raw_idx)
            if 0 <= raw_idx < C:
                return raw_idx
            raise ValueError(f"Band {raw_idx} not available in picked_bands={picked}")

        for prefix in ("BAND_", "BAND", "B"):
            if s.startswith(prefix) and s[len(prefix):].isdigit():
                raw_idx = int(s[len(prefix):])
                if raw_idx in picked:
                    return picked.index(raw_idx)
                if 0 <= raw_idx < C:
                    return raw_idx
                raise ValueError(f"Band {raw_idx} not available in picked_bands={picked}")

        if s in self._ALIAS_NM:
            return _closest_wavelength_index(int(self._ALIAS_NM[s]))

        raise ValueError(f"Cannot resolve band spec: {band!r}")

    def get_band(self, band: BandSpec) -> np.ndarray:
        """
        Return one band from the event as a 2D array.

        The band can be selected by:
        - integer band index,
        - float wavelength in nanometers,
        - string specification such as `"NIR"`, `"RED"`, `"B3"`, `"BAND_7"`,
            or `"842nm"`.

        String aliases are resolved through the event wavelength metadata using the
        closest matching wavelength when needed.

        Args:
            band: Band selector.

        Returns:
            A 2D NumPy array of shape `(H, W)` corresponding to the selected band.

        Raises:
            ValueError: If the band specification cannot be resolved.
        """
        i = self._resolve_band(band)
        return self._arr[i]

    # processing
    def rgb(
        self,
        bands=("RED", "GREEN", "BLUE"),
        stretch=(2, 98),
        arr: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build an RGB composite from three selected bands.

        By default, the composite uses the `"RED"`, `"GREEN"`, and `"BLUE"` aliases.
        An optional source array can be provided instead of the event internal array,
        but it must follow the same band ordering and use shape `(C, H, W)`.

        Args:
            bands: A length-3 sequence describing the red, green, and blue channels.
                Each entry can use any valid band selector supported by
                :meth:`get_band`.
            stretch: Percentile stretch applied before composing the RGB image.
            arr: Optional source array with shape `(C, H, W)`. If `None`, the event
                internal array is used.

        Returns:
            An RGB image as a NumPy array, typically with shape `(H, W, 3)`.
        """
        src = self._arr if arr is None else arr

        r = src[self._resolve_band(bands[0])].astype(np.float32)
        g = src[self._resolve_band(bands[1])].astype(np.float32)
        b = src[self._resolve_band(bands[2])].astype(np.float32)

        return make_rgb(r, g, b, stretch=stretch)

    def index(self, name: str, **kwargs) -> np.ndarray:
        """
        Compute a built-in normalized spectral index.

        Currently supported indices are:
        - `"NDVI"`: `(NIR - RED) / (NIR + RED)`
        - `"NDWI"`: `(GREEN - NIR) / (GREEN + NIR)`

        Band selectors can be overridden through keyword arguments. For example,
        `nir`, `red`, and `green` may each be given as an integer band index, a
        wavelength in nanometers, or a string alias such as `"NIR"` or `"B3"`.

        Args:
            name: Name of the spectral index to compute.
            **kwargs: Optional band selector overrides used by the selected index.

        Returns:
            A 2D NumPy array containing the computed index.

        Raises:
            ValueError: If the requested index name is not supported.
        """
        n = name.strip().upper()

        if n == "NDVI":
            nir = kwargs.get("nir", "NIR")
            red = kwargs.get("red", "RED")
            return normalized_difference(self.get_band(nir), self.get_band(red))

        if n == "NDWI":
            green = kwargs.get("green", "GREEN")
            nir = kwargs.get("nir", "NIR")
            return normalized_difference(self.get_band(green), self.get_band(nir))

        raise ValueError(f"Unknown index: {name!r}")

    def crop_px(self, y0: int, y1: int, x0: int, x1: int) -> "L1_event":
        """
        Crop the event in pixel coordinates and return a new :class:`L1_event`.

        The crop is defined using half-open intervals `[y0:y1, x0:x1]`. Bounds are
        clamped to the valid image extent. When affine metadata is available, the
        transform and bounds are updated to match the cropped window.

        Args:
            y0: Start row index.
            y1: End row index (exclusive).
            x0: Start column index.
            x1: End column index (exclusive).

        Returns:
            A new :class:`L1_event` containing the cropped array and updated metadata.

        Raises:
            ValueError: If the resulting crop is empty or invalid.
        """
        H, W = int(self._arr.shape[1]), int(self._arr.shape[2])

        y0c = max(0, min(H, int(y0)))
        y1c = max(0, min(H, int(y1)))
        x0c = max(0, min(W, int(x0)))
        x1c = max(0, min(W, int(x1)))

        if y0c >= y1c or x0c >= x1c:
            raise ValueError(f"Invalid crop after clamp: y[{y0c},{y1c}) x[{x0c},{x1c}) for H={H}, W={W}")

        arr_c = self._arr[:, y0c:y1c, x0c:x1c].copy()

        meta_c = dict(self._meta)
        meta_c["height"] = int(y1c - y0c)
        meta_c["width"] = int(x1c - x0c)
        meta_c["crop_box"] = [y0c, y1c, x0c, x1c]

        t0 = self._meta.get("transform", None)
        if t0 is not None:
            win = Window(col_off=x0c, row_off=y0c, width=(x1c - x0c), height=(y1c - y0c))
            meta_c["transform"] = window_transform(win, t0)
            try:
                left, bottom, right, top = window_bounds(win, t0)
                meta_c["bounds"] = (left, bottom, right, top)
            except Exception:
                pass

        return L1_event(
            arr=arr_c,
            meta=meta_c,
            product_folder=self._product_folder,
            scene_id=self._scene_id,
            product_kind=self._product_kind,
            device=self._device,
        )

    
    

    def show_event_info(self) -> None:
        """
        Print a concise summary of the event.

        The summary includes the scene identifier, product kind, folder and source path,
        array shape and dtype, CRS, geographic bounds, band wavelengths, and the main
        sidecar files if available.

        Returns:
            None.
        """
        print(colored("Event:", "blue"), f"scene_id={self._scene_id} kind={self._product_kind}")
        print("  folder:", colored(self._product_folder, "red"))
        print("  path:", colored(str(self._meta.get("path", None)), "red"))
        print("  shape:", colored(str(tuple(self._arr.shape)), "red"), "  dtype:", colored(str(self._arr.dtype), "red"))
        print("  crs:", colored(str(self._meta.get("crs", None)), "red"))
        print("  bounds:", colored(str(self._meta.get("bounds", None)), "red"))
        print("  wavelengths_nm:", colored(str(self.get_wavelengths()), "red"))
        print("  gl_path:", colored(str(self._meta.get("gl_path", None)), "red"))
        print("  processing_config:", colored(str(self._meta.get("processing_config_path", None)), "red"))
    
    
    

    # export
    def export_to_tif(
        self,
        out_path: str,
        arr: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Export the event data, or a provided array, to a GeoTIFF file.

        If `arr` is not provided, the event internal array is exported. If `meta` is
        not provided, the event metadata is used. Additional keyword arguments are
        forwarded to the low-level GeoTIFF export utility.

        Args:
            out_path: Output path of the GeoTIFF file to create.
            arr: Optional array to export. If `None`, the event internal array is
                used.
            meta: Optional metadata dictionary to use for export. If `None`, the
                event metadata is used.
            **kwargs: Additional keyword arguments forwarded to the underlying export
                utility.

        Returns:
            The output path of the written GeoTIFF file.

        Raises:
            ValueError: If the provided metadata is incomplete or incompatible with
                GeoTIFF export.
            OSError: If the file cannot be written.
        """
        if arr is None:
            arr = self._arr
        if meta is None:
            meta = self._meta
        return _export_to_tif(out_path=out_path, arr=arr, meta=meta, **kwargs)