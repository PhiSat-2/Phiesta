from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Union, List

import numpy as np

from ..utils.export_utils import export_to_tif as _export_to_tif, export_plain_tif


from .reader import load_l0_stack

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..remote.insula_client import InsulaClient
    
from pathlib import Path
from .rawbin_converter import convert_l0_rawbin_inplace

from ..remote.constants import DEFAULT_L0_DOWNLOAD_DIR, VM_L0_ROOT
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
    show_catalog_geo_in_sentinel as _show_catalog_geo_in_sentinel,
    show_coordinates_in_sentinel as _show_coordinates_in_sentinel,
)

from ..utils.display import (
    prepare_event_display_image as _prepare_event_display_image,
    show_prepared_display as _show_prepared_display,
    format_event_display_title as _format_event_display_title,
)

from ..utils.l0_l1_registration import (
    register_event_bands_to_master as _register_event_bands_to_master,
)

from ..utils.stats import (
    compute_band_stats as _compute_band_stats,
    plot_value_distribution as _plot_value_distribution,
)

BandSelector = Union[int, float, str]


_ALIAS_TO_WAVELENGTH = {
    "BLUE": 490.0,
    "GREEN": 560.0,
    "RED": 665.0,
    "RE1": 705.0,
    "RE2": 740.0,
    "RE3": 783.0,
    "NIR": 842.0,
}



def _looks_prepared_for_l0_event(product_folder: str | Path) -> bool:
    """
    A product is considered ready for L0_event.from_path(...) if:
    - it has metadata(.json)
    - and a raw/ directory with TIFFs
    """
    p = Path(product_folder)

    has_metadata = (p / "metadata").exists() or (p / "metadata.json").exists()
    raw_dir = p / "raw"
    has_raw_tiffs = raw_dir.exists() and any(
        x.is_file() and x.suffix.lower() in {".tif", ".tiff"} for x in raw_dir.iterdir()
    )

    return has_metadata and has_raw_tiffs

@dataclass
class L0_event:
    """
    Represent one local ΦSat-2 L0 acquisition as a multispectral cube with shape
    (C, H, W) and associated metadata.

    This class is the main entry point for working with raw ΦSat-2 L0 data inside
    PyRawPh. It provides:
    - loading from a local product folder,
    - access to spectral bands by index, wavelength, or alias,
    - conversion to NumPy or PyTorch,
    - simple display helpers for debugging,
    - pixel-space cropping,
    - replacement of the underlying array while preserving metadata structure,
    - optional registration hooks,
    - export to TIFF, either plain or georeferenced depending on metadata.

    The event stores the raw data in its native pixel space unless explicitly
    replaced by a registered array.
    """
    arr: np.ndarray
    meta: Dict[str, Any]
    product_folder: str
    scene_id: int = 0
    device: str = "cpu"

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
            cached = self.meta.get("sentinel_mosaic_path")
            if cached is not None:
                kwargs["mosaic_path"] = cached

        result = _show_coordinates_in_sentinel(
            self,
            cache_dir=cache_dir,
            **kwargs,
        )

        mosaic = result.get("mosaic")
        if mosaic is not None and getattr(mosaic, "source_path", None) is not None:
            self.meta["sentinel_mosaic_path"] = str(mosaic.source_path)

        return result

    def __post_init__(self) -> None:
        """
        Validate and normalize the internal array after dataclass initialization.

        The event data is converted to a NumPy array and must have shape `(C, H, W)`.
        This check ensures that all downstream methods can assume a channel-first
        multispectral layout.

        Raises:
            ValueError: If the provided array is not three-dimensional.
        """
        self.arr = np.asarray(self.arr)
        if self.arr.ndim != 3:
            raise ValueError(f"L0_event expects (C,H,W), got {self.arr.shape}.")
    

    

    def _resolve_band_index_for_registration(self, band):
        """
        Resolve a band selector to an integer index for registration purposes.
        """
        return self._resolve_band_index(band)


    def _registered_for_display(
        self,
        master_band="RED",
        max_shifts=(80, 80),
        force=False,
    ):
        """
        Return a band-registered copy of this L0 event for display.
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


    def show(
        self,
        bands=("RED", "GREEN", "BLUE"),
        normalize=True,
        normalization="percentile",
        percentiles=(2, 98),
        per_band=True,
        registered=False,
        registration_master="RED",
        max_shifts=(80, 80),
        force_registration=False,
        interpolation="nearest",
        figsize=(8, 8),
        title=None,
        out_png=None,
    ):
        """
        Display a PhiSat-2 L0 event.

        Args:
            bands: One band, three bands for RGB, or "all".
            normalize: Whether to normalize values for display.
            normalization: "percentile", "minmax", "zscore", or "none".
            percentiles: Percentiles used when normalization="percentile".
            per_band: Normalize each displayed band/channel independently.
            registered: If True, align bands to registration_master before display.
            registration_master: Master band used for display registration.
            max_shifts: Maximum allowed registration shift.
            force_registration: Recompute registration even if cached.
            interpolation: Matplotlib interpolation.
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
                level="L0",
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

 



    @classmethod
    def from_path(
        cls,
        product_folder: str,
        scene_id: int = 0,
        bands: Optional[Sequence[int]] = None,
        device: str = "cpu",
    ) -> "L0_event":
        """
        Create an :class:`L0_event` from a local ΦSat-2 L0 product folder.

        This constructor reads the raw TIFF bands, assembles them into a channel-first
        array of shape `(C, H, W)`, and builds the corresponding metadata dictionary.

        Args:
            product_folder: Path to the local ΦSat-2 L0 product folder.
            scene_id: Scene index to load from the product metadata.
            bands: Optional subset of band indices to load. If `None`, all available
                raw bands are loaded.
            device: Target device used when converting the event to a torch tensor.

        Returns:
            A new :class:`L0_event` instance initialized from disk.

        Raises:
            FileNotFoundError: If the expected files are missing.
            ValueError: If the metadata is invalid or inconsistent.
        """
        arr, meta = load_l0_stack(product_folder=product_folder, scene_id=scene_id, bands=bands)
        return cls(arr=arr, meta=meta, product_folder=product_folder, scene_id=scene_id, device=device)
    
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
        bands: Optional[Sequence[int]] = None,
        device: str = "cpu",
        convert: bool = True,
        converter: Optional[Callable[..., Union[str, Path]]] = None,
        converter_kwargs: Optional[Dict[str, Any]] = None,
        dest_dir: str | Path | None = None,
        local_fallback: bool = True,
        vm_fallback: bool = False,
        local_roots: Optional[List[str | Path]] = None,
        attach_catalog_geo: bool = True,
    ) -> Union["L0_event", Path]:
        search_roots = []

        if local_fallback:
            search_roots.append(dest_dir or DEFAULT_L0_DOWNLOAD_DIR)

        if vm_fallback:
            search_roots.append(VM_L0_ROOT)

        if local_roots:
            search_roots.extend(local_roots)

        existing = resolve_existing_product(
            identifier=identifier,
            roots=search_roots,
        )

        feature = None

        if existing is not None:
            downloaded_folder = Path(existing)
            if attach_catalog_geo:
                feature = client.get_feature_by_identifier(
                    ref_data_collection=ref_data_collection,
                    identifier=identifier,
                )
        else:
            feature = client.get_feature_by_identifier(
                ref_data_collection=ref_data_collection,
                identifier=identifier,
            )

            downloaded_folder = Path(
                client.download_feature(
                    feature,
                    dest_dir=dest_dir or DEFAULT_L0_DOWNLOAD_DIR,
                    extract=True,
                    keep_zip=keep_zip,
                    skip_existing=skip_existing,
                    force_redownload=force_redownload,
                )
            )

        if not convert:
            return downloaded_folder

        if _looks_prepared_for_l0_event(downloaded_folder):
            prepared_folder = downloaded_folder
        else:
            converter = converter or convert_l0_rawbin_inplace
            converter_kwargs = converter_kwargs or {}
            prepared_folder = Path(converter(downloaded_folder, **converter_kwargs))

        event = cls.from_path(
            product_folder=str(prepared_folder),
            scene_id=scene_id,
            bands=bands,
            device=device,
        )

        event.meta["source"] = "local_or_vm" if existing is not None else "insula"
        event.meta["resolved_product_folder"] = str(downloaded_folder)
        event.meta["prepared_product_folder"] = str(prepared_folder)
        event.meta["insula_converted"] = True

        if feature is not None:
            enrich_meta_with_insula_feature(
                meta=event.meta,
                feature=feature,
                ref_data_collection=ref_data_collection,
            )

        return event

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
        bands: Optional[Sequence[int]] = None,
        device: str = "cpu",
        convert: bool = True,
        converter: Optional[Callable[..., Union[str, Path]]] = None,
        converter_kwargs: Optional[Dict[str, Any]] = None,
        dest_dir: str | Path | None = None,
        **search_filters,
    ) -> Union["L0_event", Path]:
        """
        Search remote PHISAT-2 L0 products on Insula, download one result, and
        optionally convert it into a PyRawPh-readable local product.

        This is the low-level remote-search constructor. For high-level usage,
        prefer `client.search_l0(...)` followed by `client.load_l0(...)`.

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
            bands: Optional subset of L0 band indices to load.
            device: Device used by `as_tensor()`.
            convert: If True, convert `raw.bin` into raw TIFF bands before loading.
            converter: Optional custom converter callable.
            converter_kwargs: Optional kwargs forwarded to the converter.
            dest_dir: Optional destination directory for downloads.
            **search_filters: Additional Insula search parameters.

        Returns:
            If `convert=True`, a loaded `L0_event`.
            If `convert=False`, the downloaded/extracted product folder as a `Path`.

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

        downloaded_folder = Path(
            client.download_feature(
                feature,
                dest_dir=dest_dir or DEFAULT_L0_DOWNLOAD_DIR,
                extract=True,
                keep_zip=keep_zip,
                skip_existing=skip_existing,
                force_redownload=force_redownload,
            )
        )

        if not convert:
            return downloaded_folder

        if _looks_prepared_for_l0_event(downloaded_folder):
            prepared_folder = downloaded_folder
        else:
            converter = converter or convert_l0_rawbin_inplace
            converter_kwargs = converter_kwargs or {}
            prepared_folder = Path(converter(downloaded_folder, **converter_kwargs))

        event = cls.from_path(
            product_folder=str(prepared_folder),
            scene_id=scene_id,
            bands=bands,
            device=device,
        )

        event.meta["source"] = "insula"
        event.meta["resolved_product_folder"] = str(downloaded_folder)
        event.meta["prepared_product_folder"] = str(prepared_folder)
        event.meta["insula_converted"] = True

        enrich_meta_with_insula_feature(
            meta=event.meta,
            feature=feature,
            ref_data_collection=ref_data_collection,
        )

        return event

    def as_numpy(self) -> np.ndarray:
        """
        Return the event data as a NumPy array.

        Returns:
            The internal multispectral array with shape `(C, H, W)`.
        """
        return self.arr

    def as_tensor(self, as_float32: bool = True):
        """
        Return the event data as a PyTorch tensor on the configured device.

        Args:
            as_float32: If `True`, cast the tensor to `torch.float32` before moving it
                to the target device.

        Returns:
            A torch tensor containing the event data, typically with shape `(C, H, W)`.

        Raises:
            ImportError: If PyTorch is not available in the current environment.
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyTorch is required for as_tensor().") from exc

        x = torch.from_numpy(self.arr)
        if as_float32:
            x = x.float()
        return x.to(self.device)

    def get_meta(self) -> Dict[str, Any]:
        """
        Return the metadata dictionary associated with the event.

        Returns:
            The event metadata dictionary.
        """
        return self.meta
    
    def with_array(self, arr: np.ndarray, meta_updates: Optional[Dict[str, Any]] = None) -> "L0_event":
        """
        Create a new :class:`L0_event` with the same identity and updated array content.

        This helper is useful after registration, reprojection, filtering, or any other
        operation that produces a new `(C, H, W)` array but should preserve the product
        context and most of the metadata.

        Basic shape- and dtype-related metadata fields are refreshed automatically.
        Additional metadata updates can be supplied through `meta_updates`.

        Args:
            arr: New channel-first array with shape `(C, H, W)`.
            meta_updates: Optional metadata fields to override or add.

        Returns:
            A new :class:`L0_event` instance using the provided array.

        Raises:
            ValueError: If `arr` is not three-dimensional.
        """
        arr = np.asarray(arr)
        if arr.ndim != 3:
            raise ValueError(f"with_array expects (C,H,W), got {arr.shape}")

        new_meta = dict(self.meta)
        new_meta["count"] = int(arr.shape[0])
        new_meta["height"] = int(arr.shape[1])
        new_meta["width"] = int(arr.shape[2])
        new_meta["dtype"] = str(arr.dtype)

        if meta_updates is not None:
            new_meta.update(meta_updates)

        return L0_event(
            arr=arr,
            meta=new_meta,
            product_folder=self.product_folder,
            scene_id=self.scene_id,
            device=self.device,
        )

    def _resolve_band_index(self, band: BandSelector) -> int:
        """
        Resolve a band selector to a zero-based local channel index.

        Supported selectors include:
        - integer local channel indices,
        - float wavelengths in nanometers,
        - strings such as `"842nm"`, `"B3"`, or `"BAND_7"`,
        - aliases such as `"BLUE"`, `"GREEN"`, `"RED"`, `"RE1"`, `"RE2"`, `"RE3"`,
        and `"NIR"`.

        When wavelength-like selectors are used, the closest available wavelength in the
        metadata is selected.

        Args:
            band: Band selector to resolve.

        Returns:
            The resolved local channel index.

        Raises:
            ValueError: If the selector cannot be resolved.
        """
        band_indices = self.meta.get("band_indices", list(range(self.arr.shape[0])))
        wavelengths = np.asarray(self.meta.get("band_wavelength_nm", []), dtype=float)

        if isinstance(band, int):
            if 0 <= band < self.arr.shape[0]:
                return band
            raise ValueError(f"Invalid band index: {band}")

        if isinstance(band, float):
            if wavelengths.size == 0:
                raise ValueError("No band wavelengths available in metadata.")
            return int(np.argmin(np.abs(wavelengths - band)))

        if isinstance(band, str):
            s = band.strip().upper()

            if s.startswith("BAND_"):
                s = s.replace("BAND_", "B")
            if s.startswith("B") and s[1:].isdigit():
                raw_idx = int(s[1:])
                if raw_idx in band_indices:
                    return band_indices.index(raw_idx)
                raise ValueError(f"Band {s} is not present in this event.")

            if s.endswith("NM"):
                target = float(s[:-2])
                if wavelengths.size == 0:
                    raise ValueError("No band wavelengths available in metadata.")
                return int(np.argmin(np.abs(wavelengths - target)))

            if s in _ALIAS_TO_WAVELENGTH:
                if wavelengths.size == 0:
                    raise ValueError("No band wavelengths available in metadata.")
                target = _ALIAS_TO_WAVELENGTH[s]
                return int(np.argmin(np.abs(wavelengths - target)))

        raise ValueError(f"Could not resolve band selector: {band}")

    def get_band(self, band: BandSelector) -> np.ndarray:
        """
        Return one band from the event as a 2D array.

        Args:
            band: Band selector accepted by :meth:`_resolve_band_index`.

        Returns:
            A 2D NumPy array with shape `(H, W)` corresponding to the selected band.
        """
        idx = self._resolve_band_index(band)
        return self.arr[idx]

    def rgb(self, bands=("RED", "GREEN", "BLUE"), stretch=(2, 98), arr: Optional[np.ndarray] = None) -> np.ndarray:
        src = self.arr if arr is None else np.asarray(arr)
        if src.ndim != 3:
            raise ValueError(f"rgb expects (C,H,W), got {src.shape}")

        idxs = [self._resolve_band_index(b) for b in bands]
        rgb = np.stack([src[i] for i in idxs], axis=-1).astype(np.float32)

        lo = np.percentile(rgb, stretch[0], axis=(0, 1), keepdims=True)
        hi = np.percentile(rgb, stretch[1], axis=(0, 1), keepdims=True)
        rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
        return rgb

    def show_event_info(self) -> None:
        """
        Print a concise textual summary of the event.

        The summary includes product path information, array shape and dtype, band
        metadata, and key L0-specific acquisition fields such as band start rows and
        line period.

        Returns:
            None.
        """
        print("=== L0_event ===")
        print(f"Product folder      : {self.product_folder}")
        print(f"Scene ID            : {self.scene_id}")
        print(f"Array shape         : {self.arr.shape}")
        print(f"Dtype               : {self.arr.dtype}")
        print(f"Band indices        : {self.meta.get('band_indices')}")
        print(f"Band wavelengths nm : {self.meta.get('band_wavelength_nm')}")
        print(f"Band start rows     : {self.meta.get('band_start_row')}")
        print(f"Line period         : {self.meta.get('line_period')}")
        print(f"Native space        : {self.meta.get('native_space')}")
        print(f"Metadata path       : {self.meta.get('metadata_path')}")
        print(f"Ancillary path      : {self.meta.get('ancillary_path')}")

    def crop_px(self, y0: int, y1: int, x0: int, x1: int) -> "L0_event":
        """
        Crop the event in pixel coordinates and return a new :class:`L0_event`.

        The crop is defined by half-open intervals `[y0:y1, x0:x1]` in image
        coordinates. Bounds are clamped to the valid image extent.

        Args:
            y0: Start row index.
            y1: End row index (exclusive).
            x0: Start column index.
            x1: End column index (exclusive).

        Returns:
            A new :class:`L0_event` containing the cropped array and updated metadata.

        Raises:
            ValueError: If the resulting crop is empty or invalid.
        """
        y0 = max(0, int(y0))
        x0 = max(0, int(x0))
        y1 = min(int(y1), self.arr.shape[1])
        x1 = min(int(x1), self.arr.shape[2])

        if y1 <= y0 or x1 <= x0:
            raise ValueError("Invalid crop coordinates.")

        cropped = self.arr[:, y0:y1, x0:x1].copy()
        new_meta = dict(self.meta)
        new_meta["height"] = int(cropped.shape[1])
        new_meta["width"] = int(cropped.shape[2])
        new_meta["crop_box"] = [y0, y1, x0, x1]

        return L0_event(
            arr=cropped,
            meta=new_meta,
            product_folder=self.product_folder,
            scene_id=self.scene_id,
            device=self.device,
        )

    

    def register(self, register_fn: Callable[..., np.ndarray], **kwargs) -> "L0_event":
        """
        Grabiele Inzurillo's registration function.
        Apply a user-provided registration function to the full event cube.

        This is a lightweight generic wrapper intended for experimentation. The
        registration function receives the current array and metadata, and must return
        a new array with the same shape `(C, H, W)`.

        Args:
            register_fn: Callable implementing the registration logic.
            **kwargs: Additional keyword arguments forwarded to `register_fn`.

        Returns:
            A new :class:`L0_event` containing the registered array and updated
            registration metadata.

        Raises:
            ValueError: If the returned array shape does not match the input shape.
        """
        reg_arr = register_fn(self.arr, meta=self.meta, **kwargs)
        reg_arr = np.asarray(reg_arr)

        if reg_arr.shape != self.arr.shape:
            raise ValueError(
                f"Registration output shape mismatch: got {reg_arr.shape}, expected {self.arr.shape}"
            )
        
        new_meta = dict(self.meta)
        new_meta["native_space"] = "L0_registered_view"
        new_meta["parent_native_space"] = self.meta.get("native_space", "L0_native")
        new_meta["registration_info"] = {
            "method": getattr(register_fn, "__name__", "custom_registration"),
            "kwargs": kwargs,
        }

        return L0_event(
            arr=reg_arr,
            meta=new_meta,
            product_folder=self.product_folder,
            scene_id=self.scene_id,
            device=self.device,
        )
    
    def export_to_tif(self, out_path: str) -> None:
        """
        Export the event array to TIFF.

        If geospatial metadata (`crs` and `transform`) is available, the event is
        written as a GeoTIFF. Otherwise, a plain TIFF file is written without spatial
        reference.

        Args:
            out_path: Destination path of the output TIFF file.

        Returns:
            None.
        """
        if self.meta.get("transform", None) is not None and self.meta.get("crs", None) is not None:
            _export_to_tif(out_path, self.arr, self.meta)
        else:
            export_plain_tif(self.arr, out_path)