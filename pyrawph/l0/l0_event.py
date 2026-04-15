from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

from ..utils.export_utils import export_to_tif as _export_to_tif, export_plain_tif


from .reader import load_l0_stack


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
    
    def _normalize_for_display(self, img: np.ndarray, stretch=(2, 98)) -> np.ndarray:
        """
        Normalize one 2D image for display using a percentile stretch.

        This helper converts the input image to `float32`, computes lower and upper
        percentiles, and rescales the values to `[0, 1]`. It is intended only for
        visualization and does not modify the event data.

        Args:
            img: Input 2D image to normalize for display.
            stretch: Two-element tuple `(p_lo, p_hi)` defining the percentiles used for
                contrast stretching.

        Returns:
            A 2D `float32` array normalized to `[0, 1]`.
        """
        img = img.astype(np.float32)
        lo = np.percentile(img, stretch[0])
        hi = np.percentile(img, stretch[1])
        return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    def plot_band(self, band=0, stretch=(2, 98), figsize=(6, 6), title=None):
        """
        Display one band of the event as a grayscale image.

        The selected band is normalized with a percentile stretch before display.

        Args:
            band: Band selector accepted by :meth:`get_band`.
            stretch: Two-element tuple `(p_lo, p_hi)` used for contrast stretching.
            figsize: Matplotlib figure size.
            title: Optional plot title. If `None`, a default title is used.

        Returns:
            None.
        """
        img = self.get_band(band)
        disp = self._normalize_for_display(img, stretch=stretch)

        plt.figure(figsize=figsize)
        plt.imshow(disp, cmap="gray")
        plt.axis("off")
        plt.title(title or f"Band {band}")
        plt.show()

    def plot_rgb(self, bands=(0, 2, 7), stretch=(2, 98), figsize=(8, 8), title=None):
        """
        Display a quick RGB-like composite built from three selected bands.

        This method is intended for debugging and qualitative inspection. The result is
        not guaranteed to be a physically correct true-color rendering.

        Args:
            bands: Three band selectors describing the red, green, and blue channels.
            stretch: Two-element tuple `(p_lo, p_hi)` used for contrast stretching.
            figsize: Matplotlib figure size.
            title: Optional plot title. If `None`, a default title is used.

        Returns:
            None.
        """
        rgb = self.rgb(bands=bands, stretch=stretch)

        plt.figure(figsize=figsize)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(title or f"RGB-like {bands}")
        plt.show()

    def plot_all_bands(self, stretch=(2, 98), figsize=(16, 8)):
        """
        Display all bands of the event in a 2x4 grid.

        Each band is shown independently after percentile normalization. If wavelength
        metadata is available, the wavelength is included in the subplot title.

        Args:
            stretch: Two-element tuple `(p_lo, p_hi)` used for contrast stretching.
            figsize: Matplotlib figure size.

        Returns:
            None.
        """
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.ravel()

        for i in range(self.arr.shape[0]):
            disp = self._normalize_for_display(self.arr[i], stretch=stretch)
            axes[i].imshow(disp, cmap="gray")
            axes[i].set_title(
                f"B{i} - {self.meta.get('band_wavelength_nm', ['?']*self.arr.shape[0])[i]} nm"
            )
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

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

    def rgb(self, bands=(0, 2, 7), stretch=(2, 98), arr: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Debug RGB-like composite for L0.
        Default is (band0, band2, band7), not a guaranteed physical true-color.
        """
        src = self.arr if arr is None else np.asarray(arr)
        if src.ndim != 3:
            raise ValueError(f"rgb expects (C,H,W), got {src.shape}")

        rgb = np.stack([src[b] for b in bands], axis=-1).astype(np.float32)
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