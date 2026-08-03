"""Simulation pipeline wrapping phisat2_utils tasks for on-the-fly ΦSat-2 synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime
import json
import traceback

import cv2
import numpy as np
import rasterio
from affine import Affine
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from sentinelhub.geometry import BBox
from sentinelhub.constants import CRS

from eolearn.core.eodata import EOPatch
from eolearn.core.constants import FeatureType
from eolearn.core.core_tasks import MapFeatureTask
from eolearn.features.utils import spatially_resize_image as resize_images

from .simulation_config import SimulationConfig
from .phisat2_utils import (
    AddPANBandTask,
    AddMetadataTask,
    BandMisalignmentTask,
    CalculateRadianceTask,
    CalculateReflectanceTask,
    AlternativePhisatCalculationTask,
    PhisatCalculationTask,
)
from .phisat2_constants import S2_RESOLUTION, PHISAT2_RESOLUTION, ProcessingLevels


PHIESTA_CROP_BAND_ORDER = [
    "B02",  # BLUE
    "B03",  # GREEN
    "B04",  # RED
    "B08",  # NIR broad
    "B05",  # RED EDGE 1
    "B06",  # RED EDGE 2
    "B07",  # RED EDGE 3
]

SIMULATED_OUTPUT_BAND_ORDER_WITH_PAN = [
    "B02_BLUE",
    "B03_GREEN",
    "B04_RED",
    "PAN",
    "B08_NIR",
    "B05_RED_EDGE_1",
    "B06_RED_EDGE_2",
    "B07_RED_EDGE_3",
]


class SimulationPipeline:
    """Run the OrbitalAI / PhiSat-2 simulation pipeline on Sentinel-2 crops."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_bbox_from_rasterio(self, src) -> BBox:
        bounds = src.bounds
        return BBox(
            bbox=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            crs=CRS(src.crs),
        )

    def _load_sen1floods_metadata(self, metadata_path: str):
        try:
            path = Path(metadata_path).expanduser().resolve()
            if not path.exists():
                print(f"Warning: Metadata file not found at {path}")
                return None

            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception as exc:
            print(f"Warning: Could not load metadata file: {exc}")
            return None

    def _get_acquisition_date_from_country(
        self,
        s2_tiff_path: Path | str,
        metadata: dict,
    ) -> Optional[datetime]:
        """Legacy Sen1Floods date extraction fallback."""
        if metadata is None or "features" not in metadata:
            return None

        try:
            location_name = Path(s2_tiff_path).stem.split("_")[0].lower()

            for feature in metadata.get("features", []):
                properties = feature.get("properties", {})
                location_property = properties.get("location", "").lower()

                if location_property == location_name:
                    s2_date_str = properties.get("s2_date")
                    if s2_date_str:
                        return datetime.strptime(s2_date_str, "%Y/%m/%d")

                if location_property == "cambodia" and location_name == "mekong":
                    s2_date_str = properties.get("s2_date")
                    if s2_date_str:
                        return datetime.strptime(s2_date_str, "%Y/%m/%d")

            return None

        except Exception as exc:
            print(f"Warning: Error extracting acquisition date from legacy metadata: {exc}")
            return None

    def _parse_acquisition_date(
        self,
        s2_tiff_path: Path | str,
        metadata: dict,
    ) -> datetime:
        """Prefer Phiesta metadata, then legacy metadata, then now()."""
        if isinstance(metadata, dict):
            dt_str = metadata.get("s2_datetime") or metadata.get("acquisition_datetime")
            if dt_str is not None:
                try:
                    return datetime.fromisoformat(
                        str(dt_str).replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except Exception as exc:
                    print(f"Warning: Could not parse s2_datetime={dt_str!r}: {exc}")

        legacy_dt = self._get_acquisition_date_from_country(s2_tiff_path, metadata)
        if legacy_dt is not None:
            return legacy_dt

        return datetime.now()

    def _inject_phiesta_metadata(self, eopatch: EOPatch, metadata: dict, spatial_shape) -> bool:
        """
        Inject metadata already extracted by Phiesta.

        This is intentionally defensive because the vendored phisat2_utils tasks
        may look for metadata under slightly different names.
        """
        if not isinstance(metadata, dict):
            return False

        ok = False

        earth_sun_dist = metadata.get("earth_sun_dist") or metadata.get("earth_sun_distance")
        if earth_sun_dist is not None:
            try:
                earth_sun_dist = float(earth_sun_dist)
                eopatch.meta_info["earth_sun_dist"] = earth_sun_dist
                eopatch.meta_info["earth_sun_distance"] = earth_sun_dist
                ok = True
            except Exception:
                pass

        solar_irradiances = metadata.get("solar_irradiances")
        if isinstance(solar_irradiances, dict) and solar_irradiances:
            eopatch.meta_info["solar_irradiances"] = solar_irradiances
            eopatch.meta_info["solarIrradiance"] = solar_irradiances
            ok = True

        sun_zenith = metadata.get("sun_zenith_angles")
        if sun_zenith is not None:
            try:
                sun_zenith = np.asarray(sun_zenith, dtype=np.float32)
                target_h, target_w = spatial_shape

                if sun_zenith.ndim == 2:
                    sun_zenith = resize(
                        sun_zenith,
                        (target_h, target_w),
                        order=1,
                        preserve_range=True,
                        anti_aliasing=True,
                    )

                elif sun_zenith.ndim == 3:
                    sun_zenith = sun_zenith[..., 0]
                    sun_zenith = resize(
                        sun_zenith,
                        (target_h, target_w),
                        order=1,
                        preserve_range=True,
                        anti_aliasing=True,
                    )

                else:
                    raise ValueError(f"Unexpected sun_zenith shape: {sun_zenith.shape}")

                eopatch[FeatureType.DATA, "sunZenithAngles"] = sun_zenith[
                    np.newaxis, :, :, np.newaxis
                ].astype(np.float32)
                ok = True

            except Exception as exc:
                print(f"Warning: Could not inject sun zenith angles: {exc}")

        return ok

    def _select_simulation_bands(self, s2_data: np.ndarray, metadata: dict | None) -> np.ndarray:
        """
        Return Sentinel-2 data in simulation order:
        B02, B03, B04, B08, B05, B06, B07.
        """
        if s2_data.ndim != 3:
            raise ValueError(f"Expected rasterio array with shape (C,H,W), got {s2_data.shape}")

        if s2_data.shape[0] == 7:
            # Phiesta crop already has the exact expected order.
            return s2_data

        if s2_data.shape[0] >= 8:
            # Legacy stack order used by the original simulation scripts.
            band_indices = [1, 2, 3, 7, 4, 5, 6]
            return s2_data[band_indices, :, :]

        raise ValueError(
            "Expected a 7-band Phiesta crop or an >=8-band Sentinel stack, "
            f"got shape {s2_data.shape}"
        )
 
    def simulate_single_file(
        self,
        s2_tiff_path: Path | str,
        output_tiff_path: Path | str,
        metadata: dict,
    ) -> bool:
        """
        Apply the PhiSat-2 simulation pipeline to one Sentinel-2 crop.

        Args:
            s2_tiff_path: Sentinel-2 GeoTIFF. For Phiesta, expected order is
                B02, B03, B04, B08, B05, B06, B07.
            output_tiff_path: Output simulated PhiSat-2 GeoTIFF.
            metadata: Phiesta metadata JSON dictionary.

        Returns:
            True if successful, False otherwise.
        """
        try:
            s2_tiff_path = Path(s2_tiff_path)
            output_tiff_path = Path(output_tiff_path)
            output_tiff_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(s2_tiff_path) as src:
                s2_data = src.read().astype(np.float32)
                profile = src.profile.copy()
                src_height = src.height
                src_width = src.width

                try:
                    bbox = self._create_bbox_from_rasterio(src)
                except Exception as exc:
                    print(f"Warning: Could not extract bbox from {s2_tiff_path}: {exc}")
                    bbox = None

            s2_data = self._select_simulation_bands(s2_data, metadata)

            target_size = metadata.get("target_size", None) if isinstance(metadata, dict) else None
            if target_size is not None:
                target_size = tuple(target_size)
                if s2_data.shape[1:] != target_size:
                    s2_data = resize(
                        s2_data,
                        (s2_data.shape[0], *target_size),
                        order=1,
                        preserve_range=True,
                        anti_aliasing=True,
                    ).astype(np.float32)

            # Rasterio format (C,H,W) -> EO-Learn format (H,W,C)
            s2_data = np.transpose(s2_data, (1, 2, 0)).astype(np.float32)
            print(f"Loaded S2 data with shape {s2_data.shape}")

            eopatch = EOPatch(bbox=bbox)
            acquisition_date = self._parse_acquisition_date(s2_tiff_path, metadata)
            eopatch.timestamp = [acquisition_date]

            eopatch[FeatureType.DATA, "S2_BANDS"] = s2_data[np.newaxis, :, :, :]

            metadata_injected = self._inject_phiesta_metadata(
                eopatch=eopatch,
                metadata=metadata,
                spatial_shape=s2_data.shape[:2],
            )

            try:
                add_meta_task = AddMetadataTask()
                eopatch = add_meta_task.execute(eopatch)
                metadata_injected = True
            except Exception as exc:
                print(f"Warning: AddMetadataTask failed: {exc}")
                if not metadata_injected:
                    print("Skipping radiance conversion because metadata is missing.")
                    self.config.steps.radiance = False
                else:
                    print("Continuing with Phiesta-provided metadata.")

            # Radiance conversion
            if self.config.steps.radiance:
                radiance_task = CalculateRadianceTask(
                    (FeatureType.DATA, "S2_BANDS"),
                    (FeatureType.DATA, "S2_RADIANCE"),
                )
                eopatch = radiance_task.execute(eopatch)
                current_feature = "S2_RADIANCE"
            else:
                current_feature = "S2_BANDS"

            # Add panchromatic band
            if self.config.steps.add_panchromatic:
                pan_task = AddPANBandTask(
                    (FeatureType.DATA, current_feature),
                    (FeatureType.DATA, "BANDS-RAD-PAN"),
                )
                eopatch = pan_task.execute(eopatch)
                current_feature = "BANDS-RAD-PAN"

            # Spatial resampling to PhiSat-2 pixel size
            features_to_resize = {FeatureType.DATA: [current_feature]}

            if "sunZenithAngles" in eopatch.data:
                features_to_resize[FeatureType.DATA].append("sunZenithAngles")

            h, w = eopatch[FeatureType.DATA, current_feature].shape[1:3]
            new_size = (
                int(round(h * (S2_RESOLUTION / PHISAT2_RESOLUTION))),
                int(round(w * (S2_RESOLUTION / PHISAT2_RESOLUTION))),
            )

            for feature_type, feature_names in features_to_resize.items():
                for feature in feature_names:
                    resize_task = MapFeatureTask(
                        (feature_type, feature),
                        (feature_type, f"{feature}_RES"),
                        resize_images,
                        new_size=new_size,
                        resize_method="nearest",
                    )
                    eopatch = resize_task(eopatch)

            current_feature = f"{current_feature}_RES"

            # Band misalignment
            if self.config.steps.band_misalignment:
                processing_level = ProcessingLevels[self.config.processing_level]
                misalign_task = BandMisalignmentTask(
                    (FeatureType.DATA, current_feature),
                    (FeatureType.DATA, "S2_MISALIGNED"),
                    processing_level=processing_level,
                    std_sea=self.config.misalignment_std_sea,
                    interpolation_method=cv2.INTER_NEAREST,
                )
                eopatch = misalign_task.execute(eopatch)
                current_feature = "S2_MISALIGNED"

            # SNR + PSF simulation
            if self.config.steps.snr_simulation or self.config.steps.psf_filtering:
                use_executable = (
                    self.config.snr_psf_method == "executable"
                    and self.config.phisat2_exec_path
                )

                if use_executable:
                    if self.config.steps.snr_simulation:
                        snr_task = PhisatCalculationTask(
                            input_feature=(FeatureType.DATA, current_feature),
                            output_feature=(FeatureType.DATA, "L_out_SNR"),
                            executable=self.config.phisat2_exec_path,
                            calculation="SNR",
                        )
                        eopatch = snr_task.execute(eopatch)
                        current_feature = "L_out_SNR"

                    if self.config.steps.psf_filtering:
                        psf_task = PhisatCalculationTask(
                            input_feature=(FeatureType.DATA, current_feature),
                            output_feature=(FeatureType.DATA, "L_out_PSF"),
                            executable=self.config.phisat2_exec_path,
                            calculation="PSF",
                        )
                        eopatch = psf_task.execute(eopatch)
                        current_feature = "L_out_PSF"

                elif self.config.snr_values:
                    psf_kernel = self._create_psf_kernels(self.config.psf_kernel_sigma)

                    snr_psf_task = AlternativePhisatCalculationTask(
                        input_feature=(FeatureType.DATA, current_feature),
                        snr_feature=(
                            (FeatureType.DATA, "S2_NOISY")
                            if self.config.steps.snr_simulation
                            else (FeatureType.DATA, current_feature)
                        ),
                        snr_values=self.config.snr_values,
                        l_ref=self.config.radiance_reference,
                        psf_feature=(FeatureType.DATA, "S2_PSF"),
                        psf_kernel=psf_kernel,
                    )
                    eopatch = snr_psf_task.execute(eopatch)
                    current_feature = "S2_PSF"

            # Reflectance conversion
            if (
                self.config.steps.reflectance_conversion
                and self.config.processing_level == "L1C"
            ):
                reflectance_task = CalculateReflectanceTask(
                    (FeatureType.DATA, current_feature),
                    (FeatureType.DATA, "S2_REFLECTANCE"),
                    processing_level=ProcessingLevels.L1C,
                )
                eopatch = reflectance_task.execute(eopatch)
                current_feature = "S2_REFLECTANCE"

            output_data = eopatch[FeatureType.DATA, current_feature][0]
            output_data = np.transpose(output_data, (2, 0, 1)).astype(np.float32)

            out_height = output_data.shape[1]
            out_width = output_data.shape[2]

            transform = profile["transform"]
            scale_x = src_width / out_width
            scale_y = src_height / out_height
            out_transform = transform * Affine.scale(scale_x, scale_y)

            profile.update(
                driver="GTiff",
                height=out_height,
                width=out_width,
                count=output_data.shape[0],
                dtype="float32",
                transform=out_transform,
                compress="deflate",
                bigtiff="if_safer",
            )

            with rasterio.open(output_tiff_path, "w", **profile) as dst:
                dst.write(output_data)

                if output_data.shape[0] == len(SIMULATED_OUTPUT_BAND_ORDER_WITH_PAN):
                    dst.descriptions = tuple(SIMULATED_OUTPUT_BAND_ORDER_WITH_PAN)

            return True

        except Exception as exc:
            print(f"Error simulating {s2_tiff_path}: {exc}")
            traceback.print_exc()
            return False

    def _create_psf_kernels(self, sigma) -> dict:
        """Create approximate Gaussian PSF kernels for all PhiSat-2 bands."""
        kernel_bands = ["B1", "B2", "B3", "B0", "B7", "B4", "B5", "B6"]
        psf_kernels = {}

        for band in kernel_bands:
            kernel = np.zeros((7, 7))
            kernel[3, 3] = 1
            kernel = gaussian_filter(kernel, sigma)
            kernel = kernel / kernel.sum()
            psf_kernels[band] = kernel

        return psf_kernels

    def batch_simulate_from_source_dir(
        self,
        source_dir: Optional[Path | str] = None,
        pattern: str = "*.tiff",
        metadata_path: Optional[str] = None,
    ) -> dict:
        """
        Apply simulation to all Sentinel-2 TIFF files in a source directory.

        This method is kept for compatibility with the original scripts.
        Phiesta mainly uses simulate_single_file(...).
        """
        source_dir = Path(source_dir or self.config.s2_source_dir)
        results = {"successful": [], "failed": []}

        metadata = None
        if metadata_path:
            metadata = self._load_sen1floods_metadata(metadata_path)

        for s2_file in sorted(source_dir.glob(pattern)):
            output_file = self.config.output_dir / f"simulated_{s2_file.name}"
            success = self.simulate_single_file(s2_file, output_file, metadata)

            if success:
                results["successful"].append(str(output_file))
                print(f"✓ Simulated: {s2_file.name}")
            else:
                results["failed"].append(str(s2_file))
                print(f"✗ Failed: {s2_file.name}")

        print(
            f"\nSummary: {len(results['successful'])} successful, "
            f"{len(results['failed'])} failed"
        )
        return results