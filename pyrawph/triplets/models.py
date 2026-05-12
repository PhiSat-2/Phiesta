from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import json
import numpy as np


@dataclass
class SentinelSource:
    """
    Metadata for the Sentinel-2 source selected for a PhiSat-2 acquisition.
    """

    product_id: str
    satellite: str = "S2B"
    s2_datetime: Optional[str] = None
    delta_days: Optional[float] = None
    cloud_cover: Optional[float] = None
    coverage: Optional[float] = None
    l1c_paths: list[str] = field(default_factory=list)
    l2a_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SentinelCropResult:
    """
    Result of cropping Sentinel-2 around the PhiSat-2 catalog footprint.
    """

    crop_path: Optional[str] = None
    metadata_path: Optional[str] = None
    cloud_mask_path: Optional[str] = None
    buffer_km: float = 10.0
    resolution_m: float = 10.0
    bands: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """
    Result of simulating PhiSat-2-like data from a Sentinel-2 crop.
    """

    simulated_path: Optional[str] = None
    phisat2_exec_path: Optional[str] = None
    processing_level: str = "L1C"
    backend: str = "executable"
    band_order: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentResult:
    """
    Geometric alignment result between real PhiSat-2 and simulated PhiSat-2.
    """

    status: str = "NOT_RUN"
    transform_model: str = "homography"
    match_band: str = "PAN"

    homography: Optional[list[list[float]]] = None
    affine: Optional[list[list[float]]] = None

    num_keypoints_real: Optional[int] = None
    num_keypoints_simulated: Optional[int] = None
    num_matches: Optional[int] = None
    num_inliers: Optional[int] = None
    inlier_ratio: Optional[float] = None

    mean_reprojection_error: Optional[float] = None
    median_reprojection_error: Optional[float] = None

    aligned_simulated_path: Optional[str] = None
    aligned_sentinel_path: Optional[str] = None
    aligned_mask_path: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TripletPaths:
    """
    Standard output paths for one PhiSat-2 / Sentinel-2 / simulated triplet.
    """

    root_dir: str
    sentinel_dir: str
    simulated_dir: str
    aligned_dir: str
    qc_path: str

    @classmethod
    def from_root(cls, root_dir: str | Path) -> "TripletPaths":
        root = Path(root_dir)
        return cls(
            root_dir=str(root),
            sentinel_dir=str(root / "sentinel"),
            simulated_dir=str(root / "simulated"),
            aligned_dir=str(root / "aligned"),
            qc_path=str(root / "qc.json"),
        )

    def make_dirs(self) -> None:
        Path(self.sentinel_dir).mkdir(parents=True, exist_ok=True)
        Path(self.simulated_dir).mkdir(parents=True, exist_ok=True)
        Path(self.aligned_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class TripletResult:
    """
    Full result object returned by build_sentinel_triplet(...).
    """

    product_id: str
    status: str = "NOT_RUN"

    paths: Optional[TripletPaths] = None
    sentinel_source: Optional[SentinelSource] = None
    sentinel_crop: Optional[SentinelCropResult] = None
    simulation: Optional[SimulationResult] = None
    alignment: Optional[AlignmentResult] = None

    qc: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_qc(self, path: str | Path | None = None) -> Path:
        if path is None:
            if self.paths is None:
                raise ValueError("No qc path provided and self.paths is None.")
            path = self.paths.qc_path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        return path

    def summary(self) -> dict[str, Any]:
        out = {
            "product_id": self.product_id,
            "status": self.status,
        }

        if self.sentinel_source is not None:
            out.update(
                {
                    "satellite": self.sentinel_source.satellite,
                    "delta_days": self.sentinel_source.delta_days,
                    "cloud_cover": self.sentinel_source.cloud_cover,
                    "coverage": self.sentinel_source.coverage,
                }
            )

        if self.alignment is not None:
            out.update(
                {
                    "transform_model": self.alignment.transform_model,
                    "match_band": self.alignment.match_band,
                    "num_matches": self.alignment.num_matches,
                    "num_inliers": self.alignment.num_inliers,
                    "inlier_ratio": self.alignment.inlier_ratio,
                    "mean_reprojection_error": self.alignment.mean_reprojection_error,
                }
            )

        return out