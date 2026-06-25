from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


LatLon = Tuple[float, float]


@dataclass
class SentinelMosaic:
    image: np.ndarray
    transform: Any
    crs: Any
    bbox_latlon: Tuple[float, float, float, float]
    acquisition_datetime: Optional[str] = None
    source_path: Optional[str] = None


@dataclass
class Candidate:
    x: int
    y: int
    w: int
    h: int
    coarse_score: float
    coarse_corr: float = float("nan")
    scale: float = 1.0
    angle: float = 0.0
    orientation: str = "identity"
    method: str = "template"
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoarseResult:
    candidate: Candidate
    candidates: List[Candidate]
    query_shape: Tuple[int, int]
    query_transform_feat_to_query: np.ndarray  # (3,3)
    phi_query: np.ndarray
    s2_crop: np.ndarray


@dataclass
class RefinementResult:
    ok: bool
    method: str
    affine_query_to_crop: np.ndarray  # (2,3)
    scores: Dict[str, float] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    similarity: Dict[str, float] = field(default_factory=dict)
    geometry: Dict[str, float] = field(default_factory=dict)
    polygon: Dict[str, float] = field(default_factory=dict)


@dataclass
class RelocalizationResult:
    corners_latlon: Dict[str, LatLon]
    center_latlon: LatLon

    affine_phi_to_s2: np.ndarray  # (2,3), original L1 pixel -> S2 pixel
    T_total: np.ndarray           # (3,3), original L1 pixel -> S2 pixel

    pixel_corners_in_s2: np.ndarray
    world_transform: Any

    scores: Dict[str, float] = field(default_factory=dict)
    status: str = "unknown"

    matched_s2_datetime: Optional[str] = None
    reference_product: Optional[str] = None
    search_bbox_latlon: Optional[Tuple[float, float, float, float]] = None

    coarse_method: Optional[str] = None
    refiner_method: Optional[str] = None
    coarse_candidate: Optional[Candidate] = None
    coarse_candidates: List[Candidate] = field(default_factory=list)

    crop_xywh: Optional[Tuple[int, int, int, int]] = None
    scale: Optional[float] = None
    angle: Optional[float] = None
    orientation: Optional[str] = None
    coarse_corr: Optional[float] = None
    final_query_shape: Optional[Tuple[int, int]] = None

    evaluation: Optional[EvaluationMetrics] = None
    acq_id: Optional[int] = None
    mosaic_path: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)
