from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    cam_id: str
    frame_id: int
    t_capture: float
    image: np.ndarray


@dataclass(frozen=True)
class PoseResult:
    cam_id: str
    frame_id: int
    t_capture: float
    kpts_2d: np.ndarray
    kpts_3d_rel: np.ndarray
    kpts_3d_cam: np.ndarray
    fit_error_px: float
    t_infer_done: float
    inference_ms: float
    pose_age_ms: float
    source_meta: dict[str, Any]
    bbox_xyxy: np.ndarray | None = None
    detection_fresh: bool = True
    detection_confidence: float = 0.0


@dataclass(frozen=True)
class PersonObservation:
    cam_id: str
    frame_id: int
    t_capture: float
    kpts_2d: np.ndarray
    bbox_xyxy: np.ndarray | None
    confidence: float
    detection_fresh: bool
    image_shape: tuple[int, int, int]
    appearance_embedding: np.ndarray | None = None
    upper_hist: np.ndarray | None = None
    lower_hist: np.ndarray | None = None
    body_proportions: np.ndarray | None = None
    pose_result: PoseResult | None = None


@dataclass(frozen=True)
class MatchEvidence:
    frame_id_a: int
    frame_id_b: int
    skew_ms: float
    usable_joints: int
    epipolar_error_px: float | None
    appearance_score: float | None
    motion_score: float | None
    fused_score: float | None
    valid: bool
    reason: str
    geometry_source: str = "none"
    geometry_state: str = "none"
    geometry_matches: int = 0
    geometry_inliers: int = 0
    geometry_inlier_ratio: float = 0.0
    geometry_model_age_ms: float | None = None
    mode: str = "unknown"


class IdentityState(str, Enum):
    UNKNOWN = "unknown"
    SAME = "same"
    DIFFERENT = "different"


@dataclass(frozen=True)
class IdentityMatch:
    state: IdentityState
    shared_id: str | None
    evidence: MatchEvidence | None
    updated_at: float
