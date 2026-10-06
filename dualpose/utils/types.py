from __future__ import annotations

from dataclasses import dataclass
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
