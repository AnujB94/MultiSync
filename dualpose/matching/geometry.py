from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from dualpose.calibration import StereoCalibration
from dualpose.utils.types import PersonObservation


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    c = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    idx = int(np.searchsorted(c, cutoff, side="left"))
    idx = max(0, min(idx, len(v) - 1))
    return float(v[idx])


@dataclass(frozen=True)
class GeometryModel:
    f: np.ndarray
    source: str  # "calibrated" | "auto"
    state: str = "healthy"
    inliers: int = 0
    matches: int = 0
    inlier_ratio: float = 0.0
    median_error_px: float = 0.0
    p90_error_px: float = 0.0
    updated_at: float = 0.0
    k1: np.ndarray | None = None
    d1: np.ndarray | None = None
    k2: np.ndarray | None = None
    d2: np.ndarray | None = None

    @property
    def age_ms(self) -> float:
        if self.updated_at <= 0:
            return 0.0
        return max(0.0, (time.monotonic() - self.updated_at) * 1000.0)


class EpipolarScorer:
    def __init__(
        self,
        calibration: StereoCalibration | None,
        min_common_joints: int,
        sigma_px: float,
        hard_reject_px: float,
    ) -> None:
        self.calib = calibration
        self.min_common_joints = min_common_joints
        self.sigma_px = sigma_px
        self.hard_reject_px = hard_reject_px

    def _undistort(self, pts: np.ndarray, k: np.ndarray | None, d: np.ndarray | None) -> np.ndarray:
        if k is None or d is None:
            return np.asarray(pts, dtype=np.float64)
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, k, d, P=k)
        return und.reshape(-1, 2)

    def _sampson(self, x1: np.ndarray, x2: np.ndarray, f: np.ndarray) -> np.ndarray:
        ones = np.ones((x1.shape[0], 1), dtype=np.float64)
        x1h = np.concatenate([x1, ones], axis=1)
        x2h = np.concatenate([x2, ones], axis=1)
        fx1 = x1h @ f.T
        ftx2 = x2h @ f
        numer = np.sum(x2h * (x1h @ f.T), axis=1)
        denom = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2 + 1e-9
        return np.abs(numer) / np.sqrt(denom)

    def score(
        self,
        obs_a: PersonObservation,
        obs_b: PersonObservation,
        model: GeometryModel | None,
    ) -> tuple[float | None, float | None, int, bool, str]:
        if model is None:
            return None, None, 0, False, "no_geometry_model"
        conf_a = obs_a.kpts_2d[:, 2]
        conf_b = obs_b.kpts_2d[:, 2]
        mask = (conf_a > 0.05) & (conf_b > 0.05)
        idx = np.where(mask)[0]
        if idx.size < self.min_common_joints:
            return None, None, int(idx.size), False, "insufficient_common_joints"

        pts_a = obs_a.kpts_2d[idx, :2]
        pts_b = obs_b.kpts_2d[idx, :2]
        if model.source == "calibrated":
            und_a = self._undistort(pts_a, model.k1, model.d1)
            und_b = self._undistort(pts_b, model.k2, model.d2)
        else:
            und_a = np.asarray(pts_a, dtype=np.float64)
            und_b = np.asarray(pts_b, dtype=np.float64)
        errs = self._sampson(und_a, und_b, model.f)

        conf = np.minimum(conf_a[idx], conf_b[idx]).astype(np.float64)
        # Torso/core joints are typically more stable.
        torso_ids = {0, 1, 4, 7, 8, 11, 14}
        boost = np.array([1.5 if int(i) in torso_ids else 1.0 for i in idx], dtype=np.float64)
        weights = np.clip(conf * boost, 0.05, 2.0)

        median_error = _weighted_median(errs, weights)
        hard_fail = bool(median_error > self.hard_reject_px)
        score = float(np.exp(-0.5 * (median_error / max(self.sigma_px, 1e-6)) ** 2))
        reason = "ok" if not hard_fail else "hard_epipolar_reject"
        return score, median_error, int(idx.size), hard_fail, reason

    @staticmethod
    def calibrated_model(calibration: StereoCalibration | None) -> GeometryModel | None:
        if calibration is None:
            return None
        return GeometryModel(
            f=np.asarray(calibration.f, dtype=np.float64),
            source="calibrated",
            state="healthy",
            inliers=0,
            matches=0,
            inlier_ratio=1.0,
            median_error_px=float(calibration.metrics.get("stereo_rms", 0.0)),
            p90_error_px=float(calibration.metrics.get("stereo_rms", 0.0)),
            updated_at=time.monotonic(),
            k1=np.asarray(calibration.k1, dtype=np.float64),
            d1=np.asarray(calibration.d1, dtype=np.float64),
            k2=np.asarray(calibration.k2, dtype=np.float64),
            d2=np.asarray(calibration.d2, dtype=np.float64),
        )
