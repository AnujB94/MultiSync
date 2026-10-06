from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class PlacementResult:
    kpts_3d_cam: np.ndarray
    reprojection_error_px: float


class PlacementFitter:
    def __init__(self, weak_perspective: bool, fx_scale: float, fy_scale: float) -> None:
        self.weak_perspective = weak_perspective
        self.fx_scale = fx_scale
        self.fy_scale = fy_scale
        self._last_params: dict[str, np.ndarray] = {}

    def _weights(self, conf: np.ndarray) -> np.ndarray:
        return np.clip(conf, 0.05, 1.0)

    def _fit_weak(
        self,
        cam_id: str,
        kpts_3d_rel: np.ndarray,
        kpts_2d: np.ndarray,
    ) -> PlacementResult:
        conf = kpts_2d[:, 2]
        w = self._weights(conf)
        target = kpts_2d[:, :2]
        src = kpts_3d_rel[:, :2]

        x0 = self._last_params.get(
            cam_id,
            np.array([1.0, target[0, 0], target[0, 1]], dtype=np.float64),
        )

        def residuals(params: np.ndarray) -> np.ndarray:
            s, tx, ty = params
            pred = src * s + np.array([tx, ty], dtype=np.float64)
            r = (pred - target) * w[:, None]
            return r.reshape(-1)

        sol = least_squares(residuals, x0, max_nfev=30)
        self._last_params[cam_id] = sol.x
        s, tx, ty = sol.x
        k_cam = np.zeros((17, 3), dtype=np.float32)
        k_cam[:, :2] = src * s + np.array([tx, ty], dtype=np.float64)
        k_cam[:, 2] = kpts_3d_rel[:, 2] * s + 1000.0

        reproj = k_cam[:, :2]
        err = float(np.sqrt(np.mean(np.sum((reproj - target) ** 2, axis=1))))
        return PlacementResult(kpts_3d_cam=k_cam, reprojection_error_px=err)

    def _fit_pinhole(
        self,
        cam_id: str,
        kpts_3d_rel: np.ndarray,
        kpts_2d: np.ndarray,
        image_shape: tuple[int, int, int],
    ) -> PlacementResult:
        h, w_img = image_shape[:2]
        fx = self.fx_scale * w_img
        fy = self.fy_scale * w_img
        cx = w_img * 0.5
        cy = h * 0.5

        conf = kpts_2d[:, 2]
        w = self._weights(conf)
        target = kpts_2d[:, :2].astype(np.float64)
        src = kpts_3d_rel.astype(np.float64)

        x0 = self._last_params.get(
            cam_id,
            np.array([250.0, cx - src[0, 0], cy - src[0, 1], 2500.0], dtype=np.float64),
        )

        def project(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            s, tx, ty, tz = params
            pts = src * s + np.array([tx, ty, tz], dtype=np.float64)
            z = np.maximum(pts[:, 2], 1.0)
            u = fx * (pts[:, 0] / z) + cx
            v = fy * (pts[:, 1] / z) + cy
            return np.stack([u, v], axis=1), pts

        def residuals(params: np.ndarray) -> np.ndarray:
            proj, _ = project(params)
            r = (proj - target) * w[:, None]
            return r.reshape(-1)

        sol = least_squares(
            residuals,
            x0,
            bounds=([1.0, -10000.0, -10000.0, 100.0], [5000.0, 10000.0, 10000.0, 50000.0]),
            max_nfev=50,
        )
        self._last_params[cam_id] = sol.x
        proj, pts = project(sol.x)
        err = float(np.sqrt(np.mean(np.sum((proj - target) ** 2, axis=1))))
        return PlacementResult(kpts_3d_cam=pts.astype(np.float32), reprojection_error_px=err)

    def fit(
        self,
        cam_id: str,
        kpts_3d_rel: np.ndarray,
        kpts_2d: np.ndarray,
        image_shape: tuple[int, int, int],
    ) -> PlacementResult:
        if self.weak_perspective:
            return self._fit_weak(cam_id=cam_id, kpts_3d_rel=kpts_3d_rel, kpts_2d=kpts_2d)
        return self._fit_pinhole(
            cam_id=cam_id,
            kpts_3d_rel=kpts_3d_rel,
            kpts_2d=kpts_2d,
            image_shape=image_shape,
        )
