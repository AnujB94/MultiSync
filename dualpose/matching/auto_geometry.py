from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from dualpose.config import MatchingConfig
from dualpose.matching.geometry import GeometryModel


@dataclass(frozen=True)
class AutoGeometryStatus:
    state: str
    reason: str
    matches: int
    inliers: int
    inlier_ratio: float
    median_error_px: float
    p90_error_px: float
    updated_at: float


def _sampson_err(pts_a: np.ndarray, pts_b: np.ndarray, f: np.ndarray) -> np.ndarray:
    ones = np.ones((pts_a.shape[0], 1), dtype=np.float64)
    xa = np.concatenate([pts_a.astype(np.float64), ones], axis=1)
    xb = np.concatenate([pts_b.astype(np.float64), ones], axis=1)
    fx = xa @ f.T
    ftx = xb @ f
    numer = np.sum(xb * (xa @ f.T), axis=1)
    denom = fx[:, 0] ** 2 + fx[:, 1] ** 2 + ftx[:, 0] ** 2 + ftx[:, 1] ** 2 + 1e-9
    return np.abs(numer) / np.sqrt(denom)


class AutoGeometryEstimator:
    def __init__(self, cfg: MatchingConfig) -> None:
        self.cfg = cfg
        self.model: GeometryModel | None = None
        self.status = AutoGeometryStatus(
            state="initializing",
            reason="startup",
            matches=0,
            inliers=0,
            inlier_ratio=0.0,
            median_error_px=0.0,
            p90_error_px=0.0,
            updated_at=time.monotonic(),
        )
        self._failure_count = 0
        self._last_estimate_ts = 0.0
        self._last_health_ts = 0.0

    def _new_detector(self):
        if self.cfg.auto_feature_type == "orb":
            return cv2.ORB_create(
                nfeatures=self.cfg.auto_max_features,
                fastThreshold=15,
            )
        # Fallback to ORB when SIFT is unavailable on current build.
        try:
            return cv2.SIFT_create(nfeatures=self.cfg.auto_max_features)
        except Exception:
            return cv2.ORB_create(
                nfeatures=self.cfg.auto_max_features,
                fastThreshold=15,
            )

    def _mask_from_bbox(
        self,
        shape: tuple[int, int],
        bbox_xyxy: np.ndarray | None,
    ) -> np.ndarray | None:
        if bbox_xyxy is None:
            return None
        h, w = shape
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        pad_x = (x2 - x1) * self.cfg.auto_person_mask_pad_ratio
        pad_y = (y2 - y1) * self.cfg.auto_person_mask_pad_ratio
        x1 = int(max(0, min(w - 1, round(x1 - pad_x))))
        x2 = int(max(0, min(w - 1, round(x2 + pad_x))))
        y1 = int(max(0, min(h - 1, round(y1 - pad_y))))
        y2 = int(max(0, min(h - 1, round(y2 + pad_y))))
        if x2 <= x1 or y2 <= y1:
            return None
        mask = np.full((h, w), 255, dtype=np.uint8)
        mask[y1:y2, x1:x2] = 0
        return mask

    def _detect(
        self,
        image_bgr: np.ndarray,
        person_mask: np.ndarray | None,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        detector = self._new_detector()
        return detector.detectAndCompute(gray, person_mask)

    def _mutual_knn_matches(
        self,
        des_a: np.ndarray,
        des_b: np.ndarray,
        use_hamming: bool,
    ) -> list[tuple[int, int, float]]:
        if des_a is None or des_b is None:
            return []
        if len(des_a) < 4 or len(des_b) < 4:
            return []

        norm = cv2.NORM_HAMMING if use_hamming else cv2.NORM_L2
        bf = cv2.BFMatcher(normType=norm, crossCheck=False)
        ab = bf.knnMatch(des_a, des_b, k=2)
        ba = bf.knnMatch(des_b, des_a, k=2)

        ratio = self.cfg.auto_lowe_ratio
        ab_best: dict[int, tuple[int, float]] = {}
        for row in ab:
            if len(row) < 2:
                continue
            m, n = row
            if m.distance < ratio * n.distance:
                ab_best[m.queryIdx] = (m.trainIdx, float(m.distance))

        ba_best: dict[int, int] = {}
        for row in ba:
            if len(row) < 2:
                continue
            m, n = row
            if m.distance < ratio * n.distance:
                ba_best[m.queryIdx] = m.trainIdx

        mutual: list[tuple[int, int, float]] = []
        for qa, (tb, dist) in ab_best.items():
            if ba_best.get(tb) == qa:
                mutual.append((qa, tb, dist))
        return mutual

    def _coverage_ok(
        self,
        pts_a: np.ndarray,
        pts_b: np.ndarray,
        width: int,
        height: int,
    ) -> bool:
        rows = self.cfg.auto_grid_rows
        cols = self.cfg.auto_grid_cols
        used_a = set()
        used_b = set()
        for pa, pb in zip(pts_a, pts_b, strict=False):
            ca = int(np.clip(pa[0] / max(width, 1) * cols, 0, cols - 1))
            ra = int(np.clip(pa[1] / max(height, 1) * rows, 0, rows - 1))
            cb = int(np.clip(pb[0] / max(width, 1) * cols, 0, cols - 1))
            rb = int(np.clip(pb[1] / max(height, 1) * rows, 0, rows - 1))
            used_a.add((ra, ca))
            used_b.add((rb, cb))
        if min(len(used_a), len(used_b)) < self.cfg.auto_min_covered_cells:
            return False

        hull_a = cv2.convexHull(pts_a.astype(np.float32))
        hull_b = cv2.convexHull(pts_b.astype(np.float32))
        area_a = float(cv2.contourArea(hull_a))
        area_b = float(cv2.contourArea(hull_b))
        full = float(width * height)
        if full <= 0:
            return False
        return min(area_a / full, area_b / full) >= self.cfg.auto_min_hull_ratio

    def _estimate_from_points(
        self,
        pts_a: np.ndarray,
        pts_b: np.ndarray,
    ) -> tuple[GeometryModel | None, AutoGeometryStatus]:
        n_matches = int(pts_a.shape[0])
        if n_matches < self.cfg.auto_min_mutual_matches:
            return None, AutoGeometryStatus(
                state="initializing",
                reason="few_matches",
                matches=n_matches,
                inliers=0,
                inlier_ratio=0.0,
                median_error_px=0.0,
                p90_error_px=0.0,
                updated_at=time.monotonic(),
            )

        method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
        f, mask = cv2.findFundamentalMat(
            pts_a.astype(np.float32),
            pts_b.astype(np.float32),
            method=method,
            ransacReprojThreshold=float(self.cfg.auto_ransac_threshold_px),
            confidence=0.999,
            maxIters=10000,
        )
        if f is None or mask is None or np.asarray(f).shape != (3, 3):
            return None, AutoGeometryStatus(
                state="initializing",
                reason="f_estimation_failed",
                matches=n_matches,
                inliers=0,
                inlier_ratio=0.0,
                median_error_px=0.0,
                p90_error_px=0.0,
                updated_at=time.monotonic(),
            )

        mask_flat = mask.reshape(-1).astype(bool)
        inliers = int(np.count_nonzero(mask_flat))
        if inliers < self.cfg.auto_min_inliers:
            return None, AutoGeometryStatus(
                state="initializing",
                reason="few_inliers",
                matches=n_matches,
                inliers=inliers,
                inlier_ratio=float(inliers / max(n_matches, 1)),
                median_error_px=0.0,
                p90_error_px=0.0,
                updated_at=time.monotonic(),
            )
        inlier_ratio = float(inliers / max(n_matches, 1))
        if inlier_ratio < self.cfg.auto_min_inlier_ratio:
            return None, AutoGeometryStatus(
                state="initializing",
                reason="low_inlier_ratio",
                matches=n_matches,
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                median_error_px=0.0,
                p90_error_px=0.0,
                updated_at=time.monotonic(),
            )

        ua = pts_a[mask_flat]
        ub = pts_b[mask_flat]
        # Rank-2 enforcement.
        u, s, vt = np.linalg.svd(np.asarray(f, dtype=np.float64))
        s[-1] = 0.0
        f_rank2 = u @ np.diag(s) @ vt
        if np.linalg.norm(f_rank2) > 1e-9:
            f_rank2 = f_rank2 / np.linalg.norm(f_rank2)

        errs = _sampson_err(ua, ub, f_rank2)
        med = float(np.median(errs))
        p90 = float(np.percentile(errs, 90))
        if med > self.cfg.auto_median_error_px or p90 > self.cfg.auto_p90_error_px:
            return None, AutoGeometryStatus(
                state="initializing",
                reason="high_residual",
                matches=n_matches,
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                median_error_px=med,
                p90_error_px=p90,
                updated_at=time.monotonic(),
            )

        model = GeometryModel(
            f=f_rank2.astype(np.float64),
            source="auto",
            state="healthy",
            matches=n_matches,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            median_error_px=med,
            p90_error_px=p90,
            updated_at=time.monotonic(),
        )
        status = AutoGeometryStatus(
            state="healthy",
            reason="ok",
            matches=n_matches,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            median_error_px=med,
            p90_error_px=p90,
            updated_at=model.updated_at,
        )
        return model, status

    def observe_pair(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        bbox_a: np.ndarray | None,
        bbox_b: np.ndarray | None,
    ) -> None:
        now = time.monotonic()
        dt_ms = (now - self._last_estimate_ts) * 1000.0
        if dt_ms < self.cfg.auto_estimation_interval_ms:
            return
        self._last_estimate_ts = now

        if self.model is not None and self.model.age_ms > self.cfg.auto_model_expiry_ms:
            self.model = None
            self.status = AutoGeometryStatus(
                state="reinitializing",
                reason="model_expired",
                matches=0,
                inliers=0,
                inlier_ratio=0.0,
                median_error_px=0.0,
                p90_error_px=0.0,
                updated_at=now,
            )

        scale = float(self.cfg.auto_downscale)
        h, w = image_a.shape[:2]
        ds_w = max(32, int(round(w * scale)))
        ds_h = max(32, int(round(h * scale)))
        a_ds = cv2.resize(image_a, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        b_ds = cv2.resize(image_b, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

        mask_a = self._mask_from_bbox((h, w), bbox_a)
        mask_b = self._mask_from_bbox((h, w), bbox_b)
        if mask_a is not None:
            mask_a = cv2.resize(mask_a, (ds_w, ds_h), interpolation=cv2.INTER_NEAREST)
        if mask_b is not None:
            mask_b = cv2.resize(mask_b, (ds_w, ds_h), interpolation=cv2.INTER_NEAREST)

        kp_a, des_a = self._detect(a_ds, mask_a)
        kp_b, des_b = self._detect(b_ds, mask_b)
        use_hamming = self.cfg.auto_feature_type == "orb"
        mutual = self._mutual_knn_matches(des_a, des_b, use_hamming=use_hamming)
        if not mutual:
            self._failure_count += 1
            self.status = AutoGeometryStatus(
                state="suspect" if self.model is not None else "initializing",
                reason="no_mutual_matches",
                matches=0,
                inliers=0,
                inlier_ratio=0.0,
                median_error_px=0.0,
                p90_error_px=0.0,
                updated_at=now,
            )
            if self._failure_count >= self.cfg.auto_failure_reinit_count:
                self.model = None
                self.status = AutoGeometryStatus(
                    state="reinitializing",
                    reason="reinit_after_failures",
                    matches=0,
                    inliers=0,
                    inlier_ratio=0.0,
                    median_error_px=0.0,
                    p90_error_px=0.0,
                    updated_at=now,
                )
            return

        pts_a = np.array([kp_a[ia].pt for ia, _, _ in mutual], dtype=np.float64)
        pts_b = np.array([kp_b[ib].pt for _, ib, _ in mutual], dtype=np.float64)
        # Scale points back to runtime image space.
        sx = w / max(ds_w, 1)
        sy = h / max(ds_h, 1)
        pts_a[:, 0] *= sx
        pts_a[:, 1] *= sy
        pts_b[:, 0] *= sx
        pts_b[:, 1] *= sy

        if not self._coverage_ok(pts_a, pts_b, w, h):
            self._failure_count += 1
            self.status = AutoGeometryStatus(
                state="suspect" if self.model is not None else "initializing",
                reason="poor_spatial_coverage",
                matches=int(len(mutual)),
                inliers=0,
                inlier_ratio=0.0,
                median_error_px=0.0,
                p90_error_px=0.0,
                updated_at=now,
            )
            return

        model, status = self._estimate_from_points(pts_a, pts_b)
        if model is None:
            self._failure_count += 1
            self.status = AutoGeometryStatus(
                state="suspect" if self.model is not None else "initializing",
                reason=status.reason,
                matches=status.matches,
                inliers=status.inliers,
                inlier_ratio=status.inlier_ratio,
                median_error_px=status.median_error_px,
                p90_error_px=status.p90_error_px,
                updated_at=now,
            )
            if self._failure_count >= self.cfg.auto_failure_reinit_count:
                self.model = None
                self.status = AutoGeometryStatus(
                    state="reinitializing",
                    reason="reinit_after_failures",
                    matches=status.matches,
                    inliers=status.inliers,
                    inlier_ratio=status.inlier_ratio,
                    median_error_px=status.median_error_px,
                    p90_error_px=status.p90_error_px,
                    updated_at=now,
                )
            return

        self.model = model
        self.status = status
        self._failure_count = 0
        self._last_health_ts = now
