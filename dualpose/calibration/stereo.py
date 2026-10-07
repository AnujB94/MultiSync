from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _to_float_array(value: np.ndarray | list[list[float]]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class StereoCalibration:
    k1: np.ndarray
    d1: np.ndarray
    k2: np.ndarray
    d2: np.ndarray
    r: np.ndarray
    t: np.ndarray
    e: np.ndarray
    f: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    q: np.ndarray
    image_size_1: tuple[int, int]
    image_size_2: tuple[int, int]
    transform_meta_1: dict[str, Any]
    transform_meta_2: dict[str, Any]
    metrics: dict[str, float]
    time_offset_s: float

    def validate_runtime(
        self,
        image_size_1: tuple[int, int],
        image_size_2: tuple[int, int],
        transform_meta_1: dict[str, Any],
        transform_meta_2: dict[str, Any],
    ) -> tuple[bool, str]:
        if tuple(image_size_1) != tuple(self.image_size_1):
            return (
                False,
                f"camera-1 image size mismatch: runtime={image_size_1} calib={self.image_size_1}",
            )
        if tuple(image_size_2) != tuple(self.image_size_2):
            return (
                False,
                f"camera-2 image size mismatch: runtime={image_size_2} calib={self.image_size_2}",
            )
        if transform_meta_1 != self.transform_meta_1:
            return (False, "camera-1 transform metadata differs from calibration capture.")
        if transform_meta_2 != self.transform_meta_2:
            return (False, "camera-2 transform metadata differs from calibration capture.")
        return (True, "")

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            k1=self.k1,
            d1=self.d1,
            k2=self.k2,
            d2=self.d2,
            r=self.r,
            t=self.t,
            e=self.e,
            f=self.f,
            r1=self.r1,
            r2=self.r2,
            p1=self.p1,
            p2=self.p2,
            q=self.q,
            image_size_1=np.asarray(self.image_size_1, dtype=np.int32),
            image_size_2=np.asarray(self.image_size_2, dtype=np.int32),
            transform_meta_1=np.asarray([self.transform_meta_1], dtype=object),
            transform_meta_2=np.asarray([self.transform_meta_2], dtype=object),
            metrics=np.asarray([self.metrics], dtype=object),
            time_offset_s=np.asarray([self.time_offset_s], dtype=np.float64),
        )

    @staticmethod
    def load(path: str | Path) -> StereoCalibration:
        payload = np.load(Path(path), allow_pickle=True)
        transform_meta_1_raw = payload["transform_meta_1"][0]
        transform_meta_2_raw = payload["transform_meta_2"][0]
        metrics_raw = payload["metrics"][0]

        if hasattr(transform_meta_1_raw, "item"):
            transform_meta_1_raw = transform_meta_1_raw.item()
        if hasattr(transform_meta_2_raw, "item"):
            transform_meta_2_raw = transform_meta_2_raw.item()
        if hasattr(metrics_raw, "item"):
            metrics_raw = metrics_raw.item()

        transform_meta_1 = dict(transform_meta_1_raw)
        transform_meta_2 = dict(transform_meta_2_raw)
        metrics = dict(metrics_raw)
        return StereoCalibration(
            k1=_to_float_array(payload["k1"]),
            d1=_to_float_array(payload["d1"]),
            k2=_to_float_array(payload["k2"]),
            d2=_to_float_array(payload["d2"]),
            r=_to_float_array(payload["r"]),
            t=_to_float_array(payload["t"]),
            e=_to_float_array(payload["e"]),
            f=_to_float_array(payload["f"]),
            r1=_to_float_array(payload["r1"]),
            r2=_to_float_array(payload["r2"]),
            p1=_to_float_array(payload["p1"]),
            p2=_to_float_array(payload["p2"]),
            q=_to_float_array(payload["q"]),
            image_size_1=tuple(int(x) for x in payload["image_size_1"].tolist()),
            image_size_2=tuple(int(x) for x in payload["image_size_2"].tolist()),
            transform_meta_1=transform_meta_1,
            transform_meta_2=transform_meta_2,
            metrics={str(k): float(v) for k, v in metrics.items()},
            time_offset_s=float(payload["time_offset_s"][0]),
        )


class StereoCalibrator:
    def __init__(self, inner_corners: tuple[int, int], square_size_m: float) -> None:
        self.inner_corners = inner_corners
        self.square_size_m = square_size_m
        self.obj_points: list[np.ndarray] = []
        self.img_points_1: list[np.ndarray] = []
        self.img_points_2: list[np.ndarray] = []
        self.image_size_1: tuple[int, int] | None = None
        self.image_size_2: tuple[int, int] | None = None

        grid = np.zeros((inner_corners[0] * inner_corners[1], 3), np.float32)
        xy = np.mgrid[0 : inner_corners[0], 0 : inner_corners[1]].T.reshape(-1, 2)
        grid[:, :2] = xy
        self._obj_template = grid * square_size_m

    def _detect(self, image: np.ndarray) -> tuple[bool, np.ndarray | None]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, self.inner_corners)
        if found and corners is not None:
            return True, corners.astype(np.float32)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found2, corners2 = cv2.findChessboardCorners(gray, self.inner_corners, flags)
        if not found2 or corners2 is None:
            return False, None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        cv2.cornerSubPix(gray, corners2, (11, 11), (-1, -1), criteria)
        return True, corners2.astype(np.float32)

    def detect_pair(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        ok1, c1 = self._detect(image1)
        ok2, c2 = self._detect(image2)
        return (ok1 and ok2, c1, c2)

    def add_pair(self, image1: np.ndarray, image2: np.ndarray) -> tuple[bool, str]:
        ok, c1, c2 = self.detect_pair(image1, image2)
        if not ok or c1 is None or c2 is None:
            return False, "checkerboard not found in both frames"

        size_1 = (int(image1.shape[1]), int(image1.shape[0]))
        size_2 = (int(image2.shape[1]), int(image2.shape[0]))
        if self.image_size_1 is None:
            self.image_size_1 = size_1
        if self.image_size_2 is None:
            self.image_size_2 = size_2
        if size_1 != self.image_size_1 or size_2 != self.image_size_2:
            return False, "image size changed; keep fixed capture resolution during calibration"

        self.obj_points.append(self._obj_template.copy())
        self.img_points_1.append(c1)
        self.img_points_2.append(c2)
        return True, "pair captured"

    def _mono_calibrate(
        self,
        obj_points: list[np.ndarray],
        img_points: list[np.ndarray],
        image_size: tuple[int, int],
    ) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
        flags = cv2.CALIB_RATIONAL_MODEL
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
        rms, k, d, rvecs, tvecs, _, _, per_view = cv2.calibrateCameraExtended(
            obj_points,
            img_points,
            image_size,
            None,
            None,
            flags=flags,
            criteria=criteria,
        )
        return rms, k, d, rvecs, tvecs, per_view.reshape(-1)

    def calibrate(
        self,
        transform_meta_1: dict[str, Any],
        transform_meta_2: dict[str, Any],
        min_pairs: int = 12,
    ) -> StereoCalibration:
        if len(self.obj_points) < min_pairs:
            raise ValueError(f"Need at least {min_pairs} pairs, got {len(self.obj_points)}.")
        assert self.image_size_1 is not None and self.image_size_2 is not None

        rms1, k1, d1, _, _, per1 = self._mono_calibrate(
            self.obj_points,
            self.img_points_1,
            self.image_size_1,
        )
        rms2, k2, d2, _, _, per2 = self._mono_calibrate(
            self.obj_points,
            self.img_points_2,
            self.image_size_2,
        )

        # Drop highest-error pairs once if enough samples are available.
        if len(self.obj_points) >= 18:
            pv = 0.5 * (per1 + per2)
            threshold = float(np.median(pv) + 2.5 * np.median(np.abs(pv - np.median(pv))))
            keep = [i for i, e in enumerate(pv) if e <= threshold]
            if len(keep) >= min_pairs and len(keep) < len(self.obj_points):
                self.obj_points = [self.obj_points[i] for i in keep]
                self.img_points_1 = [self.img_points_1[i] for i in keep]
                self.img_points_2 = [self.img_points_2[i] for i in keep]
                rms1, k1, d1, _, _, per1 = self._mono_calibrate(
                    self.obj_points,
                    self.img_points_1,
                    self.image_size_1,
                )
                rms2, k2, d2, _, _, per2 = self._mono_calibrate(
                    self.obj_points,
                    self.img_points_2,
                    self.image_size_2,
                )

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
        flags = cv2.CALIB_FIX_INTRINSIC
        stereo_rms, _, _, _, _, r, t, e, f = cv2.stereoCalibrate(
            self.obj_points,
            self.img_points_1,
            self.img_points_2,
            k1,
            d1,
            k2,
            d2,
            self.image_size_1,
            criteria=criteria,
            flags=flags,
        )

        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            k1,
            d1,
            k2,
            d2,
            self.image_size_1,
            r,
            t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
        )

        metrics = {
            "mono_rms_1": float(rms1),
            "mono_rms_2": float(rms2),
            "stereo_rms": float(stereo_rms),
            "pairs_used": float(len(self.obj_points)),
        }
        return StereoCalibration(
            k1=_to_float_array(k1),
            d1=_to_float_array(d1),
            k2=_to_float_array(k2),
            d2=_to_float_array(d2),
            r=_to_float_array(r),
            t=_to_float_array(t),
            e=_to_float_array(e),
            f=_to_float_array(f),
            r1=_to_float_array(r1),
            r2=_to_float_array(r2),
            p1=_to_float_array(p1),
            p2=_to_float_array(p2),
            q=_to_float_array(q),
            image_size_1=self.image_size_1,
            image_size_2=self.image_size_2,
            transform_meta_1=transform_meta_1,
            transform_meta_2=transform_meta_2,
            metrics=metrics,
            time_offset_s=0.0,
        )

    @staticmethod
    def estimate_time_offset(
        timestamps_1: np.ndarray,
        signal_1: np.ndarray,
        timestamps_2: np.ndarray,
        signal_2: np.ndarray,
        max_offset_s: float = 2.0,
        step_s: float = 0.01,
    ) -> float:
        """
        Estimate timestamp offset by maximizing normalized cross-correlation.

        Positive result means stream-2 timestamps should be shifted forward.
        """
        t1 = np.asarray(timestamps_1, dtype=np.float64)
        s1 = np.asarray(signal_1, dtype=np.float64)
        t2 = np.asarray(timestamps_2, dtype=np.float64)
        s2 = np.asarray(signal_2, dtype=np.float64)
        if t1.size < 8 or t2.size < 8:
            return 0.0

        offsets = np.arange(-max_offset_s, max_offset_s + step_s, step_s)
        best_offset = 0.0
        best_score = -1e12
        for off in offsets:
            x2 = np.interp(t1, t2 + off, s2, left=np.nan, right=np.nan)
            mask = np.isfinite(x2)
            if np.count_nonzero(mask) < 8:
                continue
            a = s1[mask]
            b = x2[mask]
            a = a - np.mean(a)
            b = b - np.mean(b)
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
            score = float(np.dot(a, b) / denom)
            if score > best_score:
                best_score = score
                best_offset = float(off)
        return best_offset
