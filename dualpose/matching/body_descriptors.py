from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BodyDescriptor:
    upper_hist: np.ndarray | None
    lower_hist: np.ndarray | None
    proportions: np.ndarray | None


def _safe_hist(image_bgr: np.ndarray) -> np.ndarray | None:
    if image_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    total = float(np.sum(hist))
    if total <= 1e-9:
        return None
    return hist / total


def _clip_box(bbox_xyxy: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _proportions(kpts_2d: np.ndarray) -> np.ndarray | None:
    conf = kpts_2d[:, 2]
    if np.count_nonzero(conf > 0.05) < 8:
        return None

    pts = kpts_2d[:, :2]
    shoulder = np.linalg.norm(pts[11] - pts[14]) + 1e-6
    hip = np.linalg.norm(pts[4] - pts[1]) + 1e-6
    torso = np.linalg.norm(pts[8] - pts[0]) + 1e-6
    l_thigh = np.linalg.norm(pts[5] - pts[4]) + 1e-6
    l_shin = np.linalg.norm(pts[6] - pts[5]) + 1e-6
    r_thigh = np.linalg.norm(pts[2] - pts[1]) + 1e-6
    r_shin = np.linalg.norm(pts[3] - pts[2]) + 1e-6

    v = np.array(
        [
            shoulder / torso,
            hip / torso,
            l_thigh / torso,
            l_shin / torso,
            r_thigh / torso,
            r_shin / torso,
        ],
        dtype=np.float32,
    )
    return v / (np.linalg.norm(v) + 1e-9)


def extract_body_descriptor(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray | None,
    kpts_2d: np.ndarray,
) -> BodyDescriptor:
    if bbox_xyxy is None:
        return BodyDescriptor(upper_hist=None, lower_hist=None, proportions=_proportions(kpts_2d))
    h, w = image_bgr.shape[:2]
    clipped = _clip_box(bbox_xyxy, w, h)
    if clipped is None:
        return BodyDescriptor(upper_hist=None, lower_hist=None, proportions=_proportions(kpts_2d))
    x1, y1, x2, y2 = clipped
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return BodyDescriptor(upper_hist=None, lower_hist=None, proportions=_proportions(kpts_2d))

    split = int(round(roi.shape[0] * 0.45))
    split = max(1, min(split, roi.shape[0] - 1))
    upper = roi[:split]
    lower = roi[split:]
    return BodyDescriptor(
        upper_hist=_safe_hist(upper),
        lower_hist=_safe_hist(lower),
        proportions=_proportions(kpts_2d),
    )


def hist_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    # Bhattacharyya distance in [0,1], convert to similarity.
    d = float(
        cv2.compareHist(
            a.astype(np.float32),
            b.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )
    return float(np.clip(1.0 - d, 0.0, 1.0))


def proportion_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    c = float(np.dot(a, b) / denom)
    return float(np.clip((c + 1.0) * 0.5, 0.0, 1.0))
