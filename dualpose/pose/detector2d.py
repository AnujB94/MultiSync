from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _HoldState:
    last_kpts: np.ndarray | None = None
    hold_count: int = 0


@dataclass(frozen=True)
class Detection2D:
    keypoints: np.ndarray
    bbox_xyxy: np.ndarray | None
    confidence: float
    is_fresh: bool


class Detector2D:
    def __init__(
        self,
        mode: str,
        backend: str,
        device: str,
        min_joint_confidence: float,
        confidence_hold_frames: int,
    ) -> None:
        self.min_joint_confidence = min_joint_confidence
        self.confidence_hold_frames = confidence_hold_frames
        self._hold: dict[str, _HoldState] = {}

        self._body: Any
        try:
            from rtmlib import Body
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("rtmlib is required for 2D pose detection.") from exc
        self._body = Body(mode=mode, backend=backend, device=device)

    def _bbox_from_keypoints(self, keypoints: np.ndarray) -> np.ndarray | None:
        conf = keypoints[:, 2]
        mask = conf >= self.min_joint_confidence
        if np.count_nonzero(mask) < 4:
            return None
        xy = keypoints[mask, :2]
        min_xy = np.min(xy, axis=0)
        max_xy = np.max(xy, axis=0)
        return np.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=np.float32)

    def _largest_person(self, keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
        if keypoints is None or scores is None:
            return None
        if len(keypoints) == 0:
            return None

        candidates: list[tuple[float, np.ndarray]] = []
        for i in range(len(keypoints)):
            xy = keypoints[i]
            conf = scores[i]
            if xy.shape[0] < 17:
                continue
            conf_mask = conf[:17] > self.min_joint_confidence
            if np.count_nonzero(conf_mask) < 6:
                continue
            pts = xy[:17, :2]
            pts_valid = pts[conf_mask]
            if pts_valid.size == 0:
                continue
            min_xy = pts_valid.min(axis=0)
            max_xy = pts_valid.max(axis=0)
            area = float(np.prod(max_xy - min_xy))
            person = np.concatenate([pts, conf[:17, None]], axis=1).astype(np.float32)
            candidates.append((area, person))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def detect_with_meta(self, image: np.ndarray, cam_id: str) -> Detection2D | None:
        keypoints, scores = self._body(image)
        person = self._largest_person(np.asarray(keypoints), np.asarray(scores))

        state = self._hold.setdefault(cam_id, _HoldState())
        if person is None:
            if state.last_kpts is not None and state.hold_count < self.confidence_hold_frames:
                state.hold_count += 1
                held = state.last_kpts.copy()
                bbox = self._bbox_from_keypoints(held)
                conf = float(np.mean(held[:, 2]))
                return Detection2D(
                    keypoints=held,
                    bbox_xyxy=bbox,
                    confidence=conf,
                    is_fresh=False,
                )
            return None

        if state.last_kpts is not None:
            # Fill low-confidence joints from previous frame for short gaps.
            low = person[:, 2] < self.min_joint_confidence
            person[low, :2] = state.last_kpts[low, :2]
            person[low, 2] = state.last_kpts[low, 2] * 0.8

        state.last_kpts = person
        state.hold_count = 0
        bbox = self._bbox_from_keypoints(person)
        conf = float(np.mean(person[:, 2]))
        return Detection2D(
            keypoints=person,
            bbox_xyxy=bbox,
            confidence=conf,
            is_fresh=True,
        )

    def detect(self, image: np.ndarray, cam_id: str) -> np.ndarray | None:
        out = self.detect_with_meta(image=image, cam_id=cam_id)
        return None if out is None else out.keypoints

    def reset(self, cam_id: str) -> None:
        self._hold.pop(cam_id, None)
