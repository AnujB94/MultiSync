from __future__ import annotations

from collections import deque

from dualpose.utils.types import PersonObservation


class TimestampPairer:
    def __init__(
        self,
        cam_a_id: str,
        cam_b_id: str,
        max_pose_age_ms: float,
        max_pair_skew_ms: float,
        time_offset_s: float,
    ) -> None:
        self.cam_a_id = cam_a_id
        self.cam_b_id = cam_b_id
        self.max_pose_age_s = max_pose_age_ms / 1000.0
        self.max_pair_skew_s = max_pair_skew_ms / 1000.0
        self.time_offset_s = time_offset_s
        self._buffers: dict[str, deque[PersonObservation]] = {
            cam_a_id: deque(maxlen=180),
            cam_b_id: deque(maxlen=180),
        }

    def _corrected_time(self, obs: PersonObservation) -> float:
        if obs.cam_id == self.cam_b_id:
            return obs.t_capture + self.time_offset_s
        return obs.t_capture

    def _is_fresh(self, obs: PersonObservation, now_ts: float) -> bool:
        return (now_ts - obs.t_capture) <= self.max_pose_age_s

    def submit(
        self,
        obs: PersonObservation,
        now_ts: float,
    ) -> tuple[PersonObservation, PersonObservation, float] | None:
        if obs.cam_id not in self._buffers:
            return None
        self._buffers[obs.cam_id].append(obs)

        for cam_id, buf in self._buffers.items():
            while buf and not self._is_fresh(buf[0], now_ts):
                buf.popleft()
            self._buffers[cam_id] = buf

        other_id = self.cam_b_id if obs.cam_id == self.cam_a_id else self.cam_a_id
        candidates = self._buffers.get(other_id)
        if not candidates:
            return None

        target_t = self._corrected_time(obs)
        best: PersonObservation | None = None
        best_abs = 1e12
        for candidate in candidates:
            diff = abs(target_t - self._corrected_time(candidate))
            if diff < best_abs:
                best = candidate
                best_abs = diff

        if best is None or best_abs > self.max_pair_skew_s:
            return None

        if obs.cam_id == self.cam_a_id:
            return obs, best, best_abs * 1000.0
        return best, obs, best_abs * 1000.0
