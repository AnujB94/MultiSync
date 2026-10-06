from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


class RateMeter:
    def __init__(self, window_seconds: float = 2.0) -> None:
        self.window_seconds = window_seconds
        self._events: deque[float] = deque()

    def tick(self, now: float | None = None) -> None:
        ts = now if now is not None else time.monotonic()
        self._events.append(ts)
        cutoff = ts - self.window_seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    @property
    def fps(self) -> float:
        if len(self._events) < 2:
            return 0.0
        elapsed = self._events[-1] - self._events[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._events) - 1) / elapsed


@dataclass
class CameraStats:
    capture_rate: RateMeter = field(default_factory=RateMeter)
    infer_rate: RateMeter = field(default_factory=RateMeter)
    dropped_frames: int = 0
    skipped_frames: int = 0
    last_inference_ms: float = 0.0
    last_pose_age_ms: float = 0.0
    last_capture_age_ms: float = 0.0
    last_fit_error_px: float = 0.0
    connected: bool = False


def now_ms() -> float:
    return time.monotonic() * 1000.0
