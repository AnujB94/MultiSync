from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from dualpose.utils.types import FramePacket


class FrameTransform(Protocol):
    def __call__(self, image: np.ndarray) -> np.ndarray: ...


@dataclass
class CaptureState:
    connected: bool = False
    last_error: str = ""
    last_frame_ts: float = 0.0


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: FramePacket | None = None
        self._dropped = 0

    def publish(self, packet: FramePacket) -> None:
        with self._lock:
            if self._packet is not None:
                self._dropped += 1
            self._packet = packet

    def get_latest(self) -> FramePacket | None:
        with self._lock:
            return self._packet

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped


class CaptureThread(threading.Thread):
    def __init__(
        self,
        cam_id: str,
        source: int | str,
        width: int,
        height: int,
        transform: FrameTransform | None = None,
        reconnect_backoff_seconds: float = 1.0,
        stale_timeout_seconds: float = 2.0,
    ) -> None:
        super().__init__(daemon=True, name=f"capture-{cam_id}")
        self.cam_id = cam_id
        self.source = source
        self.width = width
        self.height = height
        self.transform = transform
        self.reconnect_backoff_seconds = reconnect_backoff_seconds
        self.stale_timeout_seconds = stale_timeout_seconds

        self.buffer = LatestFrameBuffer()
        self.state = CaptureState()
        self._stop_event = threading.Event()
        self._frame_id = 0
        self._cap: cv2.VideoCapture | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        return cap

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def _reconnect(self) -> None:
        self._release()
        self.state.connected = False
        time.sleep(self.reconnect_backoff_seconds)
        self._cap = self._open_capture()
        self.state.connected = self._cap.isOpened()
        if not self.state.connected:
            self.state.last_error = "capture_open_failed"

    def run(self) -> None:
        self._cap = self._open_capture()
        self.state.connected = self._cap.isOpened()
        if not self.state.connected:
            self.state.last_error = "capture_open_failed"

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._reconnect()
                continue

            ok, frame = self._cap.read()
            now = time.monotonic()
            if not ok or frame is None:
                if (
                    self.state.last_frame_ts
                    and (now - self.state.last_frame_ts) > self.stale_timeout_seconds
                ):
                    self.state.last_error = "capture_stale"
                    self._reconnect()
                time.sleep(0.01)
                continue

            if self.transform is not None:
                frame = self.transform(frame)

            self._frame_id += 1
            packet = FramePacket(
                cam_id=self.cam_id,
                frame_id=self._frame_id,
                t_capture=now,
                image=frame,
            )
            self.buffer.publish(packet)
            self.state.connected = True
            self.state.last_error = ""
            self.state.last_frame_ts = now

        self._release()
