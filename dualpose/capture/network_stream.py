from __future__ import annotations

import cv2

from dualpose.capture.base import CaptureThread
from dualpose.capture.transforms import build_transform
from dualpose.config import CameraConfig


class NetworkStreamCapture(CaptureThread):
    def __init__(self, config: CameraConfig) -> None:
        transform = build_transform(
            rotate=config.rotate,
            flip_horizontal=config.flip_horizontal,
            flip_vertical=config.flip_vertical,
        )
        super().__init__(
            cam_id=config.cam_id,
            source=config.source,
            width=config.width,
            height=config.height,
            transform=transform,
            reconnect_backoff_seconds=config.reconnect_backoff_seconds,
            stale_timeout_seconds=config.stale_timeout_seconds,
        )
        self.open_timeout_ms = config.open_timeout_ms
        self.read_timeout_ms = config.read_timeout_ms

    def _open_capture(self) -> cv2.VideoCapture:
        source = str(self.source)
        cap: cv2.VideoCapture
        try:
            params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                float(self.open_timeout_ms),
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                float(self.read_timeout_ms),
            ]
            # Newer OpenCV builds allow passing params at open-time.
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
        except (cv2.error, TypeError):
            # Older builds only accept (source, apiPreference).
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            # Final fallback: let OpenCV choose backend.
            cap = cv2.VideoCapture(source)

        # On some builds timeout setters are ignored, but harmless.
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(self.open_timeout_ms))
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(self.read_timeout_ms))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        return cap
