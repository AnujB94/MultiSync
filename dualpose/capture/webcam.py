from __future__ import annotations

from dualpose.capture.base import CaptureThread
from dualpose.capture.transforms import build_transform
from dualpose.config import CameraConfig


class WebcamCapture(CaptureThread):
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
