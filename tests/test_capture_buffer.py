from __future__ import annotations

import numpy as np

from dualpose.capture.base import LatestFrameBuffer
from dualpose.utils.types import FramePacket


def test_latest_frame_buffer_overwrites_old_packets() -> None:
    b = LatestFrameBuffer()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    b.publish(FramePacket("cam", 1, 1.0, img))
    b.publish(FramePacket("cam", 2, 2.0, img))
    latest = b.get_latest()
    assert latest is not None
    assert latest.frame_id == 2
    assert b.dropped_frames == 1
