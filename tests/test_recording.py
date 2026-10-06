from __future__ import annotations

from pathlib import Path

import numpy as np

from dualpose.recording import SessionRecorder
from dualpose.utils.types import FramePacket


def test_recorder_writes_video_and_timestamps(tmp_path: Path) -> None:
    rec = SessionRecorder(str(tmp_path))
    rec.start()
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    rec.write(FramePacket(cam_id="camA", frame_id=1, t_capture=1.0, image=img))
    rec.stop()

    sessions = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert sessions, "No session directory created."
    files = {p.name for p in sessions[0].iterdir()}
    assert "camA.mp4" in files
    assert "camA_timestamps.csv" in files
