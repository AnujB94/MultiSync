from __future__ import annotations

import numpy as np

from dualpose.matching.synchronizer import TimestampPairer
from dualpose.utils.types import PersonObservation


def _obs(cam_id: str, frame_id: int, t_capture: float) -> PersonObservation:
    return PersonObservation(
        cam_id=cam_id,
        frame_id=frame_id,
        t_capture=t_capture,
        kpts_2d=np.zeros((17, 3), dtype=np.float32),
        bbox_xyxy=np.array([10, 10, 20, 20], dtype=np.float32),
        confidence=1.0,
        detection_fresh=True,
        image_shape=(360, 640, 3),
        appearance_embedding=None,
        pose_result=None,
    )


def test_pairer_applies_time_offset() -> None:
    pairer = TimestampPairer(
        cam_a_id="a",
        cam_b_id="b",
        max_pose_age_ms=1000.0,
        max_pair_skew_ms=60.0,
        time_offset_s=-0.10,
    )
    # B arrives 100 ms after A, offset should realign.
    o_b = _obs("b", 1, 1.10)
    o_a = _obs("a", 2, 1.00)
    assert pairer.submit(o_b, now_ts=1.10) is None
    pair = pairer.submit(o_a, now_ts=1.10)
    assert pair is not None
    a, b, skew_ms = pair
    assert a.cam_id == "a"
    assert b.cam_id == "b"
    assert skew_ms < 15.0
