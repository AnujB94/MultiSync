from __future__ import annotations

import numpy as np

from dualpose.calibration.stereo import StereoCalibration
from dualpose.matching.geometry import EpipolarScorer
from dualpose.utils.types import PersonObservation


def _calib() -> StereoCalibration:
    eye3 = np.eye(3, dtype=np.float64)
    return StereoCalibration(
        k1=eye3,
        d1=np.zeros((5, 1), dtype=np.float64),
        k2=eye3,
        d2=np.zeros((5, 1), dtype=np.float64),
        r=eye3,
        t=np.array([[1.0], [0.0], [0.0]], dtype=np.float64),
        e=np.zeros((3, 3), dtype=np.float64),
        # Epipolar relation: y2 == y1 for this simplified stereo setup.
        f=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        r1=eye3,
        r2=eye3,
        p1=np.hstack([eye3, np.zeros((3, 1), dtype=np.float64)]),
        p2=np.hstack([eye3, np.array([[1.0], [0.0], [0.0]], dtype=np.float64)]),
        q=np.eye(4, dtype=np.float64),
        image_size_1=(640, 360),
        image_size_2=(640, 360),
        transform_meta_1={"rotate": 0, "flip_horizontal": False, "flip_vertical": False},
        transform_meta_2={"rotate": 0, "flip_horizontal": False, "flip_vertical": False},
        metrics={},
        time_offset_s=0.0,
    )


def _obs(cam_id: str, points: np.ndarray) -> PersonObservation:
    return PersonObservation(
        cam_id=cam_id,
        frame_id=1,
        t_capture=1.0,
        kpts_2d=points,
        bbox_xyxy=np.array([10, 10, 100, 200], dtype=np.float32),
        confidence=1.0,
        detection_fresh=True,
        image_shape=(360, 640, 3),
        appearance_embedding=None,
        pose_result=None,
    )


def test_epipolar_score_same_person_like_points() -> None:
    calib = _calib()
    scorer = EpipolarScorer(calib, min_common_joints=6, sigma_px=6.0, hard_reject_px=15.0)
    model = EpipolarScorer.calibrated_model(calib)
    assert model is not None
    a = np.zeros((17, 3), dtype=np.float32)
    b = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        y = 50.0 + i * 3.0
        a[i] = np.array([100.0 + i * 2.0, y, 1.0], dtype=np.float32)
        b[i] = np.array([90.0 + i * 2.0, y, 1.0], dtype=np.float32)
    score, error, n, hard_fail, reason = scorer.score(_obs("a", a), _obs("b", b), model=model)
    assert reason == "ok"
    assert not hard_fail
    assert n >= 6
    assert score is not None and score > 0.7
    assert error is not None and error < 1.0


def test_epipolar_score_rejects_large_vertical_inconsistency() -> None:
    calib = _calib()
    scorer = EpipolarScorer(calib, min_common_joints=6, sigma_px=6.0, hard_reject_px=15.0)
    model = EpipolarScorer.calibrated_model(calib)
    assert model is not None
    a = np.zeros((17, 3), dtype=np.float32)
    b = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        a[i] = np.array([120.0 + i, 70.0 + i, 1.0], dtype=np.float32)
        b[i] = np.array([90.0 + i, 170.0 + i, 1.0], dtype=np.float32)
    score, error, _, hard_fail, reason = scorer.score(_obs("a", a), _obs("b", b), model=model)
    assert score is not None
    assert error is not None
    assert hard_fail
    assert reason == "hard_epipolar_reject"
