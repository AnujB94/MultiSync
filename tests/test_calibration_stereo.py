from __future__ import annotations

from pathlib import Path

import numpy as np

from dualpose.calibration.stereo import StereoCalibration, StereoCalibrator


def _dummy_calibration() -> StereoCalibration:
    eye3 = np.eye(3, dtype=np.float64)
    return StereoCalibration(
        k1=eye3,
        d1=np.zeros((5, 1), dtype=np.float64),
        k2=eye3,
        d2=np.zeros((5, 1), dtype=np.float64),
        r=eye3,
        t=np.array([[1.0], [0.0], [0.0]], dtype=np.float64),
        e=np.zeros((3, 3), dtype=np.float64),
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
        metrics={"stereo_rms": 0.5},
        time_offset_s=0.12,
    )


def test_calibration_save_load_roundtrip(tmp_path: Path) -> None:
    calib = _dummy_calibration()
    out = tmp_path / "calib.npz"
    calib.save(out)
    loaded = StereoCalibration.load(out)
    assert loaded.image_size_1 == (640, 360)
    assert loaded.transform_meta_1["rotate"] == 0
    assert abs(loaded.time_offset_s - 0.12) < 1e-6
    assert np.allclose(loaded.f, calib.f)


def test_calibration_validate_runtime_mismatch() -> None:
    calib = _dummy_calibration()
    ok, reason = calib.validate_runtime(
        image_size_1=(640, 360),
        image_size_2=(800, 600),
        transform_meta_1={"rotate": 0, "flip_horizontal": False, "flip_vertical": False},
        transform_meta_2={"rotate": 0, "flip_horizontal": False, "flip_vertical": False},
    )
    assert not ok
    assert "image size mismatch" in reason


def test_estimate_time_offset() -> None:
    t1 = np.linspace(0.0, 5.0, 200)
    # Build a non-periodic motion signal to avoid phase ambiguity.
    sig1 = np.sin(1.7 * t1) + 0.15 * t1 + 0.3 * np.sin(0.23 * (t1**2))
    true_offset = 0.24
    # Camera-2 reports delayed timestamps while carrying the same motion values.
    t2 = t1 + true_offset
    sig2 = sig1.copy()
    est = StereoCalibrator.estimate_time_offset(
        t1,
        sig1,
        t2,
        sig2,
        max_offset_s=1.0,
        step_s=0.01,
    )
    assert abs(est + true_offset) < 0.03
