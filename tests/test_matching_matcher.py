from __future__ import annotations

import time

import numpy as np

from dualpose.calibration.stereo import StereoCalibration
from dualpose.config import MatchingConfig
from dualpose.matching.matcher import IdentityMatcher
from dualpose.utils.types import IdentityState, PersonObservation


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


def _cfg() -> MatchingConfig:
    return MatchingConfig(
        enabled=True,
        geometry_mode="calibrated",
        max_pose_age_ms=800.0,
        max_pair_skew_ms=100.0,
        min_common_joints=6,
        epipolar_sigma_px=6.0,
        epipolar_hard_reject_px=15.0,
        reid_enabled=False,
        reid_model_path="models/osnet_x0_25_msmt17.onnx",
        reid_download_if_missing=False,
        reid_interval_ms=500.0,
        reid_min_box_area=2500,
        reid_threads=1,
        reid_gallery_ttl_ms=4000.0,
        reid_exclude_head=True,
        weight_geometry=0.55,
        weight_appearance=0.35,
        weight_motion=0.0,
        same_threshold=0.75,
        different_threshold=0.45,
        confirm_window=7,
        confirm_required=2,
        release_required=2,
        appearance_weight_reid=0.60,
        appearance_weight_hist=0.30,
        appearance_weight_proportions=0.10,
        appearance_only_same_threshold=0.82,
        appearance_only_different_threshold=0.58,
        appearance_only_confirm_window=10,
        appearance_only_confirm_required=8,
        appearance_only_release_required=5,
        auto_feature_type="sift",
        auto_downscale=0.5,
        auto_max_features=1500,
        auto_lowe_ratio=0.75,
        auto_min_mutual_matches=80,
        auto_min_inliers=40,
        auto_min_inlier_ratio=0.35,
        auto_ransac_threshold_px=1.5,
        auto_median_error_px=1.0,
        auto_p90_error_px=2.5,
        auto_grid_rows=6,
        auto_grid_cols=8,
        auto_min_covered_cells=12,
        auto_min_hull_ratio=0.25,
        auto_model_expiry_ms=12000.0,
        auto_estimation_interval_ms=900.0,
        auto_health_check_interval_ms=2500.0,
        auto_failure_reinit_count=3,
        auto_person_mask_pad_ratio=0.15,
    )


def _obs(
    cam_id: str,
    frame_id: int,
    t_capture: float,
    y_offset: float,
    fresh: bool = True,
    emb: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    prop: np.ndarray | None = None,
) -> PersonObservation:
    k = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        y = 60.0 + i * 2.0 + y_offset
        k[i] = np.array([120.0 + i * 2.0, y, 1.0], dtype=np.float32)
    return PersonObservation(
        cam_id=cam_id,
        frame_id=frame_id,
        t_capture=t_capture,
        kpts_2d=k,
        bbox_xyxy=np.array([100, 50, 200, 250], dtype=np.float32),
        confidence=1.0,
        detection_fresh=fresh,
        image_shape=(360, 640, 3),
        appearance_embedding=emb,
        upper_hist=upper,
        lower_hist=lower,
        body_proportions=prop,
        pose_result=None,
    )


def test_matcher_confirms_same_after_repeated_valid_pairs() -> None:
    matcher = IdentityMatcher("a", "b", _cfg(), _calib())
    base = time.monotonic()
    for i in range(1, 5):
        ts = base + i * 0.05
        matcher.submit(_obs("a", i, ts, y_offset=0.0))
        out = matcher.submit(_obs("b", i, ts, y_offset=0.0))
    assert out.state == IdentityState.SAME
    assert out.shared_id == "Person 1"
    assert out.evidence is not None and out.evidence.valid


def test_matcher_releases_to_unknown_on_stale_detection() -> None:
    matcher = IdentityMatcher("a", "b", _cfg(), _calib())
    base = time.monotonic()
    for i in range(1, 4):
        ts = base + i * 0.05
        matcher.submit(_obs("a", i, ts, y_offset=0.0))
        matcher.submit(_obs("b", i, ts, y_offset=0.0))
    assert matcher.get_state().state == IdentityState.SAME

    ts_stale = base + 4.0
    matcher.submit(_obs("a", 50, ts_stale, y_offset=0.0, fresh=False))
    matcher.submit(_obs("b", 50, ts_stale, y_offset=0.0, fresh=False))
    matcher.submit(_obs("a", 51, ts_stale + 0.05, y_offset=0.0, fresh=False))
    out = matcher.submit(_obs("b", 51, ts_stale + 0.05, y_offset=0.0, fresh=False))
    assert out.state == IdentityState.UNKNOWN


def test_matcher_can_run_without_calibration_appearance_only() -> None:
    cfg = _cfg()
    cfg = MatchingConfig(
        **{
            **cfg.__dict__,
            "geometry_mode": "off",
            "confirm_required": 2,
            "release_required": 2,
            "appearance_only_confirm_required": 2,
            "appearance_only_release_required": 2,
            "appearance_only_confirm_window": 4,
        }
    )
    matcher = IdentityMatcher("a", "b", cfg, calibration=None)
    assert matcher.enabled
    base = time.monotonic()

    emb = np.ones((256,), dtype=np.float32)
    emb = emb / np.linalg.norm(emb)
    hist = np.ones((32,), dtype=np.float32) / 32.0
    prop = np.ones((6,), dtype=np.float32)
    prop = prop / np.linalg.norm(prop)

    for i in range(1, 6):
        ts = base + i * 0.05
        matcher.submit(
            _obs("a", i, ts, y_offset=0.0, emb=emb, upper=hist, lower=hist, prop=prop)
        )
        out = matcher.submit(
            _obs("b", i, ts, y_offset=0.0, emb=emb, upper=hist, lower=hist, prop=prop)
        )

    assert out.evidence is not None
    assert out.evidence.mode == "appearance_only"
    assert out.state == IdentityState.SAME
