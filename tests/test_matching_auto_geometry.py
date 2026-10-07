from __future__ import annotations

import numpy as np

from dualpose.config import MatchingConfig
from dualpose.matching.auto_geometry import AutoGeometryEstimator


def _cfg(feature_type: str = "orb") -> MatchingConfig:
    return MatchingConfig(
        enabled=True,
        geometry_mode="auto",
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
        confirm_required=5,
        release_required=8,
        appearance_weight_reid=0.60,
        appearance_weight_hist=0.30,
        appearance_weight_proportions=0.10,
        appearance_only_same_threshold=0.82,
        appearance_only_different_threshold=0.58,
        appearance_only_confirm_window=10,
        appearance_only_confirm_required=8,
        appearance_only_release_required=5,
        auto_feature_type=feature_type,
        auto_downscale=0.5,
        auto_max_features=1200,
        auto_lowe_ratio=0.8 if feature_type == "orb" else 0.75,
        auto_min_mutual_matches=20,
        auto_min_inliers=12,
        auto_min_inlier_ratio=0.25,
        auto_ransac_threshold_px=2.0,
        auto_median_error_px=2.5,
        auto_p90_error_px=5.0,
        auto_grid_rows=6,
        auto_grid_cols=8,
        auto_min_covered_cells=8,
        auto_min_hull_ratio=0.10,
        auto_model_expiry_ms=6000.0,
        auto_estimation_interval_ms=10.0,
        auto_health_check_interval_ms=100.0,
        auto_failure_reinit_count=2,
        auto_person_mask_pad_ratio=0.15,
    )


def test_estimator_accepts_consistent_correspondences() -> None:
    est = AutoGeometryEstimator(_cfg())
    rng = np.random.default_rng(4)
    pts_a = rng.uniform([40, 30], [600, 330], size=(140, 2))
    # Horizontal shift plus mild perspective-like term.
    pts_b = pts_a.copy()
    pts_b[:, 0] = pts_b[:, 0] - 35.0 + 0.02 * (pts_a[:, 1] - 180.0)
    pts_b[:, 1] = pts_b[:, 1] + 0.01 * (pts_a[:, 0] - 320.0)
    pts_b += rng.normal(0.0, 0.35, size=pts_b.shape)

    model, status = est._estimate_from_points(pts_a, pts_b)
    assert model is not None
    assert status.state == "healthy"
    assert model.inliers >= 12
    assert model.inlier_ratio > 0.25


def test_estimator_rejects_few_matches() -> None:
    est = AutoGeometryEstimator(_cfg())
    pts_a = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float64)
    pts_b = pts_a + np.array([5.0, 0.0], dtype=np.float64)
    model, status = est._estimate_from_points(pts_a, pts_b)
    assert model is None
    assert status.reason == "few_matches"
