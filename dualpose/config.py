from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _require(data: dict[str, Any], key: str, section: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing key '{section}.{key}' in config.")
    return data[key]


def _as_int(value: Any, key: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Expected int for '{key}', got {type(value).__name__}.")
    return value


def _as_float(value: Any, key: str) -> float:
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise ValueError(f"Expected float for '{key}', got {type(value).__name__}.")
    return value


def _as_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Expected bool for '{key}', got {type(value).__name__}.")
    return value


@dataclass(frozen=True)
class CameraConfig:
    cam_id: str
    source: int | str
    width: int
    height: int
    rotate: int
    flip_horizontal: bool
    flip_vertical: bool
    open_timeout_ms: int = 2000
    read_timeout_ms: int = 2000
    reconnect_backoff_seconds: float = 1.0
    stale_timeout_seconds: float = 2.0


@dataclass(frozen=True)
class PipelineConfig:
    detector_mode: str
    detector_backend: str
    detector_device: str
    min_joint_confidence: float
    confidence_hold_frames: int
    temporal_window: int
    infer_every_n_frames: int
    lifter_stride: int
    lifter_checkpoint: str
    use_lifter_checkpoint: bool
    weak_perspective: bool
    fx_scale: float
    fy_scale: float


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool
    output_dir: str


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool
    path: str
    checkerboard_cols: int
    checkerboard_rows: int
    square_size_m: float
    time_offset_s: float
    require_transform_match: bool


@dataclass(frozen=True)
class MatchingConfig:
    enabled: bool
    geometry_mode: str
    max_pose_age_ms: float
    max_pair_skew_ms: float
    min_common_joints: int
    epipolar_sigma_px: float
    epipolar_hard_reject_px: float
    reid_enabled: bool
    reid_model_path: str
    reid_download_if_missing: bool
    reid_interval_ms: float
    reid_min_box_area: int
    reid_threads: int
    reid_gallery_ttl_ms: float
    reid_exclude_head: bool
    weight_geometry: float
    weight_appearance: float
    weight_motion: float
    same_threshold: float
    different_threshold: float
    confirm_window: int
    confirm_required: int
    release_required: int
    appearance_weight_reid: float
    appearance_weight_hist: float
    appearance_weight_proportions: float
    appearance_only_same_threshold: float
    appearance_only_different_threshold: float
    appearance_only_confirm_window: int
    appearance_only_confirm_required: int
    appearance_only_release_required: int
    auto_feature_type: str
    auto_downscale: float
    auto_max_features: int
    auto_lowe_ratio: float
    auto_min_mutual_matches: int
    auto_min_inliers: int
    auto_min_inlier_ratio: float
    auto_ransac_threshold_px: float
    auto_median_error_px: float
    auto_p90_error_px: float
    auto_grid_rows: int
    auto_grid_cols: int
    auto_min_covered_cells: int
    auto_min_hull_ratio: float
    auto_model_expiry_ms: float
    auto_estimation_interval_ms: float
    auto_health_check_interval_ms: float
    auto_failure_reinit_count: int
    auto_person_mask_pad_ratio: float


@dataclass(frozen=True)
class WindowConfig:
    title: str
    target_fps: int


@dataclass(frozen=True)
class AppConfig:
    window: WindowConfig
    webcam: CameraConfig
    phone: CameraConfig
    pipeline: PipelineConfig
    recording: RecordingConfig
    calibration: CalibrationConfig
    matching: MatchingConfig


def _load_camera(section_name: str, section: dict[str, Any], is_network: bool) -> CameraConfig:
    cam_id = str(_require(section, "cam_id", section_name))
    source = _require(section, "source", section_name)
    if not isinstance(source, (int, str)):
        raise ValueError(f"Expected int|str for '{section_name}.source'.")

    cfg = CameraConfig(
        cam_id=cam_id,
        source=source,
        width=_as_int(_require(section, "width", section_name), f"{section_name}.width"),
        height=_as_int(_require(section, "height", section_name), f"{section_name}.height"),
        rotate=_as_int(_require(section, "rotate", section_name), f"{section_name}.rotate"),
        flip_horizontal=_as_bool(
            _require(section, "flip_horizontal", section_name),
            f"{section_name}.flip_horizontal",
        ),
        flip_vertical=_as_bool(
            _require(section, "flip_vertical", section_name),
            f"{section_name}.flip_vertical",
        ),
        open_timeout_ms=_as_int(
            section.get("open_timeout_ms", 2000), f"{section_name}.open_timeout_ms"
        ),
        read_timeout_ms=_as_int(
            section.get("read_timeout_ms", 2000), f"{section_name}.read_timeout_ms"
        ),
        reconnect_backoff_seconds=_as_float(
            section.get("reconnect_backoff_seconds", 1.0),
            f"{section_name}.reconnect_backoff_seconds",
        ),
        stale_timeout_seconds=_as_float(
            section.get("stale_timeout_seconds", 2.0),
            f"{section_name}.stale_timeout_seconds",
        ),
    )

    if is_network and not isinstance(source, str):
        raise ValueError(f"Expected stream URL for '{section_name}.source'.")
    return cfg


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a dictionary-like YAML.")

    window_raw = _require(raw, "window", "root")
    webcam_raw = _require(raw, "webcam", "root")
    phone_raw = _require(raw, "phone", "root")
    pipeline_raw = _require(raw, "pipeline", "root")
    recording_raw = _require(raw, "recording", "root")
    calibration_raw = raw.get("calibration", {})
    matching_raw = raw.get("matching", {})

    if not all(isinstance(v, dict) for v in [window_raw, webcam_raw, phone_raw, pipeline_raw]):
        raise ValueError("window/webcam/phone/pipeline must be objects.")
    if not isinstance(calibration_raw, dict):
        raise ValueError("calibration must be an object.")
    if not isinstance(matching_raw, dict):
        raise ValueError("matching must be an object.")

    window = WindowConfig(
        title=str(_require(window_raw, "title", "window")),
        target_fps=_as_int(_require(window_raw, "target_fps", "window"), "window.target_fps"),
    )

    webcam = _load_camera("webcam", webcam_raw, is_network=False)
    phone = _load_camera("phone", phone_raw, is_network=True)
    if webcam.cam_id == phone.cam_id:
        raise ValueError("webcam.cam_id and phone.cam_id must be different.")

    pipeline = PipelineConfig(
        detector_mode=str(_require(pipeline_raw, "detector_mode", "pipeline")),
        detector_backend=str(_require(pipeline_raw, "detector_backend", "pipeline")),
        detector_device=str(_require(pipeline_raw, "detector_device", "pipeline")),
        min_joint_confidence=_as_float(
            _require(pipeline_raw, "min_joint_confidence", "pipeline"),
            "pipeline.min_joint_confidence",
        ),
        confidence_hold_frames=_as_int(
            _require(pipeline_raw, "confidence_hold_frames", "pipeline"),
            "pipeline.confidence_hold_frames",
        ),
        temporal_window=_as_int(
            _require(pipeline_raw, "temporal_window", "pipeline"), "pipeline.temporal_window"
        ),
        infer_every_n_frames=_as_int(
            _require(pipeline_raw, "infer_every_n_frames", "pipeline"),
            "pipeline.infer_every_n_frames",
        ),
        lifter_stride=_as_int(
            _require(pipeline_raw, "lifter_stride", "pipeline"),
            "pipeline.lifter_stride",
        ),
        lifter_checkpoint=str(_require(pipeline_raw, "lifter_checkpoint", "pipeline")),
        use_lifter_checkpoint=_as_bool(
            _require(pipeline_raw, "use_lifter_checkpoint", "pipeline"),
            "pipeline.use_lifter_checkpoint",
        ),
        weak_perspective=_as_bool(
            _require(pipeline_raw, "weak_perspective", "pipeline"), "pipeline.weak_perspective"
        ),
        fx_scale=_as_float(_require(pipeline_raw, "fx_scale", "pipeline"), "pipeline.fx_scale"),
        fy_scale=_as_float(_require(pipeline_raw, "fy_scale", "pipeline"), "pipeline.fy_scale"),
    )

    if not isinstance(recording_raw, dict):
        raise ValueError("recording must be an object.")
    recording = RecordingConfig(
        enabled=_as_bool(_require(recording_raw, "enabled", "recording"), "recording.enabled"),
        output_dir=str(_require(recording_raw, "output_dir", "recording")),
    )

    calibration = CalibrationConfig(
        enabled=_as_bool(calibration_raw.get("enabled", False), "calibration.enabled"),
        path=str(calibration_raw.get("path", "calibration/stereo_calibration.npz")),
        checkerboard_cols=_as_int(
            calibration_raw.get("checkerboard_cols", 9), "calibration.checkerboard_cols"
        ),
        checkerboard_rows=_as_int(
            calibration_raw.get("checkerboard_rows", 6), "calibration.checkerboard_rows"
        ),
        square_size_m=_as_float(
            calibration_raw.get("square_size_m", 0.025), "calibration.square_size_m"
        ),
        time_offset_s=_as_float(
            calibration_raw.get("time_offset_s", 0.0), "calibration.time_offset_s"
        ),
        require_transform_match=_as_bool(
            calibration_raw.get("require_transform_match", True),
            "calibration.require_transform_match",
        ),
    )

    matching = MatchingConfig(
        enabled=_as_bool(matching_raw.get("enabled", False), "matching.enabled"),
        geometry_mode=str(matching_raw.get("geometry_mode", "auto")).lower(),
        max_pose_age_ms=_as_float(
            matching_raw.get("max_pose_age_ms", 500.0), "matching.max_pose_age_ms"
        ),
        max_pair_skew_ms=_as_float(
            matching_raw.get("max_pair_skew_ms", 80.0), "matching.max_pair_skew_ms"
        ),
        min_common_joints=_as_int(
            matching_raw.get("min_common_joints", 6), "matching.min_common_joints"
        ),
        epipolar_sigma_px=_as_float(
            matching_raw.get("epipolar_sigma_px", 6.0), "matching.epipolar_sigma_px"
        ),
        epipolar_hard_reject_px=_as_float(
            matching_raw.get("epipolar_hard_reject_px", 15.0),
            "matching.epipolar_hard_reject_px",
        ),
        reid_enabled=_as_bool(matching_raw.get("reid_enabled", False), "matching.reid_enabled"),
        reid_model_path=str(
            matching_raw.get("reid_model_path", "models/osnet_x0_25_msmt17.onnx")
        ),
        reid_download_if_missing=_as_bool(
            matching_raw.get("reid_download_if_missing", False),
            "matching.reid_download_if_missing",
        ),
        reid_interval_ms=_as_float(
            matching_raw.get("reid_interval_ms", 500.0), "matching.reid_interval_ms"
        ),
        reid_min_box_area=_as_int(
            matching_raw.get("reid_min_box_area", 2500), "matching.reid_min_box_area"
        ),
        reid_threads=_as_int(matching_raw.get("reid_threads", 1), "matching.reid_threads"),
        reid_gallery_ttl_ms=_as_float(
            matching_raw.get("reid_gallery_ttl_ms", 4000.0), "matching.reid_gallery_ttl_ms"
        ),
        reid_exclude_head=_as_bool(
            matching_raw.get("reid_exclude_head", True), "matching.reid_exclude_head"
        ),
        weight_geometry=_as_float(
            matching_raw.get("weight_geometry", 0.55), "matching.weight_geometry"
        ),
        weight_appearance=_as_float(
            matching_raw.get("weight_appearance", 0.35),
            "matching.weight_appearance",
        ),
        weight_motion=_as_float(
            matching_raw.get("weight_motion", 0.10), "matching.weight_motion"
        ),
        same_threshold=_as_float(
            matching_raw.get("same_threshold", 0.75), "matching.same_threshold"
        ),
        different_threshold=_as_float(
            matching_raw.get("different_threshold", 0.45), "matching.different_threshold"
        ),
        confirm_window=_as_int(
            matching_raw.get("confirm_window", 7), "matching.confirm_window"
        ),
        confirm_required=_as_int(
            matching_raw.get("confirm_required", 5), "matching.confirm_required"
        ),
        release_required=_as_int(
            matching_raw.get("release_required", 8), "matching.release_required"
        ),
        appearance_weight_reid=_as_float(
            matching_raw.get("appearance_weight_reid", 0.60),
            "matching.appearance_weight_reid",
        ),
        appearance_weight_hist=_as_float(
            matching_raw.get("appearance_weight_hist", 0.30),
            "matching.appearance_weight_hist",
        ),
        appearance_weight_proportions=_as_float(
            matching_raw.get("appearance_weight_proportions", 0.10),
            "matching.appearance_weight_proportions",
        ),
        appearance_only_same_threshold=_as_float(
            matching_raw.get("appearance_only_same_threshold", 0.82),
            "matching.appearance_only_same_threshold",
        ),
        appearance_only_different_threshold=_as_float(
            matching_raw.get("appearance_only_different_threshold", 0.58),
            "matching.appearance_only_different_threshold",
        ),
        appearance_only_confirm_window=_as_int(
            matching_raw.get("appearance_only_confirm_window", 10),
            "matching.appearance_only_confirm_window",
        ),
        appearance_only_confirm_required=_as_int(
            matching_raw.get("appearance_only_confirm_required", 8),
            "matching.appearance_only_confirm_required",
        ),
        appearance_only_release_required=_as_int(
            matching_raw.get("appearance_only_release_required", 5),
            "matching.appearance_only_release_required",
        ),
        auto_feature_type=str(matching_raw.get("auto_feature_type", "sift")).lower(),
        auto_downscale=_as_float(
            matching_raw.get("auto_downscale", 0.5), "matching.auto_downscale"
        ),
        auto_max_features=_as_int(
            matching_raw.get("auto_max_features", 1500), "matching.auto_max_features"
        ),
        auto_lowe_ratio=_as_float(
            matching_raw.get("auto_lowe_ratio", 0.75), "matching.auto_lowe_ratio"
        ),
        auto_min_mutual_matches=_as_int(
            matching_raw.get("auto_min_mutual_matches", 80),
            "matching.auto_min_mutual_matches",
        ),
        auto_min_inliers=_as_int(
            matching_raw.get("auto_min_inliers", 40), "matching.auto_min_inliers"
        ),
        auto_min_inlier_ratio=_as_float(
            matching_raw.get("auto_min_inlier_ratio", 0.35),
            "matching.auto_min_inlier_ratio",
        ),
        auto_ransac_threshold_px=_as_float(
            matching_raw.get("auto_ransac_threshold_px", 1.5),
            "matching.auto_ransac_threshold_px",
        ),
        auto_median_error_px=_as_float(
            matching_raw.get("auto_median_error_px", 1.0), "matching.auto_median_error_px"
        ),
        auto_p90_error_px=_as_float(
            matching_raw.get("auto_p90_error_px", 2.5), "matching.auto_p90_error_px"
        ),
        auto_grid_rows=_as_int(matching_raw.get("auto_grid_rows", 6), "matching.auto_grid_rows"),
        auto_grid_cols=_as_int(matching_raw.get("auto_grid_cols", 8), "matching.auto_grid_cols"),
        auto_min_covered_cells=_as_int(
            matching_raw.get("auto_min_covered_cells", 12),
            "matching.auto_min_covered_cells",
        ),
        auto_min_hull_ratio=_as_float(
            matching_raw.get("auto_min_hull_ratio", 0.25), "matching.auto_min_hull_ratio"
        ),
        auto_model_expiry_ms=_as_float(
            matching_raw.get("auto_model_expiry_ms", 12000.0),
            "matching.auto_model_expiry_ms",
        ),
        auto_estimation_interval_ms=_as_float(
            matching_raw.get("auto_estimation_interval_ms", 900.0),
            "matching.auto_estimation_interval_ms",
        ),
        auto_health_check_interval_ms=_as_float(
            matching_raw.get("auto_health_check_interval_ms", 2500.0),
            "matching.auto_health_check_interval_ms",
        ),
        auto_failure_reinit_count=_as_int(
            matching_raw.get("auto_failure_reinit_count", 3),
            "matching.auto_failure_reinit_count",
        ),
        auto_person_mask_pad_ratio=_as_float(
            matching_raw.get("auto_person_mask_pad_ratio", 0.15),
            "matching.auto_person_mask_pad_ratio",
        ),
    )

    if matching.geometry_mode not in {"auto", "off", "calibrated"}:
        raise ValueError("matching.geometry_mode must be one of: auto, off, calibrated")
    if matching.auto_feature_type not in {"sift", "orb"}:
        raise ValueError("matching.auto_feature_type must be 'sift' or 'orb'")
    if not (0.05 <= matching.auto_downscale <= 1.0):
        raise ValueError("matching.auto_downscale must be in [0.05, 1.0].")
    if not (0.0 < matching.auto_lowe_ratio < 1.0):
        raise ValueError("matching.auto_lowe_ratio must be in (0,1).")

    wsum = matching.weight_geometry + matching.weight_appearance + matching.weight_motion
    if wsum <= 0:
        raise ValueError("matching weights must sum to a positive number.")
    app_wsum = (
        matching.appearance_weight_reid
        + matching.appearance_weight_hist
        + matching.appearance_weight_proportions
    )
    if app_wsum <= 0:
        raise ValueError("appearance sub-weights must sum to a positive number.")

    return AppConfig(
        window=window,
        webcam=webcam,
        phone=phone,
        pipeline=pipeline,
        recording=recording,
        calibration=calibration,
        matching=matching,
    )
