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

    if not all(isinstance(v, dict) for v in [window_raw, webcam_raw, phone_raw, pipeline_raw]):
        raise ValueError("window/webcam/phone/pipeline must be objects.")

    window = WindowConfig(
        title=str(_require(window_raw, "title", "window")),
        target_fps=_as_int(_require(window_raw, "target_fps", "window"), "window.target_fps"),
    )

    webcam = _load_camera("webcam", webcam_raw, is_network=False)
    phone = _load_camera("phone", phone_raw, is_network=True)

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

    return AppConfig(
        window=window,
        webcam=webcam,
        phone=phone,
        pipeline=pipeline,
        recording=recording,
    )
