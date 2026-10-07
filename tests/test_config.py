from __future__ import annotations

from pathlib import Path

import pytest

from dualpose.config import load_config


def test_load_config_happy_path(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        """
window:
  title: "X"
  target_fps: 30
webcam:
  cam_id: "a"
  source: 0
  width: 640
  height: 360
  rotate: 0
  flip_horizontal: false
  flip_vertical: false
phone:
  cam_id: "b"
  source: "rtsp://x"
  width: 640
  height: 360
  rotate: 0
  flip_horizontal: false
  flip_vertical: false
pipeline:
  detector_mode: "lightweight"
  detector_backend: "onnxruntime"
  detector_device: "cpu"
  min_joint_confidence: 0.2
  confidence_hold_frames: 4
  temporal_window: 243
  infer_every_n_frames: 1
  lifter_stride: 1
  lifter_checkpoint: "checkpoints/a.bin"
  use_lifter_checkpoint: false
  weak_perspective: true
  fx_scale: 1.1
  fy_scale: 1.1
recording:
  enabled: false
  output_dir: "recordings"
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.window.title == "X"
    assert cfg.webcam.source == 0
    assert cfg.phone.source.startswith("rtsp://")
    assert cfg.calibration.enabled is False
    assert cfg.matching.enabled is False
    assert cfg.matching.geometry_mode == "auto"
    assert cfg.matching.auto_feature_type in {"sift", "orb"}


def test_load_config_requires_phone_url(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cfg_bad.yaml"
    cfg_file.write_text(
        """
window:
  title: "X"
  target_fps: 30
webcam:
  cam_id: "a"
  source: 0
  width: 640
  height: 360
  rotate: 0
  flip_horizontal: false
  flip_vertical: false
phone:
  cam_id: "b"
  source: 1
  width: 640
  height: 360
  rotate: 0
  flip_horizontal: false
  flip_vertical: false
pipeline:
  detector_mode: "lightweight"
  detector_backend: "onnxruntime"
  detector_device: "cpu"
  min_joint_confidence: 0.2
  confidence_hold_frames: 4
  temporal_window: 243
  infer_every_n_frames: 1
  lifter_stride: 1
  lifter_checkpoint: "checkpoints/a.bin"
  use_lifter_checkpoint: false
  weak_perspective: true
  fx_scale: 1.1
  fy_scale: 1.1
recording:
  enabled: false
  output_dir: "recordings"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(cfg_file)
