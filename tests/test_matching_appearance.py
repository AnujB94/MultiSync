from __future__ import annotations

import numpy as np

from dualpose.matching.appearance import AppearanceReID


def test_mask_head_suppresses_top_region() -> None:
    reid = AppearanceReID(
        enabled=False,
        model_path="models/missing.onnx",
        download_if_missing=False,
        interval_ms=500.0,
        min_box_area=100,
        threads=1,
        gallery_ttl_ms=3000.0,
        exclude_head=True,
    )
    crop = np.zeros((100, 40, 3), dtype=np.uint8)
    crop[:40, :, :] = (0, 0, 255)
    crop[40:, :, :] = (0, 255, 0)
    masked = reid._mask_head(crop, keypoints=None)
    assert not np.array_equal(masked[:20], crop[:20])


def test_gallery_reset_by_cam() -> None:
    reid = AppearanceReID(
        enabled=False,
        model_path="models/missing.onnx",
        download_if_missing=False,
        interval_ms=500.0,
        min_box_area=100,
        threads=1,
        gallery_ttl_ms=3000.0,
        exclude_head=True,
    )
    reid._gallery["a"] = []  # type: ignore[assignment]
    reid._last_run_ts["a"] = 1.0
    reid._last_seen_ts["a"] = 1.0
    reid.reset("a")
    assert "a" not in reid._gallery
    assert "a" not in reid._last_run_ts
