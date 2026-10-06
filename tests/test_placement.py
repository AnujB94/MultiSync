from __future__ import annotations

import numpy as np

from dualpose.pose.placement import PlacementFitter


def test_placement_weak_perspective_reprojects() -> None:
    fitter = PlacementFitter(weak_perspective=True, fx_scale=1.1, fy_scale=1.1)

    rel = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        rel[i, 0] = float(i)
        rel[i, 1] = float(i * 0.5)
        rel[i, 2] = float(i * 0.2)

    # target = rel_xy * 20 + [100, 50]
    k2d = np.zeros((17, 3), dtype=np.float32)
    k2d[:, :2] = rel[:, :2] * 20.0 + np.array([100.0, 50.0], dtype=np.float32)
    k2d[:, 2] = 1.0

    res = fitter.fit("cam", rel, k2d, (360, 640, 3))
    assert res.kpts_3d_cam.shape == (17, 3)
    assert res.reprojection_error_px < 1e-2
