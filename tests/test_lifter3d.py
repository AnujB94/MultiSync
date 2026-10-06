from __future__ import annotations

import numpy as np

from dualpose.pose.lifter3d import PoseLifter3D


def test_lifter_fallback_output_shape() -> None:
    lifter = PoseLifter3D(
        temporal_window=27,
        lifter_checkpoint="checkpoints/does_not_exist.bin",
        use_lifter_checkpoint=True,
    )
    k = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        k[i, 0] = i * 2.0
        k[i, 1] = i * 3.0
        k[i, 2] = 1.0
    out = lifter.infer("cam", k, (360, 640, 3))
    assert out.shape == (17, 3)
    assert np.allclose(out[0], np.zeros(3), atol=1e-6)
