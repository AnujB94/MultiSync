from __future__ import annotations

import numpy as np

from dualpose.pose.keypoints import coco17_to_h36m17


def test_coco_to_h36m_mapping_shape_and_pelvis() -> None:
    k = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        k[i, 0] = float(i * 10)
        k[i, 1] = float(i * 5)
        k[i, 2] = 1.0
    out = coco17_to_h36m17(k)
    assert out.shape == (17, 3)
    expected_pelvis_x = (k[11, 0] + k[12, 0]) * 0.5
    assert abs(out[0, 0] - expected_pelvis_x) < 1e-5
    assert np.allclose(out[1], k[12])
    assert np.allclose(out[4], k[11])
