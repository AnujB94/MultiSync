from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dualpose.pose.lifter3d import PoseLifter3D


@pytest.mark.slow
def test_checkpoint_smoke_if_present() -> None:
    ckpt = Path("checkpoints/cpn-ft-243-dbb-causal.bin")
    if not ckpt.exists():
        pytest.skip("Checkpoint not present; skipping smoke test.")
    lifter = PoseLifter3D(
        temporal_window=243,
        lifter_checkpoint=str(ckpt),
        use_lifter_checkpoint=True,
    )
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 2] = 1.0
    out = lifter.infer("cam", k, (360, 640, 3))
    assert out.shape == (17, 3)
