from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - heavy dependency
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - runtime dependency
    torch = None
    nn = None


@dataclass
class LifterInfo:
    using_checkpoint: bool
    reason: str


class _TemporalBlock(nn.Module):  # pragma: no cover
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.drop(self.relu(self.conv1(x)))
        y = self.drop(self.relu(self.conv2(y)))
        return y + residual


class _TemporalModel(nn.Module):  # pragma: no cover
    """
    Lightweight causal temporal model inspired by VideoPose3D.

    Notes:
    - Input shape: (N, T, J, 2)
    - Output shape: (N, T, J, 3)
    """

    def __init__(self, num_joints: int = 17, channels: int = 1024, blocks: int = 5) -> None:
        super().__init__()
        in_features = num_joints * 2
        out_features = num_joints * 3

        self.inp = nn.Conv1d(in_features, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_TemporalBlock(channels, dilation=2**i) for i in range(blocks)]
        )
        self.out = nn.Conv1d(channels, out_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, t, j, c = x.shape
        y = x.reshape(n, t, j * c).permute(0, 2, 1)
        y = self.inp(y)
        for block in self.blocks:
            y = block(y)
        y = self.out(y).permute(0, 2, 1)
        return y.reshape(n, t, j, 3)


class PoseLifter3D:
    def __init__(
        self,
        temporal_window: int,
        lifter_checkpoint: str,
        use_lifter_checkpoint: bool,
    ) -> None:
        self.temporal_window = temporal_window
        self._buffers: dict[str, deque[np.ndarray]] = {}
        self._checkpoint = Path(lifter_checkpoint)
        self._model: Any = None
        self.info = self._init_model(use_lifter_checkpoint)

    def _init_model(self, use_lifter_checkpoint: bool) -> LifterInfo:
        if not use_lifter_checkpoint:
            return LifterInfo(False, "disabled_in_config")
        if torch is None:
            return LifterInfo(False, "torch_not_available")
        if not self._checkpoint.exists():
            return LifterInfo(False, "checkpoint_missing")

        model = _TemporalModel(num_joints=17, channels=1024, blocks=5)
        try:
            payload = torch.load(str(self._checkpoint), map_location="cpu")
            if isinstance(payload, dict) and "model_pos" in payload:
                payload = payload["model_pos"]
            model.load_state_dict(payload, strict=False)
            model.eval()
            self._model = model
            return LifterInfo(True, "checkpoint_loaded")
        except Exception:
            self._model = None
            return LifterInfo(False, "checkpoint_incompatible")

    def _normalize_xy(self, kpts_h36m: np.ndarray, width: int, height: int) -> np.ndarray:
        xy = kpts_h36m[:, :2].copy()
        xy[:, 0] = (xy[:, 0] / max(width, 1)) * 2.0 - 1.0
        xy[:, 1] = (xy[:, 1] / max(height, 1)) * 2.0 - 1.0
        return xy.astype(np.float32)

    def _fallback_depth(self, kpts_h36m: np.ndarray) -> np.ndarray:
        # Deterministic pseudo-depth from vertical displacement around pelvis.
        out = np.zeros((17, 3), dtype=np.float32)
        out[:, :2] = kpts_h36m[:, :2]
        pelvis_y = kpts_h36m[0, 1]
        out[:, 2] = (pelvis_y - kpts_h36m[:, 1]) * 0.005
        out -= out[0]
        return out

    def _get_buffer(self, cam_id: str) -> deque[np.ndarray]:
        if cam_id not in self._buffers:
            self._buffers[cam_id] = deque(maxlen=self.temporal_window)
        return self._buffers[cam_id]

    def reset(self, cam_id: str) -> None:
        self._buffers.pop(cam_id, None)

    def infer(
        self,
        cam_id: str,
        kpts_h36m: np.ndarray,
        image_shape: tuple[int, int, int],
    ) -> np.ndarray:
        h, w = image_shape[:2]
        normalized = self._normalize_xy(kpts_h36m, w, h)
        buf = self._get_buffer(cam_id)
        buf.append(normalized)
        while len(buf) < self.temporal_window:
            buf.appendleft(buf[0])

        if self._model is None:
            return self._fallback_depth(kpts_h36m)

        window = np.stack(list(buf), axis=0)  # (T,17,2)
        x = torch.from_numpy(window[None, ...])
        with torch.no_grad():
            pred = self._model(x).cpu().numpy()[0, -1]
        pred -= pred[0]
        return pred.astype(np.float32)
