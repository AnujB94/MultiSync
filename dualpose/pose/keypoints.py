from __future__ import annotations

import numpy as np

COCO17 = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

H36M17 = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]

H36M_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]

H36M_LEFT = [4, 5, 6, 11, 12, 13]
H36M_RIGHT = [1, 2, 3, 14, 15, 16]


def _safe_mean(points: np.ndarray) -> np.ndarray:
    conf = points[:, 2]
    valid = conf > 0
    if not np.any(valid):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    xy = points[valid, :2].mean(axis=0)
    c = float(conf[valid].mean())
    return np.array([xy[0], xy[1], c], dtype=np.float32)


def coco17_to_h36m17(kpts: np.ndarray) -> np.ndarray:
    """Convert COCO-17 (x, y, conf) to approximate H36M-17 ordering."""
    if kpts.shape != (17, 3):
        raise ValueError(f"Expected (17,3), got {kpts.shape}")

    out = np.zeros((17, 3), dtype=np.float32)

    lhip = kpts[11]
    rhip = kpts[12]
    lshoulder = kpts[5]
    rshoulder = kpts[6]
    nose = kpts[0]
    leye = kpts[1]
    reye = kpts[2]
    lear = kpts[3]
    rear = kpts[4]

    pelvis = _safe_mean(np.vstack([lhip, rhip]))
    thorax = _safe_mean(np.vstack([lshoulder, rshoulder]))
    spine = np.array(
        [
            (pelvis[0] + thorax[0]) * 0.5,
            (pelvis[1] + thorax[1]) * 0.5,
            min(pelvis[2], thorax[2]),
        ],
        dtype=np.float32,
    )
    head = _safe_mean(np.vstack([nose, leye, reye, lear, rear]))
    neck = _safe_mean(np.vstack([nose, lshoulder, rshoulder]))

    out[0] = pelvis
    out[1] = rhip
    out[2] = kpts[14]
    out[3] = kpts[16]
    out[4] = lhip
    out[5] = kpts[13]
    out[6] = kpts[15]
    out[7] = spine
    out[8] = thorax
    out[9] = neck
    out[10] = head
    out[11] = lshoulder
    out[12] = kpts[7]
    out[13] = kpts[9]
    out[14] = rshoulder
    out[15] = kpts[8]
    out[16] = kpts[10]
    return out
