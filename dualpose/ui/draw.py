from __future__ import annotations

import cv2
import numpy as np

from dualpose.pose.keypoints import H36M_BONES
from dualpose.utils.types import PoseResult


def _depth_color(z: float, z_min: float, z_max: float) -> tuple[int, int, int]:
    if z_max - z_min < 1e-6:
        t = 0.5
    else:
        t = float((z - z_min) / (z_max - z_min))
    # Near warm (red-ish), far cool (blue-ish) in BGR.
    b = int(255 * t)
    r = int(255 * (1.0 - t))
    g = int(180 * (1.0 - abs(0.5 - t) * 2.0))
    return (b, g, r)


def _overlay_points_from_result(result: PoseResult) -> np.ndarray:
    # For weak perspective this already carries pixel-space x/y.
    # For pinhole fit, these points are camera-space; fallback to 2D keypoints.
    pts = result.kpts_3d_cam[:, :2]
    if np.all(np.isfinite(pts)) and np.max(np.abs(pts)) < 20000:
        return pts
    return result.kpts_2d[:, :2]


def draw_pose_overlay(image: np.ndarray, result: PoseResult | None) -> np.ndarray:
    out = image.copy()
    if result is None:
        return out

    xy = _overlay_points_from_result(result)
    conf = result.kpts_2d[:, 2]
    z = result.kpts_3d_cam[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))

    for a, b in H36M_BONES:
        if conf[a] < 0.05 or conf[b] < 0.05:
            continue
        pa = tuple(np.round(xy[a]).astype(int))
        pb = tuple(np.round(xy[b]).astype(int))
        c = _depth_color((z[a] + z[b]) * 0.5, z_min, z_max)
        cv2.line(out, pa, pb, c, 2, cv2.LINE_AA)

    for j in range(17):
        if conf[j] < 0.05:
            continue
        p = tuple(np.round(xy[j]).astype(int))
        radius = 3 if conf[j] > 0.3 else 2
        c = _depth_color(z[j], z_min, z_max)
        cv2.circle(out, p, radius, c, -1, cv2.LINE_AA)
    return out


def draw_identity_box(
    image: np.ndarray,
    bbox_xyxy: np.ndarray | None,
    label: str,
    color: tuple[int, int, int] = (80, 255, 80),
) -> np.ndarray:
    if bbox_xyxy is None:
        return image
    out = image.copy()
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    h, w = out.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return out
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.putText(
        out,
        label,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


def render_3d_panel(result: PoseResult | None, width: int, height: int) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:, :] = (24, 24, 24)
    if result is None:
        cv2.putText(panel, "No Pose", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
        return panel

    pts = result.kpts_3d_rel.astype(np.float32).copy()
    pts -= pts[0]

    # Fixed yaw rotation to reveal depth in a compact panel.
    yaw = np.deg2rad(35.0)
    rot = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    p = pts @ rot.T
    x = p[:, 0]
    y = -p[:, 1]
    z = p[:, 2]

    if np.max(np.abs(x)) > 1e-6:
        x = x / np.max(np.abs(x))
    if np.max(np.abs(y)) > 1e-6:
        y = y / np.max(np.abs(y))
    px = ((x * 0.4 + 0.5) * width).astype(int)
    py = ((y * 0.4 + 0.5) * height).astype(int)

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    for a, b in H36M_BONES:
        pa = (int(px[a]), int(py[a]))
        pb = (int(px[b]), int(py[b]))
        c = _depth_color((z[a] + z[b]) * 0.5, z_min, z_max)
        cv2.line(panel, pa, pb, c, 2, cv2.LINE_AA)

    cv2.putText(panel, "3D view", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    return panel
