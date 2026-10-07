from __future__ import annotations

import time

import cv2
import numpy as np

from dualpose.ui.draw import draw_identity_box, draw_pose_overlay, render_3d_panel
from dualpose.utils.timing import CameraStats
from dualpose.utils.types import FramePacket, IdentityMatch, IdentityState, PoseResult


class Viewer:
    def __init__(self, title: str, target_fps: int) -> None:
        self.title = title
        self.target_fps = max(1, target_fps)
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)

    def _status_line(
        self,
        cam_id: str,
        stats: CameraStats | None,
        result: PoseResult | None,
    ) -> str:
        if stats is None:
            return f"{cam_id}: waiting"
        conn = "ok" if stats.connected else "down"
        fit = result.fit_error_px if result is not None else stats.last_fit_error_px
        return (
            f"{cam_id} conn={conn} cap_fps={stats.capture_rate.fps:4.1f} "
            f"infer_fps={stats.infer_rate.fps:4.1f} inf_ms={stats.last_inference_ms:4.1f} "
            f"pose_age={stats.last_pose_age_ms:4.0f}ms fit={fit:4.1f}px drop={stats.dropped_frames}"
        )

    def render(
        self,
        frame_a: FramePacket | None,
        frame_b: FramePacket | None,
        pose_a: PoseResult | None,
        pose_b: PoseResult | None,
        identity: IdentityMatch | None,
        stats: dict[str, CameraStats],
        recording: bool,
    ) -> int:
        now = time.monotonic()
        if frame_a is None and frame_b is None:
            blank = np.zeros((480, 1280, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Waiting for cameras...",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (240, 240, 240),
                2,
            )
            cv2.imshow(self.title, blank)
            return cv2.waitKey(1) & 0xFF

        base_frame = frame_a if frame_a is not None else frame_b
        assert base_frame is not None
        base_h, base_w = base_frame.image.shape[:2]

        img_a = (
            draw_pose_overlay(frame_a.image, pose_a)
            if frame_a is not None
            else self._missing_feed(base_w, base_h, "No webcam frame")
        )
        img_b = (
            draw_pose_overlay(frame_b.image, pose_b)
            if frame_b is not None
            else self._missing_feed(base_w, base_h, "No phone frame")
        )
        if identity is not None and identity.state == IdentityState.SAME:
            if pose_a is not None:
                img_a = draw_identity_box(img_a, pose_a.bbox_xyxy, "Person 1")
            if pose_b is not None:
                img_b = draw_identity_box(img_b, pose_b.bbox_xyxy, "Person 1")

        h = min(img_a.shape[0], img_b.shape[0], base_h)
        w = min(img_a.shape[1], img_b.shape[1], base_w)
        img_a = cv2.resize(img_a, (w, h), interpolation=cv2.INTER_LINEAR)
        img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_LINEAR)

        panel_a = render_3d_panel(pose_a, w, max(180, h // 2))
        panel_b = render_3d_panel(pose_b, w, max(180, h // 2))

        top = np.hstack([img_a, img_b])
        bottom = np.hstack([panel_a, panel_b])

        strip_h = 80
        strip = np.zeros((strip_h, top.shape[1], 3), dtype=np.uint8)
        strip[:, :] = (30, 30, 30)

        cam_a_id = (
            pose_a.cam_id
            if pose_a is not None
            else (frame_a.cam_id if frame_a is not None else "laptop")
        )
        cam_b_id = (
            pose_b.cam_id
            if pose_b is not None
            else (frame_b.cam_id if frame_b is not None else "phone")
        )
        stats_a = stats.get(cam_a_id)
        stats_b = stats.get(cam_b_id)
        if stats_a is not None and frame_a is not None:
            stats_a.last_capture_age_ms = (now - frame_a.t_capture) * 1000.0
        if stats_b is not None and frame_b is not None:
            stats_b.last_capture_age_ms = (now - frame_b.t_capture) * 1000.0

        line1 = self._status_line(cam_a_id, stats_a, pose_a)
        line2 = self._status_line(cam_b_id, stats_b, pose_b)
        rec = "REC" if recording else "IDLE"
        cv2.putText(
            strip,
            line1,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            strip,
            line2,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        if identity is None:
            match_line = "match=disabled"
        elif identity.evidence is None:
            match_line = f"match={identity.state.value}"
        else:
            ev = identity.evidence
            fs = "n/a" if ev.fused_score is None else f"{ev.fused_score:.2f}"
            ge = "n/a" if ev.epipolar_error_px is None else f"{ev.epipolar_error_px:.2f}px"
            ap = "n/a" if ev.appearance_score is None else f"{ev.appearance_score:.2f}"
            gm = f"{ev.geometry_source}/{ev.geometry_state}"
            match_line = (
                f"match={identity.state.value} mode={ev.mode} score={fs} geo={ge} "
                f"app={ap} geom={gm} skew={ev.skew_ms:.0f}ms reason={ev.reason}"
            )

        cv2.putText(
            strip,
            f"[{rec}] q=quit r=record | {match_line}",
            (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (150, 255, 150),
            1,
            cv2.LINE_AA,
        )

        canvas = np.vstack([top, bottom, strip])
        cv2.imshow(self.title, canvas)
        wait_ms = int(round(1000.0 / self.target_fps))
        return cv2.waitKey(wait_ms) & 0xFF

    def _missing_feed(self, width: int, height: int, label: str) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:, :] = (22, 22, 22)
        cv2.putText(
            panel,
            label,
            (20, max(40, height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (170, 170, 170),
            2,
            cv2.LINE_AA,
        )
        return panel

    def close(self) -> None:
        cv2.destroyWindow(self.title)
