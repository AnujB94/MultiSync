from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from dualpose.capture.transforms import build_transform
from dualpose.pose.detector2d import Detector2D
from dualpose.pose.keypoints import coco17_to_h36m17
from dualpose.pose.lifter3d import PoseLifter3D
from dualpose.pose.placement import PlacementFitter
from dualpose.ui.draw import draw_pose_overlay, render_3d_panel
from dualpose.utils.timing import RateMeter
from dualpose.utils.types import PoseResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-camera test script for 2D->3D pose + overlay (laptop webcam)."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Laptop webcam index.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=360, help="Capture height.")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--detector-mode", type=str, default="lightweight")
    parser.add_argument("--detector-backend", type=str, default="onnxruntime")
    parser.add_argument("--detector-device", type=str, default="cpu")
    parser.add_argument("--min-joint-confidence", type=float, default=0.2)
    parser.add_argument("--confidence-hold-frames", type=int, default=4)
    parser.add_argument("--temporal-window", type=int, default=243)
    parser.add_argument(
        "--lifter-checkpoint",
        type=str,
        default="checkpoints/cpn-ft-243-dbb-causal.bin",
    )
    parser.add_argument(
        "--no-lifter-checkpoint",
        action="store_true",
        help="Disable checkpoint loading and use fallback pseudo-depth.",
    )
    parser.add_argument("--weak-perspective", action="store_true", default=True)
    parser.add_argument("--fx-scale", type=float, default=1.1)
    parser.add_argument("--fy-scale", type=float, default=1.1)
    return parser.parse_args()


def open_webcam(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap


def main() -> None:
    args = parse_args()
    transform = build_transform(
        rotate=args.rotate,
        flip_horizontal=args.flip_horizontal,
        flip_vertical=args.flip_vertical,
    )
    detector = Detector2D(
        mode=args.detector_mode,
        backend=args.detector_backend,
        device=args.detector_device,
        min_joint_confidence=args.min_joint_confidence,
        confidence_hold_frames=args.confidence_hold_frames,
    )
    lifter = PoseLifter3D(
        temporal_window=args.temporal_window,
        lifter_checkpoint=args.lifter_checkpoint,
        use_lifter_checkpoint=not args.no_lifter_checkpoint,
    )
    fitter = PlacementFitter(
        weak_perspective=args.weak_perspective,
        fx_scale=args.fx_scale,
        fy_scale=args.fy_scale,
    )

    cap = open_webcam(args.camera_index, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError("Could not open laptop webcam.")

    window_title = "DualPose Test (Laptop Camera)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    fps_meter = RateMeter()
    frame_id = 0
    cam_id = "laptop_test"
    last_result: PoseResult | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t_capture = time.monotonic()
            frame_id += 1
            frame = transform(frame)
            fps_meter.tick(t_capture)

            t0 = time.monotonic()
            kpts2d = detector.detect(frame, cam_id=cam_id)
            result: PoseResult | None = None
            if kpts2d is not None:
                kpts_h36m = coco17_to_h36m17(kpts2d)
                kpts_3d_rel = lifter.infer(
                    cam_id=cam_id,
                    kpts_h36m=kpts_h36m,
                    image_shape=frame.shape,
                )
                fit = fitter.fit(
                    cam_id=cam_id,
                    kpts_3d_rel=kpts_3d_rel,
                    kpts_2d=kpts_h36m,
                    image_shape=frame.shape,
                )
                t_done = time.monotonic()
                result = PoseResult(
                    cam_id=cam_id,
                    frame_id=frame_id,
                    t_capture=t_capture,
                    kpts_2d=kpts_h36m,
                    kpts_3d_rel=kpts_3d_rel,
                    kpts_3d_cam=fit.kpts_3d_cam,
                    fit_error_px=fit.reprojection_error_px,
                    t_infer_done=t_done,
                    inference_ms=(t_done - t0) * 1000.0,
                    pose_age_ms=(t_done - t_capture) * 1000.0,
                    source_meta={"lifter_mode": lifter.info.reason},
                )
                last_result = result
            else:
                result = last_result

            overlay = draw_pose_overlay(frame, result)
            panel = render_3d_panel(
                result,
                width=max(260, frame.shape[1] // 2),
                height=frame.shape[0],
            )
            canvas = np.hstack([overlay, panel])

            infer_ms = 0.0 if result is None else result.inference_ms
            fit_err = 0.0 if result is None else result.fit_error_px
            status = (
                f"FPS {fps_meter.fps:4.1f} | infer {infer_ms:5.1f} ms | "
                f"fit {fit_err:4.1f}px | lifter={lifter.info.reason} | q=quit"
            )
            cv2.putText(
                canvas,
                status,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 160),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_title, canvas)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
