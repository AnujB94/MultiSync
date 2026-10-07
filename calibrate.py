from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dualpose.calibration import StereoCalibrator
from dualpose.capture.transforms import build_transform
from dualpose.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkerboard stereo calibration tool.")
    parser.add_argument("--config", type=str, default="config.local.yaml")
    parser.add_argument("--output", type=str, default="calibration/stereo_calibration.npz")
    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="Checkerboard inner corners along columns.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Checkerboard inner corners along rows.",
    )
    parser.add_argument(
        "--square-size-m",
        type=float,
        default=0.025,
        help="Square size in meters.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=20,
        help="Minimum captured pairs to calibrate.",
    )
    return parser.parse_args()


def _open_camera(source: int | str, width: int, height: int, is_network: bool) -> cv2.VideoCapture:
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        if is_network:
            try:
                params = [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    3000.0,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    3000.0,
                ]
                cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, params)
            except (cv2.error, TypeError):
                cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(str(source))
        else:
            cap = cv2.VideoCapture(str(source))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap


def _draw_status(
    canvas: np.ndarray,
    text: str,
    line: int,
    color: tuple[int, int, int] = (180, 255, 180),
) -> None:
    cv2.putText(
        canvas,
        text,
        (10, 24 + line * 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    transform_webcam = build_transform(
        rotate=cfg.webcam.rotate,
        flip_horizontal=cfg.webcam.flip_horizontal,
        flip_vertical=cfg.webcam.flip_vertical,
    )
    transform_phone = build_transform(
        rotate=cfg.phone.rotate,
        flip_horizontal=cfg.phone.flip_horizontal,
        flip_vertical=cfg.phone.flip_vertical,
    )

    cap_a = _open_camera(cfg.webcam.source, cfg.webcam.width, cfg.webcam.height, is_network=False)
    cap_b = _open_camera(cfg.phone.source, cfg.phone.width, cfg.phone.height, is_network=True)
    if not cap_a.isOpened():
        raise RuntimeError("Could not open webcam source.")
    if not cap_b.isOpened():
        raise RuntimeError("Could not open phone stream source.")

    calibrator = StereoCalibrator(
        inner_corners=(args.cols, args.rows),
        square_size_m=args.square_size_m,
    )

    win_name = "Stereo Calibration"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()
            if not ok_a or frame_a is None or not ok_b or frame_b is None:
                continue

            frame_a = transform_webcam(frame_a)
            frame_b = transform_phone(frame_b)
            ok_pair, corners_a, corners_b = calibrator.detect_pair(frame_a, frame_b)

            vis_a = frame_a.copy()
            vis_b = frame_b.copy()
            if ok_pair and corners_a is not None and corners_b is not None:
                cv2.drawChessboardCorners(vis_a, (args.cols, args.rows), corners_a, ok_pair)
                cv2.drawChessboardCorners(vis_b, (args.cols, args.rows), corners_b, ok_pair)

            h = min(vis_a.shape[0], vis_b.shape[0])
            w = min(vis_a.shape[1], vis_b.shape[1])
            vis_a = cv2.resize(vis_a, (w, h))
            vis_b = cv2.resize(vis_b, (w, h))
            canvas = np.hstack([vis_a, vis_b])

            _draw_status(canvas, f"captured_pairs={len(calibrator.obj_points)}", 0)
            _draw_status(canvas, "space: capture pair  c: calibrate+save  q: quit", 1)
            if not ok_pair:
                _draw_status(
                    canvas,
                    "checkerboard not found in both views",
                    2,
                    color=(180, 180, 255),
                )

            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                ok, msg = calibrator.add_pair(frame_a, frame_b)
                print(f"[capture] {msg}")
            if key == ord("c"):
                calibration = calibrator.calibrate(
                    transform_meta_1={
                        "rotate": cfg.webcam.rotate,
                        "flip_horizontal": cfg.webcam.flip_horizontal,
                        "flip_vertical": cfg.webcam.flip_vertical,
                    },
                    transform_meta_2={
                        "rotate": cfg.phone.rotate,
                        "flip_horizontal": cfg.phone.flip_horizontal,
                        "flip_vertical": cfg.phone.flip_vertical,
                    },
                    min_pairs=args.min_pairs,
                )
                output = Path(args.output)
                calibration.save(output)
                print(f"[save] wrote calibration to {output}")
                print(f"[save] metrics={calibration.metrics}")
                break
    finally:
        cap_a.release()
        cap_b.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
