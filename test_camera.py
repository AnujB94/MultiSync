from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick stream tester for webcam/RTSP/HTTP/HTTPS links."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Camera source. Examples: 0 (webcam index), "
            "rtsp://..., http://..., https://..."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.local.yaml",
        help="Optional config file to read phone.source if --source is not provided.",
    )
    parser.add_argument(
        "--open-timeout-ms",
        type=int,
        default=2000,
        help="Open timeout in milliseconds (FFmpeg backend only).",
    )
    parser.add_argument(
        "--read-timeout-ms",
        type=int,
        default=2000,
        help="Read timeout in milliseconds (FFmpeg backend only).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Optional requested frame width. 0 keeps backend default.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Optional requested frame height. 0 keeps backend default.",
    )
    return parser.parse_args()


def _parse_source_text(value: str) -> int | str:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return value


def _source_from_config(config_path: str) -> int | str:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No --source provided and config file not found: {config_path}"
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config is not a mapping.")

    phone = raw.get("phone")
    if not isinstance(phone, dict):
        raise ValueError("Config is missing 'phone' section.")
    source = phone.get("source")
    if source is None:
        raise ValueError("Config is missing 'phone.source'.")
    if not isinstance(source, (int, str)):
        raise ValueError("Config 'phone.source' must be int or str.")
    return source


def _open_capture(
    source: int | str,
    open_timeout_ms: int,
    read_timeout_ms: int,
    width: int,
    height: int,
) -> cv2.VideoCapture:
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        src_lower = source.lower()
        use_ffmpeg = src_lower.startswith(("rtsp://", "http://", "https://"))
        if use_ffmpeg:
            try:
                params = [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    float(open_timeout_ms),
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    float(read_timeout_ms),
                ]
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
            except (cv2.error, TypeError):
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(open_timeout_ms))
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(read_timeout_ms))
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap


def _stream_kind(source: int | str) -> str:
    if isinstance(source, int):
        return "webcam-index"
    src = source.lower()
    if src.startswith("rtsp://"):
        return "rtsp"
    if src.startswith("https://"):
        return "https"
    if src.startswith("http://"):
        return "http"
    return "other"


def _annotate(frame: Any, fps: float, source: int | str) -> Any:
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        f"source={source}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"resolution={w}x{h} fps={fps:4.1f}",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "q=quit",
        (10, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 180),
        1,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    args = parse_args()

    source: int | str
    if args.source is not None:
        source = _parse_source_text(args.source)
    else:
        source = _source_from_config(args.config)

    kind = _stream_kind(source)
    print(f"[test_camera] source kind: {kind}")
    print(f"[test_camera] opening: {source}")

    if kind == "https":
        print(
            "[note] HTTPS works only if the URL is a real video stream endpoint "
            "(for example MJPEG/HLS over HTTPS) and your OpenCV/FFmpeg build supports it."
        )

    cap = _open_capture(
        source=source,
        open_timeout_ms=args.open_timeout_ms,
        read_timeout_ms=args.read_timeout_ms,
        width=args.width,
        height=args.height,
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open source. Verify URL/credentials/network. "
            "For IP cameras, RTSP is usually the most reliable option."
        )

    window_title = "Camera Link Test"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    t_prev = time.monotonic()
    fps = 0.0
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[test_camera] read failed; stream may have dropped.")
                break

            frame_count += 1
            t_now = time.monotonic()
            dt = t_now - t_prev
            if dt > 0:
                inst = 1.0 / dt
                # Exponential moving average for stable display.
                fps = inst if fps == 0.0 else (0.9 * fps + 0.1 * inst)
            t_prev = t_now

            vis = _annotate(frame, fps=fps, source=source)
            cv2.imshow(window_title, vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[test_camera] frames received: {frame_count}")


if __name__ == "__main__":
    main()
