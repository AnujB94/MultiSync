from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from dualpose.utils.types import FramePacket


@dataclass
class _CamRecorder:
    writer: cv2.VideoWriter
    ts_file: object
    ts_writer: csv.writer


class SessionRecorder:
    def __init__(self, output_dir: str) -> None:
        self.output_root = Path(output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._recorders: dict[str, _CamRecorder] = {}
        self._enabled = False
        self._session_dir: Path | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if self._enabled:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._session_dir = self.output_root / stamp
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = True

    def stop(self) -> None:
        if not self._enabled:
            return
        for rec in self._recorders.values():
            rec.writer.release()
            rec.ts_file.close()
        self._recorders.clear()
        self._enabled = False

    def _create_cam_recorder(self, packet: FramePacket) -> _CamRecorder:
        assert self._session_dir is not None
        h, w = packet.image.shape[:2]
        video_path = self._session_dir / f"{packet.cam_id}.mp4"
        ts_path = self._session_dir / f"{packet.cam_id}_timestamps.csv"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
        ts_file = ts_path.open("w", newline="", encoding="utf-8")
        ts_writer = csv.writer(ts_file)
        ts_writer.writerow(["frame_id", "t_capture"])
        return _CamRecorder(writer=writer, ts_file=ts_file, ts_writer=ts_writer)

    def write(self, packet: FramePacket) -> None:
        if not self._enabled:
            return
        rec = self._recorders.get(packet.cam_id)
        if rec is None:
            rec = self._create_cam_recorder(packet)
            self._recorders[packet.cam_id] = rec
        rec.writer.write(packet.image)
        rec.ts_writer.writerow([packet.frame_id, f"{packet.t_capture:.6f}"])
