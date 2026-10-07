from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from dualpose.calibration import StereoCalibration
from dualpose.capture import NetworkStreamCapture, WebcamCapture
from dualpose.config import AppConfig
from dualpose.matching.auto_geometry import AutoGeometryEstimator
from dualpose.matching.body_descriptors import extract_body_descriptor
from dualpose.matching.matcher import IdentityMatcher
from dualpose.pose.detector2d import Detector2D
from dualpose.pose.keypoints import coco17_to_h36m17
from dualpose.pose.lifter3d import PoseLifter3D
from dualpose.pose.placement import PlacementFitter
from dualpose.recording import SessionRecorder
from dualpose.ui.viewer import Viewer
from dualpose.utils.timing import CameraStats
from dualpose.utils.types import IdentityMatch, PersonObservation, PoseResult


@dataclass
class _CamRuntime:
    cam_id: str
    capture: WebcamCapture | NetworkStreamCapture
    stats: CameraStats
    last_processed_frame_id: int = -1
    last_pose: PoseResult | None = None
    missing_count: int = 0


class _PoseStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, PoseResult] = {}

    def publish(self, result: PoseResult) -> None:
        with self._lock:
            self._latest[result.cam_id] = result

    def get(self, cam_id: str) -> PoseResult | None:
        with self._lock:
            return self._latest.get(cam_id)


class DualPoseApp:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.viewer = Viewer(config.window.title, config.window.target_fps)
        self.detector = Detector2D(
            mode=config.pipeline.detector_mode,
            backend=config.pipeline.detector_backend,
            device=config.pipeline.detector_device,
            min_joint_confidence=config.pipeline.min_joint_confidence,
            confidence_hold_frames=config.pipeline.confidence_hold_frames,
        )
        self.lifter = PoseLifter3D(
            temporal_window=config.pipeline.temporal_window,
            lifter_checkpoint=config.pipeline.lifter_checkpoint,
            use_lifter_checkpoint=config.pipeline.use_lifter_checkpoint,
        )
        self.fitter = PlacementFitter(
            weak_perspective=config.pipeline.weak_perspective,
            fx_scale=config.pipeline.fx_scale,
            fy_scale=config.pipeline.fy_scale,
        )
        self.recorder = SessionRecorder(config.recording.output_dir)
        if config.recording.enabled:
            self.recorder.start()

        self.webcam = WebcamCapture(config.webcam)
        self.phone = NetworkStreamCapture(config.phone)
        self.calibration = self._load_calibration()
        self.matcher = IdentityMatcher(
            cam_a_id=config.webcam.cam_id,
            cam_b_id=config.phone.cam_id,
            config=config.matching,
            calibration=self.calibration,
        )
        self.auto_geometry = (
            AutoGeometryEstimator(config.matching)
            if config.matching.enabled and config.matching.geometry_mode == "auto"
            else None
        )
        self._runtime_calibration_checked = False
        self.runtimes = [
            _CamRuntime(config.webcam.cam_id, self.webcam, CameraStats()),
            _CamRuntime(config.phone.cam_id, self.phone, CameraStats()),
        ]
        self.by_cam_id = {rt.cam_id: rt for rt in self.runtimes}

        self._stop_event = threading.Event()
        self._pose_store = _PoseStore()
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="inference-worker",
            daemon=True,
        )
        self._geometry_thread = threading.Thread(
            target=self._geometry_loop,
            name="geometry-worker",
            daemon=True,
        )

    def _load_calibration(self) -> StereoCalibration | None:
        if not self.cfg.calibration.enabled:
            return None
        try:
            calibration = StereoCalibration.load(self.cfg.calibration.path)
        except Exception as exc:
            print(f"[calibration] failed to load calibration file: {exc}")
            return None

        ok, reason = calibration.validate_runtime(
            image_size_1=(self.cfg.webcam.width, self.cfg.webcam.height),
            image_size_2=(self.cfg.phone.width, self.cfg.phone.height),
            transform_meta_1={
                "rotate": self.cfg.webcam.rotate,
                "flip_horizontal": self.cfg.webcam.flip_horizontal,
                "flip_vertical": self.cfg.webcam.flip_vertical,
            },
            transform_meta_2={
                "rotate": self.cfg.phone.rotate,
                "flip_horizontal": self.cfg.phone.flip_horizontal,
                "flip_vertical": self.cfg.phone.flip_vertical,
            },
        )
        if not ok:
            msg = f"[calibration] warning: {reason}"
            if self.cfg.calibration.require_transform_match:
                print(f"{msg}. Matching disabled.")
                return None
            print(f"{msg}. Continuing because require_transform_match=false.")
        return calibration

    def start(self) -> None:
        self.webcam.start()
        self.phone.start()
        self._inference_thread.start()
        self._geometry_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.webcam.stop()
        self.phone.stop()
        self.webcam.join(timeout=2.0)
        self.phone.join(timeout=2.0)
        self._inference_thread.join(timeout=2.0)
        self._geometry_thread.join(timeout=2.0)
        self.recorder.stop()
        self.viewer.close()

    def _inference_loop(self) -> None:
        idx = 0
        infer_every = max(1, self.cfg.pipeline.infer_every_n_frames)
        lifter_stride = max(1, self.cfg.pipeline.lifter_stride)
        while not self._stop_event.is_set():
            runtime = self.runtimes[idx % len(self.runtimes)]
            idx += 1
            packet = runtime.capture.buffer.get_latest()
            runtime.stats.connected = runtime.capture.state.connected
            runtime.stats.dropped_frames = runtime.capture.buffer.dropped_frames
            if packet is None:
                time.sleep(0.002)
                continue

            runtime.stats.capture_rate.tick()
            if packet.frame_id == runtime.last_processed_frame_id:
                runtime.stats.skipped_frames += 1
                time.sleep(0.001)
                continue
            runtime.last_processed_frame_id = packet.frame_id

            if packet.frame_id % infer_every != 0:
                runtime.stats.skipped_frames += 1
                continue

            start = time.monotonic()
            detection = self.detector.detect_with_meta(packet.image, cam_id=packet.cam_id)
            if detection is None:
                runtime.missing_count += 1
                if runtime.missing_count > self.cfg.pipeline.confidence_hold_frames:
                    self.detector.reset(packet.cam_id)
                    self.lifter.reset(packet.cam_id)
                continue
            runtime.missing_count = 0

            kpts2d = detection.keypoints
            kpts_h36m = coco17_to_h36m17(kpts2d)
            if packet.frame_id % lifter_stride == 0 or runtime.last_pose is None:
                kpts_3d_rel = self.lifter.infer(
                    cam_id=packet.cam_id,
                    kpts_h36m=kpts_h36m,
                    image_shape=packet.image.shape,
                )
            else:
                kpts_3d_rel = runtime.last_pose.kpts_3d_rel

            fit = self.fitter.fit(
                cam_id=packet.cam_id,
                kpts_3d_rel=kpts_3d_rel,
                kpts_2d=kpts_h36m,
                image_shape=packet.image.shape,
            )
            done = time.monotonic()
            infer_ms = (done - start) * 1000.0
            pose_age_ms = (done - packet.t_capture) * 1000.0

            result = PoseResult(
                cam_id=packet.cam_id,
                frame_id=packet.frame_id,
                t_capture=packet.t_capture,
                kpts_2d=kpts_h36m,
                kpts_3d_rel=kpts_3d_rel,
                kpts_3d_cam=fit.kpts_3d_cam,
                fit_error_px=fit.reprojection_error_px,
                t_infer_done=done,
                inference_ms=infer_ms,
                pose_age_ms=pose_age_ms,
                source_meta={"lifter_mode": self.lifter.info.reason},
                bbox_xyxy=detection.bbox_xyxy,
                detection_fresh=detection.is_fresh,
                detection_confidence=detection.confidence,
            )
            runtime.last_pose = result
            runtime.stats.infer_rate.tick()
            runtime.stats.last_inference_ms = infer_ms
            runtime.stats.last_pose_age_ms = pose_age_ms
            runtime.stats.last_fit_error_px = fit.reprojection_error_px
            self._pose_store.publish(result)

            embedding = self.matcher.appearance.update(
                cam_id=packet.cam_id,
                image=packet.image,
                bbox_xyxy=detection.bbox_xyxy,
                t_capture=packet.t_capture,
                keypoints=kpts_h36m,
            )
            desc = extract_body_descriptor(
                image_bgr=packet.image,
                bbox_xyxy=detection.bbox_xyxy,
                kpts_2d=kpts_h36m,
            )
            obs = PersonObservation(
                cam_id=packet.cam_id,
                frame_id=packet.frame_id,
                t_capture=packet.t_capture,
                kpts_2d=kpts_h36m,
                bbox_xyxy=detection.bbox_xyxy,
                confidence=detection.confidence,
                detection_fresh=detection.is_fresh,
                image_shape=packet.image.shape,
                appearance_embedding=embedding,
                upper_hist=desc.upper_hist,
                lower_hist=desc.lower_hist,
                body_proportions=desc.proportions,
                pose_result=result,
            )
            self.matcher.submit(obs)

    def _geometry_loop(self) -> None:
        if self.auto_geometry is None:
            return
        sleep_s = max(0.05, self.cfg.matching.auto_health_check_interval_ms / 1000.0 / 2.0)
        while not self._stop_event.is_set():
            frame_a = self.webcam.buffer.get_latest()
            frame_b = self.phone.buffer.get_latest()
            if frame_a is None or frame_b is None:
                time.sleep(sleep_s)
                continue
            pose_a = self._pose_store.get(self.cfg.webcam.cam_id)
            pose_b = self._pose_store.get(self.cfg.phone.cam_id)
            bbox_a = None if pose_a is None else pose_a.bbox_xyxy
            bbox_b = None if pose_b is None else pose_b.bbox_xyxy
            self.auto_geometry.observe_pair(
                image_a=frame_a.image,
                image_b=frame_b.image,
                bbox_a=bbox_a,
                bbox_b=bbox_b,
            )
            self.matcher.update_auto_geometry(
                model=self.auto_geometry.model,
                state=self.auto_geometry.status.state,
                reason=self.auto_geometry.status.reason,
            )
            time.sleep(sleep_s)

    def run(self) -> None:
        self.start()
        try:
            while True:
                frame_a = self.webcam.buffer.get_latest()
                frame_b = self.phone.buffer.get_latest()
                if (
                    not self._runtime_calibration_checked
                    and self.calibration is not None
                    and frame_a is not None
                    and frame_b is not None
                ):
                    ok, reason = self.calibration.validate_runtime(
                        image_size_1=(frame_a.image.shape[1], frame_a.image.shape[0]),
                        image_size_2=(frame_b.image.shape[1], frame_b.image.shape[0]),
                        transform_meta_1={
                            "rotate": self.cfg.webcam.rotate,
                            "flip_horizontal": self.cfg.webcam.flip_horizontal,
                            "flip_vertical": self.cfg.webcam.flip_vertical,
                        },
                        transform_meta_2={
                            "rotate": self.cfg.phone.rotate,
                            "flip_horizontal": self.cfg.phone.flip_horizontal,
                            "flip_vertical": self.cfg.phone.flip_vertical,
                        },
                    )
                    self._runtime_calibration_checked = True
                    if not ok and self.cfg.calibration.require_transform_match:
                        print(
                            f"[calibration] runtime mismatch: {reason}. "
                            "Calibrated geometry disabled."
                        )
                        self.matcher.disable_calibrated_geometry()
                if frame_a is not None:
                    self.recorder.write(frame_a)
                if frame_b is not None:
                    self.recorder.write(frame_b)

                pose_a = self._pose_store.get(self.cfg.webcam.cam_id)
                pose_b = self._pose_store.get(self.cfg.phone.cam_id)
                identity: IdentityMatch | None
                if self.matcher.enabled:
                    identity = self.matcher.get_state()
                else:
                    identity = None
                key = self.viewer.render(
                    frame_a=frame_a,
                    frame_b=frame_b,
                    pose_a=pose_a,
                    pose_b=pose_b,
                    identity=identity,
                    stats={k: v.stats for k, v in self.by_cam_id.items()},
                    recording=self.recorder.enabled,
                )
                if key == ord("q"):
                    break
                if key == ord("r"):
                    if self.recorder.enabled:
                        self.recorder.stop()
                    else:
                        self.recorder.start()
        finally:
            self.stop()
