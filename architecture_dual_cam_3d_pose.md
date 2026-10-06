# Step 1: Dual-Camera Live Monocular 3D Pose Viewer

## Goal

One window showing two live camera views (laptop webcam + phone camera over Wi-Fi), each with a **3D pose overlay** computed independently from that camera alone. No cross-camera matching, no occlusion handling yet. Those come in later steps, but the design below leaves hooks for them.

**Scope**
- 2 static cameras, same Wi-Fi network
- 1 person in frame (pick the largest/most confident detection)
- Per-camera monocular 3D pose, drawn on the video and in a small 3D viewport
- Target: 15-30 FPS per camera on a laptop (GPU strongly preferred)

---

## 1. High-Level Architecture

```
┌──────────────┐                              ┌───────────────────────────────┐
│ Laptop webcam│── cv2.VideoCapture ─────────▶│ CaptureThread A               │
└──────────────┘                              │  (latest-frame buffer + ts)   │
                                              └──────────────┬────────────────┘
┌──────────────┐   Wi-Fi (RTSP / MJPEG /      ┌──────────────┴────────────────┐
│ Phone camera │──────── WebRTC) ────────────▶│ CaptureThread B               │
└──────────────┘                              │  (latest-frame buffer + ts)   │
                                              └──────────────┬────────────────┘
                                                             │
                                  ┌──────────────────────────┴───────────────┐
                                  │ Inference worker (one per camera)        │
                                  │  1. 2D pose (RTMPose)                    │
                                  │  2. Keypoint conversion (COCO → H36M)    │
                                  │  3. Temporal buffer (causal window)      │
                                  │  4. 3D lifter (MotionBERT / VP3D)        │
                                  │  5. Placement fit (root pos + scale)     │
                                  └──────────────────────────┬───────────────┘
                                                             │ PoseResult
                                                             ▼
                                              ┌───────────────────────────────┐
                                              │ UI / Renderer (main thread)   │
                                              │  - 2 video panels w/ overlay  │
                                              │  - 3D viewport (optional)     │
                                              └───────────────────────────────┘
```

**Key principle:** capture, inference, and rendering run decoupled. Capture always keeps only the *latest* frame (drop old ones), so a slow model never builds up latency.

---

## 2. Phone → Laptop Video Transport

The phone is the only non-trivial I/O piece. Options, simplest first:

| Option | How | Pros | Cons |
|---|---|---|---|
| **IP-camera app (recommended to start)** | Phone app serves RTSP or MJPEG; laptop opens URL with OpenCV | 10 minutes to set up, no code on phone | Android has good options (e.g. "IP Webcam"); iOS options are fewer and varied, so verify what's current |
| **Phone as webcam apps** (DroidCam, Camo, iVCam) | Laptop sees a virtual webcam device | Trivial on the laptop side, `VideoCapture(index)` | Closed apps, sometimes limited resolution/free tier, less control over timestamps |
| **WebRTC via browser** | Phone opens a web page served by the laptop, streams camera via WebRTC; laptop receives with `aiortc` | No app install, low latency, works on iPhone and Android | More code; needs HTTPS (self-signed cert) for camera access on the phone |
| **Custom app** (send H.264 over UDP/RTP) | Native app | Full control | Overkill for now |

**Recommendation:** start with an IP-camera app and `cv2.VideoCapture("rtsp://..." or "http://.../video")`. If latency or reliability disappoints, move to the WebRTC option, which is the best long-term fit since it also gives you a path to timestamps and bitrate control.

**Transport notes**
- Use 5 GHz Wi-Fi, same router, ideally laptop on Ethernet if possible.
- Stream at 720p, 30 fps. Higher resolution costs bandwidth and decode time and buys little for pose.
- Lock exposure/focus on the phone if the app allows (auto-focus hunting disturbs keypoints).
- Disable phone sleep and plug it in; streaming drains battery and heats the phone.
- Reconnect logic: if frames stop for >2 s, close and reopen the stream.

---

## 3. Components

### 3.1 CaptureThread (one per camera)

- Loops on `cap.read()`, stores `(frame, t_capture)` in a single-slot buffer, overwriting the previous one.
- `t_capture = time.monotonic()` at receipt on the laptop. (Phone-side timestamps aren't trustworthy yet; sync is a later-step problem, but record the timestamp now so the data model is ready.)
- Exposes `get_latest() -> (frame, t_capture, frame_id)`.
- For the phone stream, set the OpenCV buffer size to 1 where supported (`CAP_PROP_BUFFERSIZE`), otherwise the dedicated thread's overwrite behavior handles it.

### 3.2 2D Pose Detector

- **RTMPose** (via `rtmlib` for lightweight ONNX runtime inference, or MMPose for the full stack). Alternatives: ViTPose, YOLO-pose.
- Output: 17 COCO keypoints `(x, y, confidence)` in pixel coordinates.
- If multiple people are detected, keep the one with the largest bounding box (simple, replaceable later by tracking).
- Low-confidence joints: carry the previous frame's value for a few frames, or interpolate. The lifter degrades badly on missing joints.

### 3.3 Keypoint Conversion

Most pretrained lifters (MotionBERT, VideoPose3D, MixSTE) use the **Human3.6M 17-joint skeleton**, which differs from COCO-17. Convert:

- Pelvis (hip center) = mean of left/right hip
- Thorax = mean of left/right shoulder
- Spine = midpoint of pelvis and thorax (approximation)
- Neck/Nose/Head = derived from nose and ears
- Reorder remaining joints to H36M order

Alternatively use a Halpe-26 or Human3.6M-trained 2D detector so no conversion is needed. Verify the exact joint order against the lifter's repository, since a wrong order produces plausible-looking garbage.

### 3.4 Temporal Buffer + Normalization

- Keep a rolling window of the last N frames of 2D keypoints (N = 27 to 81 is a reasonable real-time range; check what window the chosen checkpoint expects).
- Normalize 2D coordinates the way the lifter was trained (typically map the image to `[-1, 1]` by width/height, which depends on intrinsics/resolution). Use the same normalization at inference as in the model's training code.
- **Causal mode:** for live use, the window should end at the current frame (no future frames), or pad the start by repeating the first frame.

### 3.5 3D Lifter

- **Start with:** MotionBERT (best accuracy) or VideoPose3D (simplest, has a causal variant).
- Input: window of normalized 2D keypoints (+ confidence for MotionBERT).
- Output: 3D joints per frame, **root-relative**, in the camera's axis convention. Take the last frame's output for display.
- Run on GPU (CUDA/MPS) if available. Run it every frame or every second frame and interpolate if you are compute-limited.

### 3.6 Placement Fit (3D skeleton → image)

The lifter gives *shape*, not absolute position. For display you need to put the skeleton on the person in the image. Without calibration in this step, use a weak-perspective or approximate-pinhole fit:

1. Assume intrinsics: `fx = fy ≈ 1.0-1.2 × image_width`, principal point at the image center (good enough for visualization; replace with real calibration later).
2. Solve for scale `s` and translation `(tx, ty, tz)` that minimize 2D reprojection error between the projected 3D joints and the detected 2D joints (a few Gauss-Newton or `scipy.optimize.least_squares` iterations, warm-started from the previous frame).
3. Result: a metric-ish 3D skeleton in camera coordinates whose projection overlays the video correctly.

This is also the exact code you will reuse in later steps once real intrinsics and ground-plane constraints are available.

### 3.7 Renderer / UI

**Window layout**

```
┌────────────────────────┬────────────────────────┐
│   Laptop cam + overlay │   Phone cam + overlay  │
│                        │                        │
├────────────────────────┴────────────────────────┤
│  Status bar: FPS A / FPS B, latency, conn state │
└─────────────────────────────────────────────────┘
(optional extra panels: 3D viewport for each camera)
```

**Overlay conventions**
- Draw bones as lines, joints as circles.
- **Color by depth** (near = warm, far = cool) so the 3D nature is visible on a 2D video.
- Dim joints with low 2D confidence.
- Optional 3D viewport: a small panel plotting the root-relative skeleton from a free viewpoint (pyqtgraph/OpenGL or Open3D), which proves the 3D output is real and not just a flat redraw.

**UI toolkit**
- **Fastest to build:** OpenCV `imshow` on a horizontally concatenated frame.
- **Better:** PySide6 (Qt) with two video widgets and a pyqtgraph 3D widget. Do this once the pipeline works.

---

## 4. Threading and Data Flow

| Thread | Job | Rate |
|---|---|---|
| Capture A | read laptop camera | camera FPS |
| Capture B | read phone stream | stream FPS |
| Inference A | 2D pose → lift → fit | as fast as possible |
| Inference B | 2D pose → lift → fit | as fast as possible |
| Main/UI | draw latest frame + latest pose | ~30 Hz |

- Use Python `threading` plus queues of size 1 (latest wins). Use `multiprocessing` if the GIL becomes a bottleneck (ONNX/PyTorch inference mostly releases it, so threads are usually fine).
- A single GPU can serve both cameras; either two workers sharing models, or one worker alternating cameras. Alternating is simpler and avoids GPU contention.
- The UI draws the latest *frame* with the latest *available pose*. If the pose is slightly older than the frame, show it anyway but track the age (see below).

### Data model

```python
@dataclass
class FramePacket:
    cam_id: str            # "laptop" | "phone"
    frame_id: int
    t_capture: float       # laptop monotonic clock, seconds
    image: np.ndarray      # BGR

@dataclass
class PoseResult:
    cam_id: str
    frame_id: int
    t_capture: float
    kpts_2d: np.ndarray    # (17, 3) x, y, conf (pixels)
    kpts_3d_rel: np.ndarray  # (17, 3) root-relative (lifter output)
    kpts_3d_cam: np.ndarray  # (17, 3) camera coordinates (after placement fit)
    fit_error_px: float    # reprojection error, useful as a confidence signal
    t_infer_done: float
```

Keeping `cam_id`, `t_capture`, and `kpts_3d_cam` in the schema now means step 2 (sync, calibration, cross-camera matching) plugs in without refactoring.

---

## 5. Suggested Project Layout

```
dualpose/
├── main.py                  # starts threads, runs UI
├── config.yaml              # camera sources, resolution, model paths, window size
├── capture/
│   ├── base.py              # CaptureThread
│   ├── webcam.py            # laptop cam
│   └── network_stream.py    # RTSP / MJPEG / WebRTC receiver
├── pose/
│   ├── detector2d.py        # RTMPose wrapper
│   ├── keypoints.py         # COCO → H36M conversion, skeleton definitions
│   ├── lifter3d.py          # MotionBERT / VideoPose3D wrapper
│   └── placement.py         # weak-perspective / pinhole fit
├── ui/
│   ├── viewer.py            # window, panels
│   └── draw.py              # skeleton overlay, depth coloring
└── utils/
    ├── timing.py            # FPS counters, latency measurement
    └── types.py             # dataclasses above
```

---

## 6. Performance Budget (rough targets per camera frame)

| Stage | Target |
|---|---|
| Decode / capture | 5-15 ms (phone stream is the variable one) |
| 2D pose (RTMPose-m/s, GPU) | 5-15 ms |
| 3D lift (windowed) | 5-20 ms |
| Placement fit | 1-3 ms |
| Render | 5-10 ms |

If you are over budget: lower input resolution for the detector, use a smaller RTMPose variant, run the lifter every 2nd frame, or reduce the lifter window.

Measure and display **end-to-end latency** (`now - t_capture` at draw time) per camera from day one. It is the number that tells you whether the phone stream is acceptable.

---

## 7. Milestones

1. **Webcam only:** capture → RTMPose 2D overlay in an OpenCV window.
2. **Phone stream only:** same, but reading from the phone. Fix latency, orientation, and reconnects here.
3. **Both cameras:** side-by-side window with 2D overlays and FPS/latency readouts.
4. **Add the lifter:** root-relative 3D output, shown in a simple 3D plot to sanity-check shape (stand still, raise an arm, check it looks right).
5. **Placement fit:** reproject 3D back onto each video, color by depth.
6. **Polish:** Qt UI, 3D viewport, config file, logging recorded sessions to disk (useful for offline testing later).

**Strongly recommended:** add a "record raw frames + timestamps" mode early. Replaying recorded sessions makes debugging the model and later steps far faster than live testing.

---

## 8. Known Pitfalls

- **Phone orientation/mirroring:** streams may arrive rotated or mirrored. Add rotate/flip options in the config. A mirrored image swaps left/right joints, which silently breaks the lifter.
- **Different resolutions and aspect ratios** between the two cameras: normalize per camera and use per-camera intrinsic guesses.
- **Single-person assumption:** with several people in frame, the lifter's temporal buffer will mix identities. For now, track only the largest person and reset the buffer if the selected person jumps.
- **Buffer resets:** clear the temporal window after long detection dropouts, otherwise the lifter sees a discontinuous sequence.
- **Depth is relative in this step.** Absolute position from a single uncalibrated camera is approximate; don't judge it as metric until you add calibration.
- **Wi-Fi jitter:** latency will vary. Show it, and expect the phone view to lag the laptop view by tens to hundreds of milliseconds. Synchronization is a step-2 problem.
- **Lifter domain gap:** models trained on mocap can look wrong for unusual poses, camera angles (a high-mounted or tilted phone), or partial bodies. Frame the whole body in both cameras for the first tests.

---

## 9. Hooks for the Next Steps

| Later feature | Where it plugs in |
|---|---|
| Camera calibration | Replace guessed intrinsics in `placement.py`; add extrinsics between cameras |
| Time sync | Use `t_capture` plus an offset estimate (e.g. a clap/LED flash event) |
| Cross-camera skeleton matching | New module consuming `PoseResult` from both cameras (epipolar cost + Hungarian) |
| Multi-person | Add a tracker between the detector and the temporal buffer; one lifter buffer per track |
| Occlusion handling | Fall back to lifter + ground plane + stored bone lengths when one camera's detection drops |
| Triangulation | Fuse `kpts_2d` from both cameras once calibrated |

---

## 10. Recommended Starting Stack

- **Language:** Python 3.10+
- **Capture:** OpenCV (`cv2.VideoCapture`)
- **2D pose:** `rtmlib` (RTMPose, ONNX Runtime) for easy setup
- **3D lifting:** MotionBERT (accuracy) or VideoPose3D (simplicity, causal mode)
- **Optimization:** `scipy.optimize.least_squares`
- **UI:** OpenCV window first, then PySide6 + pyqtgraph
- **Phone link:** IP-camera app over RTSP/MJPEG first; WebRTC (`aiortc`) if you need better latency or iOS support

Check each library's current version and checkpoint availability before starting, since these projects change quickly.
