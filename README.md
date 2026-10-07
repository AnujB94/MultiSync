# DualPose: Webcam + IP Camera 3D Pose Viewer

This project implements a low-latency step-1 dual-camera viewer:

- Camera A: laptop webcam
- Camera B: IP camera stream (RTSP)
- Independent 2D and monocular 3D pose pipeline per camera
- Single OpenCV window with video overlays, small 3D panels, and status metrics
- Optional cross-camera identity matching (`Person 1`) with **zero manual calibration**

The implementation is CPU-first and prioritizes latency over absolute frame rate.

## 1) Requirements

- Windows 10/11
- Python 3.10+
- One webcam and one IP camera app/stream URL

## 2) Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## 3) Configure cameras

Copy config:

```powershell
copy config.example.yaml config.local.yaml
```

Edit `config.local.yaml`:

- Set `phone.source` to your RTSP URL
- Keep `webcam.source` as an integer index (`0` by default)
- Tune `pipeline.infer_every_n_frames` and `pipeline.lifter_stride`
- For identity matching, enable/configure `matching` (manual calibration is optional)

## 4) Optional VideoPose3D checkpoint

The lifter wrapper expects a causal checkpoint by default:

- `checkpoints/cpn-ft-243-dbb-causal.bin`

If not present, the app falls back to a deterministic lightweight pseudo-3D depth
mode so the rest of the pipeline can still run.

## 5) Run

```powershell
python -m dualpose --config config.local.yaml
```

Hotkeys:

- `q`: quit
- `r`: toggle recording

## 6) Calibration-free identity matching

Default identity matching mode is calibration-free:

- background-feature auto-geometry (SIFT/ORB) when enough overlap/texture exists
- automatic fallback to appearance-only mode when geometry is unavailable or unreliable
- no face recognition (body appearance + pose proportions only)

Enable in config:

```yaml
matching:
  enabled: true
  geometry_mode: "auto"
  reid_enabled: true
  reid_download_if_missing: true
```

The app will label both detections as `Person 1` only after temporal confirmation.
If geometry/visibility/timing quality drops, state falls back to `unknown`.

Diagnostics in the status strip show whether matching runs in `geometry` or `appearance_only` mode.

## 7) Optional legacy checkerboard calibration

If you want stronger geometric gating for a fixed setup, you can still create and use a stereo calibration file:

```powershell
python calibrate.py --config config.local.yaml --output calibration/stereo_calibration.npz --cols 9 --rows 6 --square-size-m 0.025 --min-pairs 20
```

Then set:

```yaml
calibration:
  enabled: true
  path: "calibration/stereo_calibration.npz"

matching:
  geometry_mode: "calibrated"
```
## 8) CPU tuning tips

- Use `webcam.width/height = 640x360` and same for IP stream where possible
- Keep `pipeline.detector_mode = lightweight`
- Increase `pipeline.infer_every_n_frames` to 2 or 3 if FPS drops
- Increase `pipeline.lifter_stride` to 2 if lifter is heavy
- Keep `matching.reid_interval_ms` around `500` or higher on weaker CPUs
- Keep `matching.reid_threads = 1` to avoid thread contention with pose inference
- If auto-geometry is expensive on your CPU, switch `matching.auto_feature_type` to `orb`
- Increase `matching.auto_estimation_interval_ms` and `matching.auto_health_check_interval_ms`

## 9) Tests and lint

```powershell
python -m pytest -q
python -m ruff check .
```
