# DualPose: Webcam + IP Camera 3D Pose Viewer

This project implements a low-latency step-1 dual-camera viewer:

- Camera A: laptop webcam
- Camera B: IP camera stream (RTSP)
- Independent 2D and monocular 3D pose pipeline per camera
- Single OpenCV window with video overlays, small 3D panels, and status metrics

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

## 6) CPU tuning tips

- Use `webcam.width/height = 640x360` and same for IP stream where possible
- Keep `pipeline.detector_mode = lightweight`
- Increase `pipeline.infer_every_n_frames` to 2 or 3 if FPS drops
- Increase `pipeline.lifter_stride` to 2 if lifter is heavy

## 7) Tests and lint

```powershell
python -m pytest -q
python -m ruff check .
```
