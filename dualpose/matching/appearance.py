from __future__ import annotations

import hashlib
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

OSNET_ONNX_URL = (
    "https://huggingface.co/kornia/osnet/resolve/main/"
    "osnet_x0_25_msmt17.onnx?download=true"
)
OSNET_ONNX_SHA256 = "e78604f4ccda49b8f41cd0f8f7303800ce75d2361895ebb0729513c1bf53d277"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec) + 1e-9
    return (vec / denom).astype(np.float32)


class AppearanceReID:
    def __init__(
        self,
        enabled: bool,
        model_path: str,
        download_if_missing: bool,
        interval_ms: float,
        min_box_area: int,
        threads: int,
        gallery_ttl_ms: float,
        exclude_head: bool,
    ) -> None:
        self.enabled = enabled
        self.model_path = Path(model_path)
        self.download_if_missing = download_if_missing
        self.interval_ms = interval_ms
        self.min_box_area = min_box_area
        self.threads = max(1, threads)
        self.gallery_ttl_ms = gallery_ttl_ms
        self.exclude_head = exclude_head

        self._session = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._last_run_ts: dict[str, float] = {}
        self._last_seen_ts: dict[str, float] = {}
        self._gallery: dict[str, deque[np.ndarray]] = {}
        self.status = "disabled"
        if self.enabled:
            self._init_session()

    def _init_session(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            self.status = "onnxruntime_not_available"
            self.enabled = False
            return

        if not self.model_path.exists():
            if not self.download_if_missing:
                self.status = "model_missing"
                self.enabled = False
                return
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(OSNET_ONNX_URL, str(self.model_path))

        digest = _sha256(self.model_path)
        if digest != OSNET_ONNX_SHA256:
            self.status = "model_checksum_mismatch"
            self.enabled = False
            return

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(self.model_path), sess_options=opts)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self.status = "ok"

    def _crop(self, image: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray | None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        area = (x2 - x1) * (y2 - y1)
        if area < self.min_box_area:
            return None
        return image[y1:y2, x1:x2]

    def _mask_head(self, crop_bgr: np.ndarray, keypoints: np.ndarray | None) -> np.ndarray:
        if not self.exclude_head:
            return crop_bgr
        h, w = crop_bgr.shape[:2]
        out = crop_bgr.copy()
        cutoff = int(round(h * 0.22))
        if keypoints is not None and keypoints.shape[0] >= 17:
            shoulders = keypoints[[11, 14], :2]
            ys = shoulders[:, 1]
            if np.all(np.isfinite(ys)):
                miny = float(np.min(ys))
                # Shoulders are in full frame; without crop origin we keep a conservative top mask.
                frac = np.clip(miny / max(float(h), 1.0), 0.18, 0.35)
                cutoff = int(round(h * frac))
        cutoff = max(1, min(cutoff, h - 1))
        out[:cutoff, :, :] = int(np.mean(out))
        return out

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        # OSNet uses RGB with ImageNet normalization.
        resized = cv2.resize(crop_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        chw = np.transpose(normalized, (2, 0, 1))
        return chw[None, ...].astype(np.float32)

    def _pooled_embedding(self, cam_id: str) -> np.ndarray | None:
        gallery = self._gallery.get(cam_id)
        if not gallery:
            return None
        mat = np.stack(list(gallery), axis=0)
        pooled = np.median(mat, axis=0)
        return _l2_normalize(pooled)

    def _prune_expired(self, now_ms: float) -> None:
        expired = []
        for cam_id, ts in self._last_seen_ts.items():
            if (now_ms - ts) > self.gallery_ttl_ms:
                expired.append(cam_id)
        for cam_id in expired:
            self._gallery.pop(cam_id, None)
            self._last_run_ts.pop(cam_id, None)
            self._last_seen_ts.pop(cam_id, None)

    def update(
        self,
        cam_id: str,
        image: np.ndarray,
        bbox_xyxy: np.ndarray | None,
        t_capture: float,
        keypoints: np.ndarray | None = None,
    ) -> np.ndarray | None:
        ts_ms = t_capture * 1000.0
        self._last_seen_ts[cam_id] = ts_ms
        self._prune_expired(ts_ms)
        pooled = self._pooled_embedding(cam_id)
        if not self.enabled or self._session is None:
            return pooled
        if bbox_xyxy is None:
            return pooled

        last = self._last_run_ts.get(cam_id, -1e12)
        if (ts_ms - last) < self.interval_ms:
            return pooled

        crop = self._crop(image, bbox_xyxy)
        if crop is None:
            return pooled
        crop = self._mask_head(crop, keypoints)

        inp = self._preprocess(crop)
        assert self._input_name is not None and self._output_name is not None
        out = self._session.run([self._output_name], {self._input_name: inp})[0]
        emb = _l2_normalize(np.asarray(out).reshape(-1).astype(np.float32))
        gal = self._gallery.setdefault(cam_id, deque(maxlen=5))
        gal.append(emb)
        self._last_run_ts[cam_id] = ts_ms
        return self._pooled_embedding(cam_id)

    def reset(self, cam_id: str | None = None) -> None:
        if cam_id is None:
            self._gallery.clear()
            self._last_run_ts.clear()
            self._last_seen_ts.clear()
            return
        self._gallery.pop(cam_id, None)
        self._last_run_ts.pop(cam_id, None)
        self._last_seen_ts.pop(cam_id, None)

    @staticmethod
    def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
        if a is None or b is None:
            return None
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
