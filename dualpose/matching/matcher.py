from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass

import numpy as np

from dualpose.calibration import StereoCalibration
from dualpose.config import MatchingConfig
from dualpose.matching.appearance import AppearanceReID
from dualpose.matching.body_descriptors import hist_similarity, proportion_similarity
from dualpose.matching.geometry import EpipolarScorer, GeometryModel
from dualpose.matching.synchronizer import TimestampPairer
from dualpose.utils.types import IdentityMatch, IdentityState, MatchEvidence, PersonObservation


@dataclass
class _Counters:
    same_hits: int = 0
    diff_hits: int = 0
    invalid_hits: int = 0
    ema: float = 0.0
    has_ema: bool = False


class IdentityMatcher:
    def __init__(
        self,
        cam_a_id: str,
        cam_b_id: str,
        config: MatchingConfig,
        calibration: StereoCalibration | None,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.enabled = config.enabled
        self.pairer = TimestampPairer(
            cam_a_id=cam_a_id,
            cam_b_id=cam_b_id,
            max_pose_age_ms=config.max_pose_age_ms,
            max_pair_skew_ms=config.max_pair_skew_ms,
            time_offset_s=0.0 if calibration is None else calibration.time_offset_s,
        )
        self.geometry = EpipolarScorer(
            calibration=calibration,
            min_common_joints=config.min_common_joints,
            sigma_px=config.epipolar_sigma_px,
            hard_reject_px=config.epipolar_hard_reject_px,
        )
        self._calibrated_model = EpipolarScorer.calibrated_model(calibration)
        self._auto_model: GeometryModel | None = None
        self._auto_geometry_state = "initializing"
        self._auto_geometry_reason = "startup"
        self.appearance = AppearanceReID(
            enabled=config.reid_enabled and config.enabled,
            model_path=config.reid_model_path,
            download_if_missing=config.reid_download_if_missing,
            interval_ms=config.reid_interval_ms,
            min_box_area=config.reid_min_box_area,
            threads=config.reid_threads,
            gallery_ttl_ms=config.reid_gallery_ttl_ms,
            exclude_head=config.reid_exclude_head,
        )
        self._lock = threading.Lock()
        self._state = IdentityMatch(
            state=IdentityState.UNKNOWN,
            shared_id=None,
            evidence=None,
            updated_at=time.monotonic(),
        )
        self._last_pair_key: tuple[int, int] | None = None
        self._last_pair: tuple[PersonObservation, PersonObservation] | None = None
        self._counters = _Counters()
        self._history_geometry: collections.deque[int] = collections.deque(
            maxlen=max(1, config.confirm_window)
        )
        self._history_appearance: collections.deque[int] = collections.deque(
            maxlen=max(1, config.appearance_only_confirm_window)
        )

    def disable_calibrated_geometry(self) -> None:
        with self._lock:
            self._calibrated_model = None

    def update_auto_geometry(
        self,
        model: GeometryModel | None,
        state: str,
        reason: str,
    ) -> None:
        with self._lock:
            self._auto_model = model
            self._auto_geometry_state = state
            self._auto_geometry_reason = reason

    def _active_geometry_model(self) -> GeometryModel | None:
        mode = self.config.geometry_mode
        if mode == "off":
            return None
        if mode == "calibrated":
            return self._calibrated_model
        # auto mode
        if self._auto_model is not None and self._auto_geometry_state == "healthy":
            if self._auto_model.age_ms <= self.config.auto_model_expiry_ms:
                return self._auto_model
        return None

    def _motion_score(self, obs_a: PersonObservation, obs_b: PersonObservation) -> float | None:
        if self.config.weight_motion <= 0.0 or self._last_pair is None:
            return None
        prev_a, prev_b = self._last_pair
        da = obs_a.kpts_2d[:, :2] - prev_a.kpts_2d[:, :2]
        db = obs_b.kpts_2d[:, :2] - prev_b.kpts_2d[:, :2]
        ma = np.linalg.norm(da, axis=1)
        mb = np.linalg.norm(db, axis=1)
        na = np.linalg.norm(ma)
        nb = np.linalg.norm(mb)
        if na < 1e-6 or nb < 1e-6:
            return None
        rho = float(np.dot(ma, mb) / (na * nb))
        return float(np.clip((rho + 1.0) * 0.5, 0.0, 1.0))

    def _appearance_components(
        self,
        obs_a: PersonObservation,
        obs_b: PersonObservation,
    ) -> tuple[float | None, float | None, float | None]:
        reid = AppearanceReID.cosine_similarity(
            obs_a.appearance_embedding,
            obs_b.appearance_embedding,
        )
        if reid is not None:
            reid = float(np.clip((reid + 1.0) * 0.5, 0.0, 1.0))
        upper = hist_similarity(obs_a.upper_hist, obs_b.upper_hist)
        lower = hist_similarity(obs_a.lower_hist, obs_b.lower_hist)
        if upper is not None and lower is not None:
            hist = 0.5 * (upper + lower)
        else:
            hist = upper if upper is not None else lower
        prop = proportion_similarity(obs_a.body_proportions, obs_b.body_proportions)
        return reid, hist, prop

    def _appearance_score(
        self,
        reid: float | None,
        hist: float | None,
        prop: float | None,
    ) -> tuple[float | None, int]:
        parts: list[tuple[float, float]] = []
        if reid is not None:
            parts.append((reid, self.config.appearance_weight_reid))
        if hist is not None:
            parts.append((hist, self.config.appearance_weight_hist))
        if prop is not None:
            parts.append((prop, self.config.appearance_weight_proportions))
        if not parts:
            return None, 0
        total_w = sum(w for _, w in parts)
        if total_w <= 0:
            return None, len(parts)
        return float(sum(v * w for v, w in parts) / total_w), len(parts)

    def _fuse_score(
        self,
        geometry_score: float | None,
        appearance_score: float | None,
        motion_score: float | None,
    ) -> float | None:
        parts: list[tuple[float, float]] = []
        if geometry_score is not None:
            parts.append((geometry_score, self.config.weight_geometry))
        if appearance_score is not None:
            parts.append((appearance_score, self.config.weight_appearance))
        if motion_score is not None:
            parts.append((motion_score, self.config.weight_motion))
        if not parts:
            return None
        total_w = sum(w for _, w in parts)
        if total_w <= 0:
            return None
        return float(sum(v * w for v, w in parts) / total_w)

    def _set_state(self, state: IdentityState, evidence: MatchEvidence) -> None:
        self._state = IdentityMatch(
            state=state,
            shared_id="Person 1" if state == IdentityState.SAME else None,
            evidence=evidence,
            updated_at=time.monotonic(),
        )

    def _history_counts(self, history: collections.deque[int]) -> tuple[int, int]:
        pos = sum(1 for x in history if x > 0)
        neg = sum(1 for x in history if x < 0)
        return pos, neg

    def _advance_state_from_history(
        self,
        history: collections.deque[int],
        confirm_required: int,
        release_required: int,
        evidence: MatchEvidence,
    ) -> None:
        pos, neg = self._history_counts(history)
        if pos >= confirm_required:
            self._set_state(IdentityState.SAME, evidence)
            return
        if neg >= release_required:
            if self._state.state == IdentityState.SAME:
                self._set_state(IdentityState.UNKNOWN, evidence)
            else:
                self._set_state(IdentityState.DIFFERENT, evidence)
            return
        self._set_state(self._state.state, evidence)

    def submit(self, obs: PersonObservation) -> IdentityMatch:
        if not self.enabled:
            return self._state

        now_ts = time.monotonic()
        with self._lock:
            paired = self.pairer.submit(obs=obs, now_ts=now_ts)
            if paired is None:
                return self._state
            obs_a, obs_b, skew_ms = paired
            pair_key = (obs_a.frame_id, obs_b.frame_id)
            if pair_key == self._last_pair_key:
                return self._state
            self._last_pair_key = pair_key

            geometry_model = self._active_geometry_model()
            mode = "geometry" if geometry_model is not None else "appearance_only"
            geom_state = (
                self._auto_geometry_state
                if self.config.geometry_mode == "auto"
                else ("healthy" if geometry_model is not None else "none")
            )

            if not obs_a.detection_fresh or not obs_b.detection_fresh:
                if not obs_a.detection_fresh:
                    self.appearance.reset(obs_a.cam_id)
                if not obs_b.detection_fresh:
                    self.appearance.reset(obs_b.cam_id)
                evidence = MatchEvidence(
                    frame_id_a=obs_a.frame_id,
                    frame_id_b=obs_b.frame_id,
                    skew_ms=skew_ms,
                    usable_joints=0,
                    epipolar_error_px=None,
                    appearance_score=None,
                    motion_score=None,
                    fused_score=None,
                    valid=False,
                    reason="stale_detection",
                    geometry_source="none" if geometry_model is None else geometry_model.source,
                    geometry_state=geom_state,
                    geometry_matches=0,
                    geometry_inliers=0,
                    geometry_inlier_ratio=0.0,
                    geometry_model_age_ms=None if geometry_model is None else geometry_model.age_ms,
                    mode=mode,
                )
                self._counters.invalid_hits += 1
                if (
                    self._state.state == IdentityState.SAME
                    and self._counters.invalid_hits >= self.config.release_required
                ):
                    self._set_state(IdentityState.UNKNOWN, evidence)
                else:
                    self._set_state(self._state.state, evidence)
                return self._state

            reid_score, hist_score, prop_score = self._appearance_components(obs_a, obs_b)
            a_score, n_app_cues = self._appearance_score(reid_score, hist_score, prop_score)
            g_score, g_error, common_joints, hard_fail, reason = self.geometry.score(
                obs_a,
                obs_b,
                geometry_model,
            )
            m_score = self._motion_score(obs_a, obs_b)

            if mode == "geometry":
                if hard_fail:
                    fused = None
                    valid = False
                    signal = -1
                else:
                    fused = self._fuse_score(g_score, a_score, m_score)
                    valid = fused is not None
                    if not valid or fused is None:
                        signal = 0
                    elif fused >= self.config.same_threshold:
                        signal = 1
                    elif fused <= self.config.different_threshold:
                        signal = -1
                    else:
                        signal = 0
            else:
                fused = a_score
                valid = fused is not None and n_app_cues >= 2
                if not valid or fused is None:
                    signal = 0
                elif fused >= self.config.appearance_only_same_threshold:
                    signal = 1
                elif fused <= self.config.appearance_only_different_threshold:
                    signal = -1
                else:
                    signal = 0

            evidence = MatchEvidence(
                frame_id_a=obs_a.frame_id,
                frame_id_b=obs_b.frame_id,
                skew_ms=skew_ms,
                usable_joints=common_joints,
                epipolar_error_px=g_error,
                appearance_score=a_score,
                motion_score=m_score,
                fused_score=fused,
                valid=valid,
                reason=reason if hard_fail else ("ok" if valid else "invalid_score"),
                geometry_source="none" if geometry_model is None else geometry_model.source,
                geometry_state=geom_state,
                geometry_matches=0 if geometry_model is None else geometry_model.matches,
                geometry_inliers=0 if geometry_model is None else geometry_model.inliers,
                geometry_inlier_ratio=(
                    0.0 if geometry_model is None else geometry_model.inlier_ratio
                ),
                geometry_model_age_ms=None if geometry_model is None else geometry_model.age_ms,
                mode=mode,
            )

            if not valid:
                self._counters.invalid_hits += 1
                if (
                    self._state.state == IdentityState.SAME
                    and self._counters.invalid_hits
                    >= (
                        self.config.release_required
                        if mode == "geometry"
                        else self.config.appearance_only_release_required
                    )
                ):
                    self._set_state(IdentityState.UNKNOWN, evidence)
                else:
                    self._set_state(self._state.state, evidence)
                self._last_pair = (obs_a, obs_b)
                return self._state

            self._counters.invalid_hits = 0
            if mode == "geometry":
                history = self._history_geometry
                history.append(signal)
                self._advance_state_from_history(
                    history=history,
                    confirm_required=self.config.confirm_required,
                    release_required=self.config.release_required,
                    evidence=evidence,
                )
            else:
                history = self._history_appearance
                history.append(signal)
                self._advance_state_from_history(
                    history=history,
                    confirm_required=self.config.appearance_only_confirm_required,
                    release_required=self.config.appearance_only_release_required,
                    evidence=evidence,
                )

            self._last_pair = (obs_a, obs_b)
            return self._state

    def get_state(self) -> IdentityMatch:
        with self._lock:
            return self._state
