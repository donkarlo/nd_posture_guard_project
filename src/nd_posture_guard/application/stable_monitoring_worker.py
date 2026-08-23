from __future__ import annotations

from queue import Empty
from threading import Thread
import time

import numpy as np

from nd_posture_guard.application.monitoring_frame import MonitoringFrame
from nd_posture_guard.application.monitoring_worker import MonitoringWorker
from nd_posture_guard.model.posture_geometry import PostureGeometry


class StableMonitoringWorker(MonitoringWorker):
    """Monitoring loop with non-flooding UI updates and tracking independent of classification."""

    PROGRESS_EMIT_EVERY_FRAMES = 5

    def _process_frame(self) -> None:
        frame = self._camera.read()
        if frame is None:
            return

        if self._snapshot_requested and self._training_session.waiting_for_frame:
            self._frozen_frame = frame.copy()
            self._snapshot_requested = False
            self._training_session.begin_point_selection(self._frozen_frame)
            self._training_message = self._training_session.next_point_message
            self.training_points_required.emit(
                self._training_message, 0, self._training_session.POINTS_PER_SAMPLE
            )

        if self._training_session.recording:
            finished = self._training_session.add_frame(frame)
            current = self._training_session.recording_count
            if (
                finished
                or current == 1
                or current % self.PROGRESS_EMIT_EVERY_FRAMES == 0
            ):
                self.training_recording_progress.emit(
                    current, self._training_session.frames_per_sample, self._training_message
                )
            if finished:
                self._finish_training_sample()

        display_frame = self._frozen_frame if self._frozen_frame is not None else frame
        posture_state = "training" if self._training_session.active else "not_trained"
        bad_score = 0.0
        confidence = 0.0
        nearest_distance = 0.0
        warning_text: str | None = None
        geometry: PostureGeometry | None = None

        if not self._training_session.active:
            geometry = self._geometry_tracker.track(frame)

            if self._monitoring_paused:
                posture_state = "paused"
            elif not self._classifier.ready:
                posture_state = "not_trained"
            elif geometry is None or geometry.tracking_confidence < self.MIN_TRACKING_CONFIDENCE:
                posture_state = "unknown"
            else:
                try:
                    feature = self._feature_extractor.extract(
                        geometry, frame.shape[1], frame.shape[0]
                    )
                    result = self._classifier.classify(feature)
                    posture_state = result.state
                    bad_score = result.bad_score
                    confidence = result.confidence
                    nearest_distance = result.nearest_distance
                    if result.should_alert:
                        warning_text = self._warning_text
                        self.alert_requested.emit()
                except ValueError:
                    posture_state = "unknown"

        now = time.monotonic()
        if now < self._next_ui_emit:
            return
        if self._next_ui_emit <= 0.0:
            self._next_ui_emit = now + self._display_interval
        else:
            self._next_ui_emit += self._display_interval
            if self._next_ui_emit < now - self._display_interval:
                self._next_ui_emit = now + self._display_interval

        self.frame_ready.emit(
            MonitoringFrame(
                frame=display_frame,
                posture_state=posture_state,
                bad_score=bad_score,
                confidence=confidence,
                nearest_distance=nearest_distance,
                warning_text=warning_text,
                model_ready=self._classifier.ready,
                geometry_points=geometry.points if geometry is not None else None,
                tracking_confidence=(
                    geometry.tracking_confidence if geometry is not None else 0.0
                ),
                training_active=self._training_session.active,
                training_stage_current=(
                    len(self._training_session.current_points)
                    if self._training_session.active
                    else 0
                ),
                training_stage_total=(
                    self._training_session.POINTS_PER_SAMPLE
                    if self._training_session.active
                    else 0
                ),
                training_message=self._training_message,
                monitoring_paused=self._monitoring_paused,
            )
        )

    def _reload_profile(self) -> None:
        """Build tracking with one class; enable classification only when both exist."""
        self._profile_reload_generation += 1
        generation = self._profile_reload_generation

        if self._known_good < 1 and self._known_bad < 1:
            self._classifier.clear()
            self._geometry_tracker.set_profile(None)
            return

        Thread(
            target=self._build_profile_background,
            args=(generation,),
            name=f"posture-profile-{generation}",
            daemon=True,
        ).start()

    def _apply_profile_results(self) -> None:
        while True:
            try:
                generation, profile, error = self._profile_results.get_nowait()
            except Empty:
                return

            if generation != self._profile_reload_generation:
                continue
            if error is not None:
                if not self._training_session.active:
                    self.status_changed.emit(f"Could not rebuild posture model: {error}")
                return

            if profile is None:
                self._classifier.clear()
                self._geometry_tracker.set_profile(None)
            else:
                self._geometry_tracker.set_profile(profile)
                labels = set(int(value) for value in np.unique(profile.labels))
                if labels == {0, 1}:
                    self._classifier.set_profile(profile)
                else:
                    self._classifier.clear()

            self.dataset_stats_changed.emit(
                self._known_good, self._known_bad, self._classifier.ready
            )
            if not self._training_session.active:
                if self._classifier.ready:
                    self.status_changed.emit(
                        f"Three-shape geometry model ready: "
                        f"{self._known_good} GOOD / {self._known_bad} BAD usable samples."
                    )
                elif profile is not None:
                    self.status_changed.emit(
                        f"Landmark tracking ready from {self._known_good} GOOD / "
                        f"{self._known_bad} BAD samples. Add the missing class to enable GOOD/BAD decisions."
                    )
                else:
                    self.status_changed.emit(
                        "No usable geometry profile could be built from the saved samples."
                    )
