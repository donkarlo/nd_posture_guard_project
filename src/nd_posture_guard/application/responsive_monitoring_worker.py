from __future__ import annotations

import time

from nd_posture_guard.application.monitoring_frame import MonitoringFrame
from nd_posture_guard.application.monitoring_worker import MonitoringWorker
from nd_posture_guard.model.posture_geometry import PostureGeometry


class ResponsiveMonitoringWorker(MonitoringWorker):
    """Monitoring worker that keeps Qt camera-frame delivery bounded during training too."""

    def _process_frame(self) -> None:
        """Process one camera frame while never flooding the Qt event queue."""
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

        if self._classifier.ready and not self._training_session.active:
            geometry = self._geometry_tracker.track(frame)
            if self._monitoring_paused:
                posture_state = "paused"
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

        # Point-selection/progress signals are already immediate. Camera images are
        # intentionally bounded to MAX_UI_FPS even while training is active so Qt
        # cannot accumulate hundreds of expensive cvtColor/QImage.copy updates.
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
