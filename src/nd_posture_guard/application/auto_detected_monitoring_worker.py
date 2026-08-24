from __future__ import annotations

from nd_posture_guard.application.instant_snapshot_monitoring_worker import (
    InstantSnapshotMonitoringWorker,
)


class AutoDetectedMonitoringWorker(InstantSnapshotMonitoringWorker):
    """Start automatic GOOD/BAD capture without freezing or waiting for clicks."""

    def _start_training(self, label: int) -> None:
        if self._training_session.active:
            self.status_changed.emit("A training sample is already being collected.")
            return

        self._paused_before_training = self._monitoring_paused
        self._training_session.start(label)
        self._geometry_tracker.reset()
        self._snapshot_requested = False
        self._frozen_frame = None

        label_name = self._training_session.label_name
        self.training_started.emit(label_name)
        self._training_message = self._training_session.next_point_message
        self.training_recording_progress.emit(
            0,
            self._training_session.frames_per_sample,
            self._training_message,
        )
        self.status_changed.emit(
            f"Adding a {label_name} example automatically. Hold the intended posture; "
            "fresh face/pose landmarks are detected on every accepted frame."
        )
