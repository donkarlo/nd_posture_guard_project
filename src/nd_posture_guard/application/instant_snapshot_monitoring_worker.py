from __future__ import annotations

import numpy as np

from nd_posture_guard.application.isolated_monitoring_worker import IsolatedMonitoringWorker


class InstantSnapshotMonitoringWorker(IsolatedMonitoringWorker):
    """Start training from the latest live frame instead of waiting for a future frame."""

    @staticmethod
    def _frame_is_usable(frame) -> bool:
        if frame is None or getattr(frame, "size", 0) <= 0:
            return False
        percentile_99 = float(np.percentile(frame, 99.0))
        standard_deviation = float(np.std(frame))
        return percentile_99 >= 6.0 or standard_deviation >= 3.0

    def _start_training(self, label: int) -> None:
        if self._training_session.active:
            self.status_changed.emit("A training sample is already being collected.")
            return

        frame = self._camera.read()
        if not self._frame_is_usable(frame):
            self.status_changed.emit(
                "Training was not started because the latest camera frame is unavailable or black. "
                "The live monitor remains active; try again when the camera image is visible."
            )
            return

        self._paused_before_training = self._monitoring_paused
        self._training_session.start(label)
        self._geometry_tracker.reset()
        self._snapshot_requested = False
        self._frozen_frame = frame.copy()
        self._training_session.begin_point_selection(self._frozen_frame)

        label_name = self._training_session.label_name
        self.training_started.emit(label_name)
        self._training_message = self._training_session.next_point_message
        self.training_points_required.emit(
            self._training_message,
            0,
            self._training_session.POINTS_PER_SAMPLE,
        )
        self.status_changed.emit(
            f"Adding a {label_name} example. The currently visible live frame was frozen immediately. "
            "Define exactly three shapes: face triangle, left-side-of-image shoulder line and "
            "right-side-of-image shoulder line."
        )
