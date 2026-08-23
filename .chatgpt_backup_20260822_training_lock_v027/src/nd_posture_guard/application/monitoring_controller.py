from __future__ import annotations

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QMessageBox

from nd_posture_guard.alerts.alert_service import AlertService
from nd_posture_guard.application.monitoring_frame import MonitoringFrame
from nd_posture_guard.application.monitoring_worker import MonitoringWorker
from nd_posture_guard.config.settings_repository import SettingsRepository
from nd_posture_guard.ui.main_window import MainWindow


class MonitoringController(QObject):
    """Wire the main window to the worker while keeping persistence and alert side effects explicit."""

    def __init__(
        self,
        window: MainWindow,
        worker: MonitoringWorker,
        alert_service: AlertService,
        settings_repository: SettingsRepository,
    ) -> None:
        super().__init__()
        self._window = window
        self._worker = worker
        self._alert_service = alert_service
        self._settings_repository = settings_repository
        self._stopping = False

        self._window.add_good_training_requested.connect(self._worker.request_add_good_training)
        self._window.add_bad_training_requested.connect(self._worker.request_add_bad_training)
        self._window.training_point_selected.connect(self._worker.request_training_point)
        self._window.bad_score_threshold_changed.connect(self._change_bad_score_threshold)
        self._window.monitoring_pause_requested.connect(self._worker.request_monitoring_paused)
        self._window.review_training_requested.connect(self._worker.request_training_samples)
        self._window.delete_training_sample_requested.connect(self._worker.request_delete_training_sample)
        self._window.closing.connect(self.stop)

        self._worker.frame_ready.connect(self._show_frame)
        self._worker.status_changed.connect(self._window.set_status)
        self._worker.alert_requested.connect(self._trigger_alert)
        self._worker.camera_started.connect(self._camera_started)
        self._worker.training_started.connect(self._window.set_training_started)
        self._worker.training_points_required.connect(self._window.set_training_points_required)
        self._worker.training_recording_progress.connect(self._window.set_training_recording_progress)
        self._worker.training_sample_saved.connect(self._window.set_training_sample_saved)
        self._worker.training_failed.connect(self._window.set_training_failed)
        self._worker.dataset_stats_changed.connect(self._window.set_dataset_stats)
        self._worker.monitoring_paused_changed.connect(self._window.set_monitoring_paused)
        self._worker.training_samples_ready.connect(self._training_samples_ready)
        self._worker.fatal_error.connect(self._show_fatal_error)

    def start(self) -> None:
        """Start the background monitoring worker."""
        self._worker.start()

    @Slot(float)
    def _change_bad_score_threshold(self, value: float) -> None:
        """Persist a threshold change synchronously, then apply it to the worker."""
        try:
            self._settings_repository.save_bad_score_threshold(value)
        except Exception as exc:
            self._window.set_status(f"Could not save beep threshold: {exc}")
            return
        self._worker.request_bad_score_threshold(value)

    @Slot()
    def stop(self) -> None:
        """Request worker shutdown only once and wait briefly for camera release."""
        if self._stopping:
            return
        self._stopping = True
        self._worker.request_stop()
        if self._worker.isRunning():
            self._worker.wait(3000)

    @Slot(object)
    def _show_frame(self, result: MonitoringFrame) -> None:
        """Forward one throttled frame result to the main window."""
        self._window.show_frame(
            result.frame,
            result.posture_state,
            result.bad_score,
            result.confidence,
            result.warning_text,
            result.model_ready,
            result.geometry_points,
            result.tracking_confidence,
            result.training_active,
            result.training_stage_current,
            result.training_stage_total,
            result.training_message,
            result.monitoring_paused,
        )

    @Slot(object, bool)
    def _training_samples_ready(self, samples, open_dialog: bool) -> None:
        """Open review only for explicit requests; background refreshes never resurrect a closed dialog."""
        if open_dialog:
            self._window.show_training_samples(samples)
        else:
            self._window.refresh_training_samples(samples)

    @Slot()
    def _trigger_alert(self) -> None:
        """Play the warning sound through the alert service cooldown."""
        self._alert_service.trigger()

    @Slot(str)
    def _camera_started(self, device_label: str) -> None:
        """Report the selected physical camera device."""
        self._window.set_status(f"Camera active: {device_label}.")

    @Slot(str)
    def _show_fatal_error(self, message: str) -> None:
        """Surface a fatal monitoring error in both status text and a modal dialog."""
        self._window.set_status(f"Monitoring stopped: {message}")
        QMessageBox.critical(self._window, "Monitoring error", message)
