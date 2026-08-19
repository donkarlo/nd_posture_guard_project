from __future__ import annotations

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QMessageBox

from nd_posture_guard.alerts.alert_service import AlertService
from nd_posture_guard.application.monitoring_frame import MonitoringFrame
from nd_posture_guard.application.monitoring_worker import MonitoringWorker
from nd_posture_guard.ui.main_window import MainWindow


class MonitoringController(QObject):
    def __init__(self, window: MainWindow, worker: MonitoringWorker, alert_service: AlertService) -> None:
        super().__init__()
        self._window = window
        self._worker = worker
        self._alert_service = alert_service
        self._stopping = False

        self._window.calibrate_requested.connect(self._worker.request_calibration)
        self._window.calibration_point_selected.connect(self._worker.request_calibration_point)
        self._window.clear_reference_requested.connect(self._worker.request_clear_reference)
        self._window.shoulder_drop_changed.connect(self._worker.request_shoulder_drop_percent)
        self._window.monitoring_pause_requested.connect(self._worker.request_monitoring_paused)
        self._window.closing.connect(self.stop)

        self._worker.frame_ready.connect(self._show_frame)
        self._worker.status_changed.connect(self._window.set_status)
        self._worker.alert_requested.connect(self._trigger_alert)
        self._worker.camera_started.connect(self._camera_started)
        self._worker.calibration_started.connect(self._window.set_calibration_started)
        self._worker.calibration_progress.connect(self._window.set_calibration_progress)
        self._worker.calibration_points_required.connect(self._window.set_calibration_points_required)
        self._worker.calibration_action_required.connect(self._window.set_calibration_action_required)
        self._worker.calibration_completed.connect(self._window.set_calibration_complete)
        self._worker.calibration_failed.connect(self._window.set_calibration_failed)
        self._worker.calibration_cleared.connect(self._window.set_calibration_cleared)
        self._worker.monitoring_paused_changed.connect(self._window.set_monitoring_paused)
        self._worker.fatal_error.connect(self._show_fatal_error)

    def start(self) -> None:
        self._worker.start()

    @Slot()
    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self._worker.request_stop()
        if self._worker.isRunning():
            self._worker.wait(3000)

    @Slot(object)
    def _show_frame(self, result: MonitoringFrame) -> None:
        self._window.show_frame(
            result.frame,
            result.shoulders,
            result.reference_shoulders,
            result.posture_state,
            result.shoulder_drop_percent,
            result.shoulder_width_change_percent,
            result.shoulder_tilt_change_degrees,
            result.warning_text,
            result.tracker_ready,
            result.shoulders_detected,
            result.tracker_confidence,
            result.calibration_active,
            result.calibration_current,
            result.calibration_target,
            result.calibration_message,
            result.monitoring_paused,
        )

    @Slot()
    def _trigger_alert(self) -> None:
        self._alert_service.trigger()

    @Slot(str)
    def _camera_started(self, device_label: str) -> None:
        self._window.set_status(
            f"Camera active: {device_label}. This version uses no face/head/pose model. "
            "Press Calibrate shoulders and mark 6 shoulder-line points in each of three guided straight poses (18 total)."
        )

    @Slot(str)
    def _show_fatal_error(self, message: str) -> None:
        self._window.set_status(f"Monitoring stopped: {message}")
        self._window.set_calibration_failed(message)
        QMessageBox.critical(self._window, "Monitoring error", message)
