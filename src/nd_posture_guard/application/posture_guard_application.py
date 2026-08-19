from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from nd_posture_guard.alerts.alert_service import AlertService
from nd_posture_guard.alerts.alert_sound_player import AlertSoundPlayer
from nd_posture_guard.application.monitoring_controller import MonitoringController
from nd_posture_guard.application.monitoring_worker import MonitoringWorker
from nd_posture_guard.camera.camera_device_finder import CameraDeviceFinder
from nd_posture_guard.camera.camera_reader import CameraReader
from nd_posture_guard.config.settings_repository import SettingsRepository
from nd_posture_guard.posture.posture_evaluator import PostureEvaluator
from nd_posture_guard.posture.shoulder_calibration_session import ShoulderCalibrationSession
from nd_posture_guard.ui.main_window import MainWindow
from nd_posture_guard.vision.opencv_compatibility_validator import OpenCvCompatibilityValidator
from nd_posture_guard.vision.shoulder_anchor_tracker import ShoulderAnchorTracker


class PostureGuardApplication:
    def __init__(self, settings_path: Path) -> None:
        self._qt_app = QApplication(sys.argv)
        settings_repository = SettingsRepository(settings_path)
        settings = settings_repository.load()
        self._window = MainWindow(settings.title, settings.shoulder_drop_trigger_percent)

        OpenCvCompatibilityValidator().validate()
        project_root = settings_path.parent
        camera_device = CameraDeviceFinder(
            width=settings.camera_width,
            height=settings.camera_height,
            fps=settings.camera_fps,
        ).find(settings.camera_device)

        shoulder_tracker = ShoulderAnchorTracker(
            template_size_px=settings.shoulder_anchor_template_size_px,
            search_radius_px=settings.shoulder_anchor_search_radius_px,
            minimum_match_confidence=settings.shoulder_anchor_minimum_match_confidence,
            minimum_valid_anchors=settings.shoulder_anchor_minimum_valid_anchors,
            maximum_group_motion_residual_px=settings.shoulder_anchor_maximum_group_motion_residual_px,
        )

        worker = MonitoringWorker(
            camera=CameraReader(
                camera_device,
                settings.camera_width,
                settings.camera_height,
                settings.camera_fps,
                settings.mirror_camera,
            ),
            shoulder_tracker=shoulder_tracker,
            posture_evaluator=PostureEvaluator(
                shoulder_drop_trigger_percent=settings.shoulder_drop_trigger_percent,
                shoulder_width_trigger_percent=settings.shoulder_width_trigger_percent,
                shoulder_tilt_trigger_degrees=settings.shoulder_tilt_trigger_degrees,
                required_bad_frames=settings.required_bad_frames,
            ),
            calibration_session=ShoulderCalibrationSession(),
            settings_repository=settings_repository,
            detection_interval_ms=settings.detection_interval_ms,
            warning_text=settings.warning_text,
        )

        alert_sound = project_root / settings.alert_sound_path
        alert_service = AlertService(
            settings.alert_cooldown_seconds,
            AlertSoundPlayer(alert_sound, settings.alert_volume_percent),
        )
        self._controller = MonitoringController(self._window, worker, alert_service)
        self._qt_app.aboutToQuit.connect(self._controller.stop)

    def run(self) -> int:
        self._window.show()
        self._controller.start()
        return self._qt_app.exec()
