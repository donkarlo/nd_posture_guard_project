from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from nd_posture_guard.alerts.alert_service import AlertService
from nd_posture_guard.alerts.alert_sound_player import AlertSoundPlayer
from nd_posture_guard.application.isolated_monitoring_worker import IsolatedMonitoringWorker
from nd_posture_guard.application.monitoring_controller import MonitoringController
from nd_posture_guard.camera.ffmpeg_camera_device_finder import FfmpegCameraDeviceFinder
from nd_posture_guard.camera.ffmpeg_camera_reader import FfmpegCameraReader
from nd_posture_guard.config.settings_repository import SettingsRepository
from nd_posture_guard.data.serialized_video_training_dataset_repository import (
    SerializedVideoTrainingDatasetRepository,
)
from nd_posture_guard.desktop.desktop_integration_installer import DesktopIntegrationInstaller
from nd_posture_guard.model.example_posture_classifier import ExamplePostureClassifier
from nd_posture_guard.training.example_training_session import ExampleTrainingSession
from nd_posture_guard.ui.isolated_main_window import IsolatedMainWindow
from nd_posture_guard.vision.fast_posture_geometry_tracker import FastPostureGeometryTracker
from nd_posture_guard.vision.opencv_compatibility_validator import OpenCvCompatibilityValidator
from nd_posture_guard.vision.posture_geometry_feature_extractor import PostureGeometryFeatureExtractor


class FfmpegCapturePostureGuardApplication:
    """Run posture monitoring with camera capture isolated in an ffmpeg process."""

    def __init__(self, settings_path: Path) -> None:
        project_root = settings_path.parent
        desktop_integration = DesktopIntegrationInstaller(project_root)
        desktop_integration.install()

        qt_arguments = list(sys.argv)
        if sys.platform.startswith("linux") and "-name" not in qt_arguments:
            qt_arguments.extend(("-name", desktop_integration.STARTUP_WM_CLASS))

        self._qt_app = QApplication(qt_arguments)
        self._qt_app.setApplicationName(desktop_integration.DESKTOP_ID)
        self._qt_app.setApplicationDisplayName(desktop_integration.DISPLAY_NAME)
        self._qt_app.setDesktopFileName(desktop_integration.DESKTOP_ID)
        if desktop_integration.project_icon_path.is_file():
            self._qt_app.setWindowIcon(QIcon(str(desktop_integration.project_icon_path)))

        settings_repository = SettingsRepository(settings_path)
        settings = settings_repository.load()
        self._window = IsolatedMainWindow(
            settings.title,
            settings.classifier_bad_score_threshold,
            settings.training_data_root,
        )
        if desktop_integration.project_icon_path.is_file():
            self._window.setWindowIcon(QIcon(str(desktop_integration.project_icon_path)))

        OpenCvCompatibilityValidator().validate()
        camera_device = FfmpegCameraDeviceFinder().find(settings.camera_device)

        feature_extractor = PostureGeometryFeatureExtractor()
        geometry_tracker = FastPostureGeometryTracker()
        dataset_repository = SerializedVideoTrainingDatasetRepository(
            root=Path(settings.training_data_root),
            novelty_multiplier=settings.training_novelty_multiplier,
            recording_fps=1000.0 / max(settings.detection_interval_ms, 1),
        )
        classifier = ExamplePostureClassifier(
            bad_score_threshold=settings.classifier_bad_score_threshold,
            required_bad_frames=settings.required_bad_frames,
            k_neighbors=settings.classifier_k_neighbors,
        )
        worker = IsolatedMonitoringWorker(
            camera=FfmpegCameraReader(
                camera_device,
                settings.camera_width,
                settings.camera_height,
                settings.camera_fps,
                settings.mirror_camera,
            ),
            feature_extractor=feature_extractor,
            geometry_tracker=geometry_tracker,
            classifier=classifier,
            training_session=ExampleTrainingSession(
                frames_per_sample=settings.training_frames_per_sample,
            ),
            dataset_repository=dataset_repository,
            detection_interval_ms=settings.detection_interval_ms,
            warning_text=settings.warning_text,
            camera_width=settings.camera_width,
            camera_height=settings.camera_height,
        )

        alert_service = AlertService(
            settings.alert_cooldown_seconds,
            AlertSoundPlayer(
                project_root / settings.alert_sound_path,
                settings.alert_volume_percent,
            ),
        )
        self._controller = MonitoringController(
            self._window,
            worker,
            alert_service,
            settings_repository,
        )
        self._qt_app.aboutToQuit.connect(self._controller.stop)

    def run(self) -> int:
        self._window.show()
        self._controller.start()
        return self._qt_app.exec()
