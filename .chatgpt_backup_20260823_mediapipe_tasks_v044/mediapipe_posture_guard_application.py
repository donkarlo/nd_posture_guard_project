from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from nd_posture_guard.alerts.alert_service import AlertService
from nd_posture_guard.alerts.alert_sound_player import AlertSoundPlayer
from nd_posture_guard.application.auto_detected_monitoring_worker import (
    AutoDetectedMonitoringWorker,
)
from nd_posture_guard.application.monitoring_controller import MonitoringController
from nd_posture_guard.camera.cached_usable_camera_reader import CachedUsableCameraReader
from nd_posture_guard.camera.stable_camera_device_finder import StableCameraDeviceFinder
from nd_posture_guard.camera.stable_camera_reader import StableCameraReader
from nd_posture_guard.config.settings_repository import SettingsRepository
from nd_posture_guard.data.mediapipe_detected_training_dataset_repository import (
    MediaPipeDetectedTrainingDatasetRepository,
)
from nd_posture_guard.desktop.desktop_integration_installer import (
    DesktopIntegrationInstaller,
)
from nd_posture_guard.model.example_posture_classifier import ExamplePostureClassifier
from nd_posture_guard.training.auto_detected_training_session import (
    AutoDetectedTrainingSession,
)
from nd_posture_guard.ui.auto_landmark_main_window import AutoLandmarkMainWindow
from nd_posture_guard.vision.mediapipe_posture_geometry_detector import (
    MediaPipePostureGeometryDetector,
)
from nd_posture_guard.vision.opencv_compatibility_validator import (
    OpenCvCompatibilityValidator,
)
from nd_posture_guard.vision.posture_geometry_feature_extractor import (
    PostureGeometryFeatureExtractor,
)


class MediaPipePostureGuardApplication:
    """Per-frame MediaPipe landmarks with no LK optical-flow point propagation."""

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

        self._window = AutoLandmarkMainWindow(
            settings.title,
            settings.classifier_bad_score_threshold,
            settings.training_data_root,
        )
        if desktop_integration.project_icon_path.is_file():
            self._window.setWindowIcon(QIcon(str(desktop_integration.project_icon_path)))

        OpenCvCompatibilityValidator().validate()
        camera_device = StableCameraDeviceFinder(
            width=settings.camera_width,
            height=settings.camera_height,
            fps=settings.camera_fps,
        ).find(settings.camera_device)
        camera = CachedUsableCameraReader(
            StableCameraReader(
                camera_device,
                settings.camera_width,
                settings.camera_height,
                settings.camera_fps,
                settings.mirror_camera,
            )
        )

        feature_extractor = PostureGeometryFeatureExtractor()
        detector = MediaPipePostureGeometryDetector()
        dataset_repository = MediaPipeDetectedTrainingDatasetRepository(
            root=Path(settings.training_data_root),
            novelty_multiplier=settings.training_novelty_multiplier,
            recording_fps=1000.0 / max(settings.detection_interval_ms, 1),
        )
        classifier = ExamplePostureClassifier(
            bad_score_threshold=settings.classifier_bad_score_threshold,
            required_bad_frames=settings.required_bad_frames,
            k_neighbors=settings.classifier_k_neighbors,
        )
        training_session = AutoDetectedTrainingSession(
            frames_per_sample=settings.training_frames_per_sample,
            detector=detector,
        )

        worker = AutoDetectedMonitoringWorker(
            camera=camera,
            feature_extractor=feature_extractor,
            geometry_tracker=detector,
            classifier=classifier,
            training_session=training_session,
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
