from __future__ import annotations

from queue import Empty, Queue
from threading import Event
import time

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QThread, Signal

from nd_posture_guard.application.monitoring_frame import MonitoringFrame
from nd_posture_guard.camera.camera_reader import CameraReader
from nd_posture_guard.data.training_dataset_repository import TrainingDatasetRepository
from nd_posture_guard.model.example_posture_classifier import ExamplePostureClassifier
from nd_posture_guard.training.example_training_session import ExampleTrainingSession
from nd_posture_guard.vision.shoulder_feature_extractor import ShoulderFeatureExtractor


class MonitoringWorker(QThread):
    frame_ready = Signal(object)
    status_changed = Signal(str)
    alert_requested = Signal()
    fatal_error = Signal(str)
    camera_started = Signal(str)
    training_started = Signal(str)
    training_points_required = Signal(str, int, int)
    training_recording_progress = Signal(int, int, str)
    training_sample_saved = Signal(str, int, int, bool)
    training_failed = Signal(str)
    dataset_stats_changed = Signal(int, int, bool)
    monitoring_paused_changed = Signal(bool)
    training_samples_ready = Signal(object)

    def __init__(
        self,
        camera: CameraReader,
        feature_extractor: ShoulderFeatureExtractor,
        classifier: ExamplePostureClassifier,
        training_session: ExampleTrainingSession,
        dataset_repository: TrainingDatasetRepository,
        detection_interval_ms: int,
        warning_text: str,
        camera_width: int,
        camera_height: int,
    ) -> None:
        super().__init__()
        self._camera = camera
        self._feature_extractor = feature_extractor
        self._classifier = classifier
        self._training_session = training_session
        self._dataset_repository = dataset_repository
        self._interval = max(0.03, detection_interval_ms / 1000.0)
        self._warning_text = warning_text
        self._camera_width = int(camera_width)
        self._camera_height = int(camera_height)
        self._stop_event = Event()
        self._commands: Queue[tuple[str, object | None]] = Queue()
        self._frozen_frame: NDArray[np.uint8] | None = None
        self._snapshot_requested = False
        self._training_message = ""
        self._monitoring_paused = False
        self._paused_before_training = False

    def request_stop(self) -> None:
        self._stop_event.set()

    def request_add_good_training(self) -> None:
        self._commands.put(("add_training", 0))

    def request_add_bad_training(self) -> None:
        self._commands.put(("add_training", 1))

    def request_training_point(self, x: float, y: float) -> None:
        self._commands.put(("training_point", (float(x), float(y))))

    def request_monitoring_paused(self, paused: bool) -> None:
        self._commands.put(("monitoring_paused", bool(paused)))

    def request_bad_score_threshold(self, value: float) -> None:
        self._commands.put(("bad_threshold", float(value)))

    def request_training_samples(self) -> None:
        self._commands.put(("list_training_samples", None))

    def request_delete_training_sample(self, sample_id: str) -> None:
        self._commands.put(("delete_training_sample", str(sample_id)))

    def run(self) -> None:
        try:
            self._camera.open()
            self.camera_started.emit(self._camera.device_label)
            self._reload_profile(self._camera_width, self._camera_height)
            self._emit_initial_status()
            while not self._stop_event.is_set():
                started = time.monotonic()
                self._process_commands()
                self._process_frame()
                elapsed = time.monotonic() - started
                self._stop_event.wait(max(0.0, self._interval - elapsed))
        except Exception as exc:
            self.fatal_error.emit(str(exc))
        finally:
            self._camera.close()

    def _emit_initial_status(self) -> None:
        good, bad = self._dataset_repository.sample_counts()
        self.dataset_stats_changed.emit(good, bad, self._classifier.ready)
        if self._classifier.ready:
            self.status_changed.emit(
                f"Training dataset loaded from {self._dataset_repository.root}. "
                f"Raw compatible samples are merged automatically: {good} GOOD, {bad} BAD. "
                "Runtime model uses fixed camera-space shoulder regions so shoulder drop is preserved."
            )
        else:
            self.status_changed.emit(
                f"Training data directory: {self._dataset_repository.root}. "
                f"Current samples: {good} GOOD, {bad} BAD. Add at least one example of each class."
            )

    def _process_commands(self) -> None:
        while True:
            try:
                command, value = self._commands.get_nowait()
            except Empty:
                return
            if command == "add_training" and isinstance(value, int):
                self._start_training(value)
            elif command == "training_point" and isinstance(value, tuple):
                self._handle_training_point(value)
            elif command == "monitoring_paused" and isinstance(value, bool):
                if self._training_session.active:
                    self.status_changed.emit("Finish the current training sample before changing Pause/Resume.")
                    continue
                self._monitoring_paused = value
                self.monitoring_paused_changed.emit(value)
                self.status_changed.emit(
                    "Monitoring paused. No beep will be generated."
                    if value
                    else "Monitoring resumed."
                )
            elif command == "bad_threshold" and isinstance(value, float):
                self._classifier.set_bad_score_threshold(value)
                self.status_changed.emit(
                    f"Beep threshold active: {value * 100:.0f}%."
                )
            elif command == "list_training_samples":
                self.training_samples_ready.emit(self._dataset_repository.list_samples())
            elif command == "delete_training_sample" and isinstance(value, str):
                self._delete_training_sample(value)

    def _delete_training_sample(self, sample_id: str) -> None:
        if self._training_session.active:
            self.status_changed.emit("Finish the current training sample before deleting saved data.")
            self.training_samples_ready.emit(self._dataset_repository.list_samples())
            return
        deleted = self._dataset_repository.delete_sample(sample_id)
        if not deleted:
            self.status_changed.emit(f"Training sample not found or could not be deleted: {sample_id}")
            self.training_samples_ready.emit(self._dataset_repository.list_samples())
            return
        self._reload_profile(self._camera_width, self._camera_height)
        good, bad = self._dataset_repository.sample_counts()
        self.dataset_stats_changed.emit(good, bad, self._classifier.ready)
        self.training_samples_ready.emit(self._dataset_repository.list_samples())
        self.status_changed.emit(
            f"Training sample deleted. Dataset now contains {good} GOOD and {bad} BAD samples."
        )

    def _start_training(self, label: int) -> None:
        if self._training_session.active:
            self.status_changed.emit("A training sample is already being collected.")
            return
        self._paused_before_training = self._monitoring_paused
        self._training_session.start(label)
        self._frozen_frame = None
        self._snapshot_requested = True
        self._monitoring_paused = True
        self.monitoring_paused_changed.emit(True)
        label_name = self._training_session.label_name
        self.training_started.emit(label_name)
        self._training_message = (
            f"Adding a {label_name} example. Hold the posture you want to teach. "
            "The next camera frame will freeze; then click six shoulder points."
        )
        self.status_changed.emit(self._training_message)

    def _handle_training_point(self, point: tuple[float, float]) -> None:
        if not self._training_session.selecting_points or self._frozen_frame is None:
            return
        try:
            ready_to_record = self._training_session.add_point(
                point, self._frozen_frame.shape[1], self._frozen_frame.shape[0]
            )
        except ValueError as exc:
            self._abort_training(str(exc))
            return

        selected = len(self._training_session.current_points)
        if not ready_to_record:
            self.training_points_required.emit(
                self._training_session.next_point_message,
                selected,
                self._training_session.POINTS_PER_SAMPLE,
            )
            return

        self._frozen_frame = None
        self._training_message = (
            f"RECORDING {self._training_session.label_name}: keep this posture naturally for about "
            f"{self._training_session.frames_per_sample * self._interval:.1f} seconds."
        )
        self.training_recording_progress.emit(
            0, self._training_session.frames_per_sample, self._training_message
        )

    def _abort_training(self, message: str) -> None:
        self._training_session.clear()
        self._frozen_frame = None
        self._snapshot_requested = False
        self._monitoring_paused = self._paused_before_training
        self.monitoring_paused_changed.emit(self._monitoring_paused)
        self.training_failed.emit(message)
        self.status_changed.emit(f"Training sample cancelled: {message}")

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
            left_roi = self._training_session.left_roi
            right_roi = self._training_session.right_roi
            assert left_roi is not None and right_roi is not None
            try:
                feature = self._feature_extractor.extract(frame, left_roi, right_roi)
                finished = self._training_session.add_observation(feature, frame)
                current = self._training_session.recording_count
                self.training_recording_progress.emit(
                    current, self._training_session.frames_per_sample, self._training_message
                )
                if finished:
                    self._finish_training_sample(frame.shape[1], frame.shape[0])
            except Exception as exc:
                self._abort_training(str(exc))

        display_frame = self._frozen_frame if self._frozen_frame is not None else frame
        posture_state = "training" if self._training_session.active else "not_trained"
        bad_score = 0.0
        confidence = 0.0
        nearest_distance = 0.0
        warning_text: str | None = None
        left_roi = self._classifier.profile.left_roi if self._classifier.profile is not None else None
        right_roi = self._classifier.profile.right_roi if self._classifier.profile is not None else None

        if self._classifier.ready and not self._training_session.active and not self._monitoring_paused:
            assert left_roi is not None and right_roi is not None
            feature = self._feature_extractor.extract(frame, left_roi, right_roi)
            result = self._classifier.classify(feature)
            posture_state = result.state
            bad_score = result.bad_score
            confidence = result.confidence
            nearest_distance = result.nearest_distance
            if result.should_alert:
                warning_text = self._warning_text
                self.alert_requested.emit()
        elif self._monitoring_paused and not self._training_session.active:
            posture_state = "paused"

        self.frame_ready.emit(
            MonitoringFrame(
                frame=display_frame.copy(),
                posture_state=posture_state,
                bad_score=bad_score,
                confidence=confidence,
                nearest_distance=nearest_distance,
                warning_text=warning_text,
                model_ready=self._classifier.ready,
                left_roi=left_roi,
                right_roi=right_roi,
                training_active=self._training_session.active,
                training_stage_current=1 if self._training_session.active else 0,
                training_stage_total=1 if self._training_session.active else 0,
                training_message=self._training_message,
                monitoring_paused=self._monitoring_paused,
            )
        )

    def _finish_training_sample(self, frame_width: int, frame_height: int) -> None:
        left_roi = self._training_session.left_roi
        right_roi = self._training_session.right_roi
        assert left_roi is not None and right_roi is not None
        label_name = self._training_session.label_name
        self._dataset_repository.save_sample(
            label=self._training_session.label,
            points=self._training_session.current_points,
            left_roi=left_roi,
            right_roi=right_roi,
            anchor_frame=self._training_session.anchor_frame,
            frames=self._training_session.frames,
            features=self._training_session.features,
        )
        self._training_session.clear()
        self._reload_profile(frame_width, frame_height)
        good, bad = self._dataset_repository.sample_counts()
        self._monitoring_paused = self._paused_before_training
        self.monitoring_paused_changed.emit(self._monitoring_paused)
        self.training_sample_saved.emit(label_name, good, bad, self._classifier.ready)
        self.dataset_stats_changed.emit(good, bad, self._classifier.ready)
        self.status_changed.emit(
            f"{label_name} sample saved and merged. Dataset now contains {good} GOOD and {bad} BAD samples."
        )

    def _reload_profile(self, frame_width: int, frame_height: int) -> None:
        profile = self._dataset_repository.build_profile(
            frame_width=frame_width,
            frame_height=frame_height,
            feature_extractor=self._feature_extractor,
        )
        if profile is None:
            self._classifier.clear()
        else:
            self._classifier.set_profile(profile)
