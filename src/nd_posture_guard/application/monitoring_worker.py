from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread
import time

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QThread, Signal

from nd_posture_guard.application.monitoring_frame import MonitoringFrame
from nd_posture_guard.camera.camera_reader import CameraReader
from nd_posture_guard.data.training_dataset_repository import TrainingDatasetRepository
from nd_posture_guard.model.example_posture_classifier import ExamplePostureClassifier
from nd_posture_guard.model.posture_geometry import PostureGeometry
from nd_posture_guard.training.example_training_session import ExampleTrainingSession
from nd_posture_guard.vision.posture_geometry_feature_extractor import PostureGeometryFeatureExtractor
from nd_posture_guard.vision.posture_geometry_tracker import PostureGeometryTracker


class MonitoringWorker(QThread):
    """Own camera I/O, lightweight landmark tracking, training and classification off the GUI thread."""

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
    training_samples_ready = Signal(object, bool)

    MIN_TRACKING_CONFIDENCE = 0.32
    MAX_UI_FPS = 8.0

    def __init__(
        self,
        camera: CameraReader,
        feature_extractor: PostureGeometryFeatureExtractor,
        geometry_tracker: PostureGeometryTracker,
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
        self._geometry_tracker = geometry_tracker
        self._classifier = classifier
        self._training_session = training_session
        self._dataset_repository = dataset_repository
        self._interval = max(0.04, detection_interval_ms / 1000.0)
        self._display_interval = max(self._interval, 1.0 / self.MAX_UI_FPS)
        self._warning_text = warning_text
        self._camera_width = int(camera_width)
        self._camera_height = int(camera_height)
        self._stop_event = Event()
        self._commands: Queue[tuple[str, object | None]] = Queue()
        self._profile_results: Queue[tuple[int, object | None, str | None]] = Queue()
        self._profile_reload_generation = 0
        self._known_good = 0
        self._known_bad = 0
        self._frozen_frame: NDArray[np.uint8] | None = None
        self._snapshot_requested = False
        self._training_message = ""
        self._monitoring_paused = False
        self._paused_before_training = False
        self._next_ui_emit = 0.0

    def request_stop(self) -> None:
        """Request clean worker shutdown."""
        self._stop_event.set()

    def request_add_good_training(self) -> None:
        """Queue a GOOD training session."""
        self._commands.put(("add_training", 0))

    def request_add_bad_training(self) -> None:
        """Queue a BAD training session."""
        self._commands.put(("add_training", 1))

    def request_training_point(self, x: float, y: float) -> None:
        """Queue one landmark selected in camera coordinates."""
        self._commands.put(("training_point", (float(x), float(y))))

    def request_cancel_training(self) -> None:
        """Queue cancellation of the active training sample."""
        self._commands.put(("cancel_training", None))

    def request_monitoring_paused(self, paused: bool) -> None:
        """Queue a pause/resume change."""
        self._commands.put(("monitoring_paused", bool(paused)))

    def request_bad_score_threshold(self, value: float) -> None:
        """Queue an alert-threshold change."""
        self._commands.put(("bad_threshold", float(value)))

    def request_training_samples(self) -> None:
        """Request the review dialog to open with the current sample list."""
        self._commands.put(("list_training_samples", None))

    def request_delete_training_sample(self, sample_id: str) -> None:
        """Queue deletion of one exact training sample."""
        self._commands.put(("delete_training_sample", str(sample_id)))

    def run(self) -> None:
        """Run the camera/geometry loop until shutdown is requested."""
        try:
            self._camera.open()
            self.camera_started.emit(self._camera.device_label)
            self._emit_initial_status()
            self._reload_profile()
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
        """Report current-model and legacy sample counts without pretending old samples are usable."""
        self._known_good, self._known_bad = self._dataset_repository.sample_counts()
        total_good, total_bad = self._dataset_repository.total_sample_counts()
        self.dataset_stats_changed.emit(self._known_good, self._known_bad, self._classifier.ready)
        legacy_good = max(0, total_good - self._known_good)
        legacy_bad = max(0, total_bad - self._known_bad)
        legacy_text = (
            f" Legacy preserved: {legacy_good} GOOD / {legacy_bad} BAD six-point samples."
            if legacy_good or legacy_bad
            else ""
        )
        if self._known_good and self._known_bad:
            self.status_changed.emit(
                f"Three-shape geometry dataset: {self._known_good} GOOD / {self._known_bad} BAD. "
                f"Rebuilding the personal model in background.{legacy_text}"
            )
        else:
            self.status_changed.emit(
                f"Three-shape geometry dataset: {self._known_good} GOOD / {self._known_bad} BAD. "
                f"Add at least one new example of each class.{legacy_text}"
            )

    def _process_commands(self) -> None:
        """Apply completed profile builds, then drain pending UI commands."""
        self._apply_profile_results()
        while True:
            try:
                command, value = self._commands.get_nowait()
            except Empty:
                return
            if command == "add_training" and isinstance(value, int):
                self._start_training(value)
            elif command == "training_point" and isinstance(value, tuple):
                self._handle_training_point(value)
            elif command == "cancel_training":
                if self._training_session.active:
                    self._abort_training("Cancelled by user.")
            elif command == "monitoring_paused" and isinstance(value, bool):
                if self._training_session.active:
                    self.status_changed.emit("Finish or cancel the current training sample first.")
                    continue
                self._monitoring_paused = value
                self.monitoring_paused_changed.emit(value)
                self.status_changed.emit(
                    "Monitoring paused. Geometry stays visible but no beep is generated."
                    if value
                    else "Monitoring resumed."
                )
            elif command == "bad_threshold" and isinstance(value, float):
                self._classifier.set_bad_score_threshold(value)
                self.status_changed.emit(f"Beep threshold active: {value * 100:.0f}%.")
            elif command == "list_training_samples":
                self.training_samples_ready.emit(self._dataset_repository.list_samples(), True)
            elif command == "delete_training_sample" and isinstance(value, str):
                self._delete_training_sample(value)

    def _delete_training_sample(self, sample_id: str) -> None:
        """Delete one sample and refresh an already-open review dialog without reopening it."""
        if self._training_session.active:
            self.status_changed.emit("Finish or cancel the current training sample before deleting saved data.")
            self.training_samples_ready.emit(self._dataset_repository.list_samples(), False)
            return
        samples_before = self._dataset_repository.list_samples()
        deleted_label = next(
            (sample.label for sample in samples_before if sample.sample_id == sample_id),
            None,
        )
        deleted = self._dataset_repository.delete_sample(sample_id)
        if not deleted:
            self.status_changed.emit(f"Training sample not found or could not be deleted: {sample_id}")
            self.training_samples_ready.emit(self._dataset_repository.list_samples(), False)
            return
        if deleted_label == 0:
            self._known_good = max(0, self._known_good - 1)
        elif deleted_label == 1:
            self._known_bad = max(0, self._known_bad - 1)
        else:
            self._known_good, self._known_bad = self._dataset_repository.sample_counts()
        self._reload_profile()
        self.dataset_stats_changed.emit(
            self._known_good, self._known_bad, self._classifier.ready
        )
        self.training_samples_ready.emit(self._dataset_repository.list_samples(), False)
        self.status_changed.emit(
            f"Training sample deleted. Geometry dataset now has "
            f"{self._known_good} GOOD / {self._known_bad} BAD usable samples."
        )

    def _start_training(self, label: int) -> None:
        """Freeze the next frame and begin the three-shape geometry sequence."""
        if self._training_session.active:
            self.status_changed.emit("A training sample is already being collected.")
            return
        self._paused_before_training = self._monitoring_paused
        self._training_session.start(label)
        self._geometry_tracker.reset()
        self._frozen_frame = None
        self._snapshot_requested = True
        self._monitoring_paused = True
        self.monitoring_paused_changed.emit(True)
        label_name = self._training_session.label_name
        self.training_started.emit(label_name)
        self._training_message = (
            f"Adding a {label_name} example. Hold the posture you want to teach. "
            "The next frame freezes; then define exactly three shapes: face triangle, "
            "left-side-of-image shoulder line and right-side-of-image shoulder line."
        )
        self.status_changed.emit(self._training_message)

    def _handle_training_point(self, point: tuple[float, float]) -> None:
        """Add a defining point and start clip recording after all three shapes are valid."""
        if not self._training_session.selecting_points or self._frozen_frame is None:
            return
        try:
            ready_to_record = self._training_session.add_point(
                point, self._frozen_frame.shape[1], self._frozen_frame.shape[0]
            )
        except ValueError as exc:
            label = self._training_session.label
            self._training_session.start(label)
            self._geometry_tracker.reset()
            self._frozen_frame = None
            self._snapshot_requested = True
            self._training_message = (
                f"{exc} A fresh frame will be captured; redraw all three shapes from Shape 1/3."
            )
            self.status_changed.emit(f"Point selection: {self._training_message}")
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
        """Cancel the active sample and restore the previous pause state."""
        self._training_session.clear()
        self._frozen_frame = None
        self._snapshot_requested = False
        self._training_message = ""
        self._monitoring_paused = self._paused_before_training
        self._geometry_tracker.reset()
        self.monitoring_paused_changed.emit(self._monitoring_paused)
        self.training_failed.emit(message)
        self.status_changed.emit(f"Training sample cancelled: {message}")

    def _process_frame(self) -> None:
        """Process one camera frame with bounded CPU/UI work."""
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
                    feature = self._feature_extractor.extract(geometry, frame.shape[1], frame.shape[0])
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
        must_emit = self._training_session.active or now >= self._next_ui_emit
        if not must_emit:
            return
        if self._training_session.active or self._next_ui_emit <= 0.0:
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
                tracking_confidence=(geometry.tracking_confidence if geometry is not None else 0.0),
                training_active=self._training_session.active,
                training_stage_current=(
                    len(self._training_session.current_points) if self._training_session.active else 0
                ),
                training_stage_total=(
                    self._training_session.POINTS_PER_SAMPLE if self._training_session.active else 0
                ),
                training_message=self._training_message,
                monitoring_paused=self._monitoring_paused,
            )
        )

    def _finish_training_sample(self) -> None:
        """Persist the completed sample, release training immediately, and rebuild the model elsewhere."""
        anchor = self._training_session.anchor_frame
        geometry = self._training_session.geometry
        label = self._training_session.label
        label_name = self._training_session.label_name
        try:
            feature = self._feature_extractor.extract(geometry, anchor.shape[1], anchor.shape[0])
            self._dataset_repository.save_sample(
                label=label,
                points=self._training_session.current_points,
                anchor_frame=anchor,
                frames=self._training_session.frames,
                feature=feature,
            )
        except Exception as exc:
            self._abort_training(f"Could not save training sample: {exc}")
            return

        self._training_session.clear()
        self._training_message = ""
        self._monitoring_paused = self._paused_before_training
        if label == 0:
            self._known_good += 1
        else:
            self._known_bad += 1

        self.monitoring_paused_changed.emit(self._monitoring_paused)
        self.training_sample_saved.emit(
            label_name, self._known_good, self._known_bad, self._classifier.ready
        )
        self.dataset_stats_changed.emit(
            self._known_good, self._known_bad, self._classifier.ready
        )
        self.status_changed.emit(
            f"{label_name} three-shape sample saved. Dataset: "
            f"{self._known_good} GOOD / {self._known_bad} BAD usable samples."
        )

        # A model cannot exist until both classes are present. In particular,
        # repeated GOOD-only collection must never trigger profile rebuilding.
        self._reload_profile()

    def _reload_profile(self) -> None:
        """Schedule profile rebuilding without ever doing it inside the camera loop."""
        self._profile_reload_generation += 1
        generation = self._profile_reload_generation

        if self._known_good < 1 or self._known_bad < 1:
            self._classifier.clear()
            self._geometry_tracker.set_profile(None)
            return

        Thread(
            target=self._build_profile_background,
            args=(generation,),
            name=f"posture-profile-{generation}",
            daemon=True,
        ).start()

    def _build_profile_background(self, generation: int) -> None:
        """Build the compact profile away from camera I/O and return it through a thread-safe queue."""
        try:
            profile = self._dataset_repository.build_profile(self._feature_extractor)
        except Exception as exc:
            self._profile_results.put((generation, None, str(exc)))
            return
        self._profile_results.put((generation, profile, None))

    def _apply_profile_results(self) -> None:
        """Apply only the newest completed background profile on the monitoring worker thread."""
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
                self._classifier.set_profile(profile)
                self._geometry_tracker.set_profile(profile)

            self.dataset_stats_changed.emit(
                self._known_good, self._known_bad, self._classifier.ready
            )
            if not self._training_session.active:
                if self._classifier.ready:
                    self.status_changed.emit(
                        f"Three-shape geometry model ready: "
                        f"{self._known_good} GOOD / {self._known_bad} BAD usable samples."
                    )
                else:
                    self.status_changed.emit(
                        f"Three-shape geometry dataset: "
                        f"{self._known_good} GOOD / {self._known_bad} BAD. "
                        "Add at least one example of each class."
                    )
