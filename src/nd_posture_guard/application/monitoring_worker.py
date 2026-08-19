from __future__ import annotations

from queue import Empty, Queue
from threading import Event
import time

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QThread, Signal

from nd_posture_guard.application.monitoring_frame import MonitoringFrame
from nd_posture_guard.camera.camera_reader import CameraReader
from nd_posture_guard.config.settings_repository import SettingsRepository
from nd_posture_guard.posture.posture_evaluation import PostureEvaluation
from nd_posture_guard.posture.posture_evaluator import PostureEvaluator
from nd_posture_guard.posture.shoulder_calibration_session import ShoulderCalibrationSession
from nd_posture_guard.vision.shoulder_anchor_tracker import ShoulderAnchorTracker


class MonitoringWorker(QThread):
    frame_ready = Signal(object)
    status_changed = Signal(str)
    alert_requested = Signal()
    fatal_error = Signal(str)
    camera_started = Signal(str)
    calibration_started = Signal(int)
    calibration_progress = Signal(int, int, str)
    calibration_points_required = Signal(str)
    calibration_action_required = Signal(str)
    calibration_completed = Signal()
    calibration_failed = Signal(str)
    calibration_cleared = Signal()

    def __init__(
        self,
        camera: CameraReader,
        shoulder_tracker: ShoulderAnchorTracker,
        posture_evaluator: PostureEvaluator,
        calibration_session: ShoulderCalibrationSession,
        settings_repository: SettingsRepository,
        detection_interval_ms: int,
        warning_text: str,
    ) -> None:
        super().__init__()
        self._camera = camera
        self._shoulder_tracker = shoulder_tracker
        self._posture_evaluator = posture_evaluator
        self._calibration_session = calibration_session
        self._settings_repository = settings_repository
        self._detection_interval_seconds = max(0.02, detection_interval_ms / 1000.0)
        self._warning_text = warning_text
        self._stop_event = Event()
        self._commands: Queue[tuple[str, object | None]] = Queue()
        self._frozen_frame: NDArray[np.uint8] | None = None
        self._snapshot_requested = False
        self._calibration_message = "Not calibrated"

    def request_stop(self) -> None:
        self._stop_event.set()

    def request_calibration(self) -> None:
        self._commands.put(("calibrate", None))

    def request_calibration_point(self, x: float, y: float) -> None:
        self._commands.put(("calibration_point", (float(x), float(y))))

    def request_clear_reference(self) -> None:
        self._commands.put(("clear_reference", None))

    def request_shoulder_drop_percent(self, value: float) -> None:
        self._commands.put(("shoulder_drop", float(value)))

    def run(self) -> None:
        try:
            self._camera.open()
            self.camera_started.emit(self._camera.device_label)
            while not self._stop_event.is_set():
                started = time.monotonic()
                self._process_commands()
                self._process_frame()
                elapsed = time.monotonic() - started
                self._stop_event.wait(max(0.0, self._detection_interval_seconds - elapsed))
        except Exception as exc:
            self.fatal_error.emit(str(exc))
        finally:
            self._camera.close()

    def _process_commands(self) -> None:
        while True:
            try:
                command, value = self._commands.get_nowait()
            except Empty:
                return

            if command == "calibrate":
                self._handle_calibrate_command()
            elif command == "calibration_point" and isinstance(value, tuple):
                self._handle_calibration_point(value)
            elif command == "clear_reference":
                self._posture_evaluator.clear()
                self._shoulder_tracker.clear()
                self._calibration_session.clear()
                self._frozen_frame = None
                self._snapshot_requested = False
                self._calibration_message = "Not calibrated"
                self.calibration_cleared.emit()
                self.status_changed.emit("Calibration cleared. Sit straight and press Calibrate shoulders.")
            elif command == "shoulder_drop" and isinstance(value, float):
                self._posture_evaluator.set_shoulder_drop_trigger_percent(value)
                self._settings_repository.save_shoulder_drop_trigger_percent(value)

    def _handle_calibrate_command(self) -> None:
        if self._calibration_session.active:
            if self._calibration_session.waiting_for_turn:
                if self._calibration_session.continue_after_turn():
                    self._frozen_frame = None
                    self._snapshot_requested = True
                    self._calibration_message = (
                        f"{self._calibration_session.stage_name}: hold the new straight pose for a moment..."
                    )
                    self.calibration_progress.emit(
                        self._calibration_session.point_count,
                        self._calibration_session.target_points,
                        self._calibration_message,
                    )
                return

            # Ignore duplicate/double-clicked calibration commands while a stage is
            # already collecting points. Previously this restarted CENTER and could
            # make the calibration appear to loop forever.
            self.status_changed.emit(
                "Calibration is already in progress. Finish the six requested points before continuing."
            )
            return

        self._posture_evaluator.clear()
        self._shoulder_tracker.clear()
        self._calibration_session.start()
        self._frozen_frame = None
        self._snapshot_requested = True
        target = self._calibration_session.target_points
        self._calibration_message = "CENTER: preparing a frozen frame..."
        self.calibration_started.emit(target)
        self.calibration_progress.emit(0, target, self._calibration_message)
        self.status_changed.emit(
            "Calibration started. Head position is ignored. You will mark 6 shoulder-line points in each of 3 straight poses (18 total)."
        )

    def _handle_calibration_point(self, value: tuple[float, float]) -> None:
        if (
            not self._calibration_session.active
            or self._calibration_session.waiting_for_turn
            or self._frozen_frame is None
        ):
            return
        stage = self._calibration_session.stage_name
        try:
            pose = self._calibration_session.add_point(
                value,
                self._frozen_frame.shape[1],
                self._frozen_frame.shape[0],
            )
        except ValueError as exc:
            message = f"{exc} Calibration stopped; press Restart calibration once to begin again."
            self._abort_calibration(message)
            return

        current = self._calibration_session.point_count
        target = self._calibration_session.target_points
        if pose is None:
            self._calibration_message = self._calibration_session.next_point_message
            self.calibration_progress.emit(current, target, self._calibration_message)
            return

        try:
            self._shoulder_tracker.add_calibration_pose(self._frozen_frame, pose)
        except ValueError as exc:
            message = f"{exc} Calibration stopped; press Restart calibration once to begin again."
            self._abort_calibration(message)
            return

        current = self._calibration_session.point_count
        self.calibration_progress.emit(current, target, f"{stage}: all 6 shoulder points saved.")
        self._frozen_frame = None

        if self._calibration_session.complete:
            try:
                reference = self._calibration_session.build_reference()
                self._posture_evaluator.calibrate(reference)
                self._calibration_message = "Shoulder calibration complete"
                self.calibration_completed.emit()
                self.status_changed.emit(
                    "CALIBRATION COMPLETE. 18 shoulder points were recorded. Return to your normal centered position; only shoulder movement is monitored."
                )
            except ValueError as exc:
                message = f"Invalid shoulder calibration: {exc}"
                self._abort_calibration(message)
                return
            self._calibration_session.clear()
            return

        next_stage = self._calibration_session.stage_name
        if next_stage == "LEFT":
            message = (
                "CENTER saved. Keep your back straight and rotate your upper body just a little to YOUR LEFT. "
                "Then press Continue."
            )
        else:
            message = (
                "LEFT saved. Keep your back straight and rotate your upper body just a little to YOUR RIGHT. "
                "Then press Continue."
            )
        self._calibration_message = message
        self.calibration_action_required.emit(message)
        self.status_changed.emit(message)

    def _abort_calibration(self, message: str) -> None:
        self._posture_evaluator.clear()
        self._shoulder_tracker.clear()
        self._calibration_session.clear()
        self._frozen_frame = None
        self._snapshot_requested = False
        self._calibration_message = message
        self.calibration_failed.emit(message)
        self.status_changed.emit(message)

    def _process_frame(self) -> None:
        live_frame = self._camera.read()
        if live_frame is None:
            self.status_changed.emit("Camera frame could not be read.")
            return

        if self._snapshot_requested and self._calibration_session.active:
            self._frozen_frame = live_frame.copy()
            self._snapshot_requested = False
            stage = self._calibration_session.stage_name
            message = self._calibration_session.next_point_message
            self._calibration_message = message
            self.calibration_points_required.emit(message)
            self.calibration_progress.emit(
                self._calibration_session.point_count,
                self._calibration_session.target_points,
                message,
            )

        display_frame = self._frozen_frame if self._frozen_frame is not None else live_frame
        shoulders = None
        evaluation = self._empty_evaluation()
        if not self._calibration_session.active and self._posture_evaluator.calibrated:
            shoulders = self._shoulder_tracker.track(live_frame)
            evaluation = self._posture_evaluator.evaluate(
                shoulders,
                live_frame.shape[1],
                live_frame.shape[0],
            )
            self._handle_evaluation(evaluation)
        elif not self._calibration_session.active:
            self.status_changed.emit(
                "Not calibrated. Press Calibrate shoulders. No face, head, or pose model is used."
            )

        reference = self._posture_evaluator.reference
        calibration_active = self._calibration_session.active
        warning = self._warning_text if evaluation.state == PostureEvaluator.STATE_SLOUCH else None
        self.frame_ready.emit(
            MonitoringFrame(
                frame=display_frame,
                shoulders=shoulders,
                reference_shoulders=reference.shoulders if reference is not None else None,
                posture_state=evaluation.state,
                shoulder_drop_percent=evaluation.shoulder_drop_percent,
                shoulder_width_change_percent=evaluation.shoulder_width_change_percent,
                shoulder_tilt_change_degrees=evaluation.shoulder_tilt_change_degrees,
                warning_text=warning,
                tracker_ready=self._shoulder_tracker.ready,
                shoulders_detected=shoulders is not None,
                tracker_confidence=self._shoulder_tracker.confidence,
                calibration_active=calibration_active,
                calibration_current=self._calibration_session.point_count if calibration_active else 0,
                calibration_target=self._calibration_session.target_points,
                calibration_message=self._calibration_message,
            )
        )

    def _handle_evaluation(self, evaluation: PostureEvaluation) -> None:
        if evaluation.state == PostureEvaluator.STATE_POSE_LOST:
            self.status_changed.emit(
                "Shoulder anchor tracking is incomplete. No warning will be generated until at least four anchors (two per shoulder) are valid again."
            )
            return
        if evaluation.state == PostureEvaluator.STATE_SLOUCH:
            self.status_changed.emit(
                f"Slouch detected from shoulders: drop={evaluation.shoulder_drop_percent:+.1f}%, "
                f"width={evaluation.shoulder_width_change_percent:+.1f}%, "
                f"tilt={evaluation.shoulder_tilt_change_degrees:.1f}°."
            )
            if evaluation.should_alert:
                self.alert_requested.emit()
            return
        self.status_changed.emit(
            f"Shoulders OK. drop={evaluation.shoulder_drop_percent:+.1f}%, "
            f"width={evaluation.shoulder_width_change_percent:+.1f}%, "
            f"tilt={evaluation.shoulder_tilt_change_degrees:.1f}°."
        )

    @staticmethod
    def _empty_evaluation() -> PostureEvaluation:
        return PostureEvaluation(
            state=PostureEvaluator.STATE_NOT_CALIBRATED,
            should_alert=False,
            shoulder_drop_percent=0.0,
            shoulder_width_change_percent=0.0,
            shoulder_tilt_change_degrees=0.0,
        )
