from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nd_posture_guard.data.training_sample_summary import TrainingSampleSummary
from nd_posture_guard.ui.training_data_review_dialog import TrainingDataReviewDialog
from nd_posture_guard.ui.video_widget import VideoWidget


class MainWindow(QMainWindow):
    """Main posture-monitor UI with three-shape training and persistent review controls."""

    add_good_training_requested = Signal()
    add_bad_training_requested = Signal()
    training_point_selected = Signal(float, float)
    bad_score_threshold_changed = Signal(float)
    monitoring_pause_requested = Signal(bool)
    review_training_requested = Signal()
    delete_training_sample_requested = Signal(str)
    closing = Signal()

    def __init__(self, title: str, bad_score_threshold: float, data_root: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1120, 890)
        self._monitoring_paused = False
        self._training_active_ui = False
        self._review_dialog: TrainingDataReviewDialog | None = None
        self._review_should_be_visible = False

        self._video = VideoWidget()
        self._video.geometry_point_clicked.connect(self.training_point_selected.emit)
        self._state = QLabel("THREE-SHAPE DATASET: NOT READY")
        self._state.setStyleSheet("font-weight: bold; color: #a22b2b;")
        self._diagnostics = QLabel("GEOMETRY GOOD: 0   BAD: 0")
        self._diagnostics.setStyleSheet("font-weight: bold;")
        self._status = QLabel("Starting camera...")
        self._status.setWordWrap(True)
        self._data_path = QLabel(f"Persistent data: {data_root}")
        self._data_path.setWordWrap(True)
        self._data_path.setStyleSheet("color: #555;")

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Ready to draw 3 posture shapes")
        self._progress.setMinimumHeight(24)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(50.0, 90.0)
        self._threshold_spin.setDecimals(0)
        self._threshold_spin.setSingleStep(1.0)
        self._threshold_spin.setSuffix(" %")
        self._threshold_spin.setValue(bad_score_threshold * 100.0)
        self._threshold_spin.valueChanged.connect(
            lambda value: self.bad_score_threshold_changed.emit(float(value) / 100.0)
        )

        self._pause_button = QPushButton("Pause monitoring")
        self._pause_button.clicked.connect(self._on_pause_clicked)
        self._good_button = QPushButton("Add good training data")
        self._good_button.clicked.connect(self.add_good_training_requested.emit)
        self._bad_button = QPushButton("Add bad training data")
        self._bad_button.clicked.connect(self.add_bad_training_requested.emit)
        self._review_button = QPushButton("Review / delete training videos")
        self._review_button.clicked.connect(self._on_review_clicked)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Beep threshold:"))
        controls.addWidget(self._threshold_spin)
        controls.addStretch(1)
        controls.addWidget(self._pause_button)
        controls.addWidget(self._good_button)
        controls.addWidget(self._bad_button)
        controls.addWidget(self._review_button)

        legend = QLabel(
            "For every GOOD or BAD example draw exactly three things on the frozen image: "
            "(1) a triangle defined by the centers of both eyes and the lowest point of the chin; "
            "(2) the left shoulder line from the neck/shoulder junction to the outer end of the shoulder; "
            "(3) the same line for the right shoulder. The three drawings are stored as seven points internally, "
            "so monitoring can track the same face triangle and shoulder lines without using shirt texture."
        )
        legend.setWordWrap(True)

        state_row = QHBoxLayout()
        state_row.addWidget(self._state)
        state_row.addWidget(self._diagnostics, 1)
        layout = QVBoxLayout()
        layout.addWidget(self._video, 1)
        layout.addLayout(state_row)
        layout.addWidget(self._progress)
        layout.addWidget(legend)
        layout.addLayout(controls)
        layout.addWidget(self._data_path)
        layout.addWidget(self._status)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def _on_pause_clicked(self) -> None:
        """Request the inverse of the current pause state and wait for worker acknowledgement."""
        self._pause_button.setEnabled(False)
        self.monitoring_pause_requested.emit(not self._monitoring_paused)

    def _on_review_clicked(self) -> None:
        """Mark review as explicitly requested before asking the worker for its current list."""
        self._review_should_be_visible = True
        self.review_training_requested.emit()

    def _on_review_closed(self) -> None:
        """Prevent any delayed worker response from reopening a dialog the user closed."""
        self._review_should_be_visible = False

    def show_frame(
        self,
        frame: NDArray[np.uint8],
        posture_state: str,
        bad_score: float,
        confidence: float,
        warning_text: str | None,
        model_ready: bool,
        geometry_points: tuple[tuple[float, float], ...] | None,
        tracking_confidence: float,
        training_active: bool,
        training_current: int,
        training_total: int,
        training_message: str,
        monitoring_paused: bool,
    ) -> None:
        """Apply one throttled monitoring update to the video/status view."""
        self._video.set_frame(
            frame,
            geometry_points,
            tracking_confidence,
            posture_state,
            bad_score,
            confidence,
            warning_text,
            training_active,
            training_current,
            training_total,
            training_message,
        )
        if monitoring_paused and not training_active:
            self._diagnostics.setStyleSheet("font-weight: bold; color: #c07b00;")
        elif model_ready and not training_active:
            self._diagnostics.setStyleSheet("font-weight: bold; color: #16803a;")

    def show_training_samples(self, samples: list[TrainingSampleSummary]) -> None:
        """Open review only if the user still wants it open when the worker response arrives."""
        if not self._review_should_be_visible:
            return
        dialog = self._ensure_review_dialog()
        dialog.set_samples(samples)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def refresh_training_samples(self, samples: list[TrainingSampleSummary]) -> None:
        """Refresh a visible review dialog without ever resurrecting a closed one."""
        if (
            self._review_should_be_visible
            and self._review_dialog is not None
            and self._review_dialog.isVisible()
        ):
            self._review_dialog.set_samples(samples)

    def _ensure_review_dialog(self) -> TrainingDataReviewDialog:
        """Create and connect the single reusable review dialog lazily."""
        if self._review_dialog is None:
            self._review_dialog = TrainingDataReviewDialog(self)
            self._review_dialog.delete_sample_requested.connect(
                self.delete_training_sample_requested.emit
            )
            self._review_dialog.closed.connect(self._on_review_closed)
        return self._review_dialog

    def set_status(self, text: str) -> None:
        """Replace the lower status message."""
        self._status.setText(text)

    def set_dataset_stats(self, good: int, bad: int, model_ready: bool) -> None:
        """Show counts for samples actually usable by the current seven-point model."""
        self._diagnostics.setText(f"GEOMETRY GOOD: {good}   BAD: {bad}")
        if model_ready:
            self._state.setText("PERSONAL 3-SHAPE GEOMETRY MODEL: READY")
            self._state.setStyleSheet("font-weight: bold; color: #16803a;")
        else:
            self._state.setText("3-SHAPE DATASET: NEEDS NEW GOOD + BAD")
            self._state.setStyleSheet("font-weight: bold; color: #a22b2b;")

    def set_monitoring_paused(self, paused: bool) -> None:
        """Synchronize pause-button appearance with worker state."""
        self._monitoring_paused = bool(paused)
        if not self._training_active_ui:
            self._pause_button.setEnabled(True)
        self._pause_button.setText("Resume monitoring" if paused else "Pause monitoring")
        self._pause_button.setStyleSheet(
            "font-weight: bold; background: #d59a00; color: black;" if paused else ""
        )

    def set_training_started(self, label_name: str) -> None:
        """Enter three-shape drawing mode for a new training sample."""
        self._training_active_ui = True
        self._progress.setRange(0, 7)
        self._progress.setValue(0)
        self._progress.setFormat(f"{label_name}: draw 3 shapes (7 defining points)")
        self._state.setText(f"ADDING {label_name} — DRAW 3 SHAPES")
        self._state.setStyleSheet("font-weight: bold; color: #c07b00;")
        self._pause_button.setEnabled(False)
        self._good_button.setEnabled(False)
        self._bad_button.setEnabled(False)
        self._review_button.setEnabled(False)
        self._video.set_overlay("")
        self._video.clear_selection_points()

    def set_training_points_required(self, message: str, selected: int, target: int) -> None:
        """Show the current shape instruction and enable defining-point clicks."""
        self._progress.setRange(0, target)
        self._progress.setValue(selected)
        self._progress.setFormat(self._shape_progress_text(selected))
        self._video.set_selection_enabled(True, target)
        self._video.set_overlay(message)

    @staticmethod
    def _shape_progress_text(selected: int) -> str:
        """Translate seven internal points into three visible drawing stages."""
        if selected < 3:
            return f"Shape 1/3 — face triangle: {selected}/3 points"
        if selected < 5:
            return f"Shape 2/3 — left shoulder line: {selected - 3}/2 points"
        if selected < 7:
            return f"Shape 3/3 — right shoulder line: {selected - 5}/2 points"
        return "All 3 shapes defined"

    def set_training_recording_progress(self, current: int, target: int, message: str) -> None:
        """Disable clicks while recording the short sample clip."""
        self._progress.setRange(0, target)
        self._progress.setValue(current)
        self._progress.setFormat(f"Recording after 3 shapes: {current}/{target}")
        self._video.set_selection_enabled(False)
        self._video.set_overlay(message)

    def set_training_sample_saved(
        self, label_name: str, good: int, bad: int, model_ready: bool
    ) -> None:
        """Return the controls to idle after a geometry sample is persisted."""
        self._training_active_ui = False
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setFormat(f"{label_name} three-shape sample saved")
        self._good_button.setEnabled(True)
        self._bad_button.setEnabled(True)
        self._review_button.setEnabled(True)
        self._pause_button.setEnabled(True)
        self._video.set_selection_enabled(False)
        self._video.clear_selection_points()
        self._video.set_overlay(
            f"{label_name} SAMPLE SAVED\nGeometry dataset: {good} GOOD / {bad} BAD."
        )
        self.set_dataset_stats(good, bad, model_ready)

    def set_training_failed(self, message: str) -> None:
        """Restore idle controls after a cancelled training session."""
        self._training_active_ui = False
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Training sample cancelled")
        self._good_button.setEnabled(True)
        self._bad_button.setEnabled(True)
        self._review_button.setEnabled(True)
        self._pause_button.setEnabled(True)
        self._video.set_selection_enabled(False)
        self._video.clear_selection_points()
        self._video.set_overlay(f"TRAINING SAMPLE CANCELLED\n{message}")

    def closeEvent(self, event) -> None:
        """Persist the threshold, close review state and stop background monitoring."""
        self._review_should_be_visible = False
        if self._review_dialog is not None and self._review_dialog.isVisible():
            self._review_dialog.close()
        self._threshold_spin.interpretText()
        self.bad_score_threshold_changed.emit(float(self._threshold_spin.value()) / 100.0)
        self.closing.emit()
        super().closeEvent(event)
