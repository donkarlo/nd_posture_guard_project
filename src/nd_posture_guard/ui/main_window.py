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

        self._video = VideoWidget()
        self._video.shoulder_point_clicked.connect(self.training_point_selected.emit)

        self._state = QLabel("TRAINING DATASET: NOT READY")
        self._state.setStyleSheet("font-weight: bold; color: #a22b2b;")
        self._diagnostics = QLabel("GOOD: 0   BAD: 0")
        self._diagnostics.setStyleSheet("font-weight: bold;")
        self._status = QLabel("Starting camera...")
        self._status.setWordWrap(True)
        self._data_path = QLabel(f"Persistent data: {data_root}")
        self._data_path.setWordWrap(True)
        self._data_path.setStyleSheet("color: #555;")

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat("Ready to add examples")
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
        self._review_button.clicked.connect(self.review_training_requested.emit)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Beep threshold:"))
        controls.addWidget(self._threshold_spin)
        controls.addStretch(1)
        controls.addWidget(self._pause_button)
        controls.addWidget(self._good_button)
        controls.addWidget(self._bad_button)
        controls.addWidget(self._review_button)

        legend = QLabel(
            "Training is incremental. Add GOOD examples that must NOT beep and BAD examples that SHOULD beep. "
            "Each example asks for exactly six points: three from the inner/neck-side to the outer tip of the left "
            "shoulder, then three in the same order on the right shoulder. The points define fixed camera-space "
            "shoulder regions; the regions do not follow you during monitoring, so downward/forward shoulder movement "
            "remains measurable and there is no long-term point drift. A bad-score above 50% is classified as BAD. "
            "The Beep threshold controls only when the warning sound starts; 50% is the natural decision boundary. "
            "New examples are appended to the "
            "persistent dataset. Review/delete lets you watch saved clips newest-first and remove bad training examples."
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
        self._pause_button.setEnabled(False)
        self.monitoring_pause_requested.emit(not self._monitoring_paused)

    def show_frame(
        self,
        frame: NDArray[np.uint8],
        posture_state: str,
        bad_score: float,
        confidence: float,
        warning_text: str | None,
        model_ready: bool,
        left_roi: tuple[int, int, int, int] | None,
        right_roi: tuple[int, int, int, int] | None,
        training_active: bool,
        training_current: int,
        training_total: int,
        training_message: str,
        monitoring_paused: bool,
    ) -> None:
        self._video.set_frame(
            frame,
            left_roi,
            right_roi,
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
        if self._review_dialog is None:
            self._review_dialog = TrainingDataReviewDialog(self)
            self._review_dialog.delete_sample_requested.connect(
                self.delete_training_sample_requested.emit
            )
        self._review_dialog.set_samples(samples)
        self._review_dialog.show()
        self._review_dialog.raise_()
        self._review_dialog.activateWindow()

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_dataset_stats(self, good: int, bad: int, model_ready: bool) -> None:
        self._diagnostics.setText(f"GOOD samples: {good}   BAD samples: {bad}")
        if model_ready:
            self._state.setText("PERSONAL SHOULDER MODEL: READY")
            self._state.setStyleSheet("font-weight: bold; color: #16803a;")
        else:
            self._state.setText("TRAINING DATASET: NEEDS BOTH GOOD AND BAD")
            self._state.setStyleSheet("font-weight: bold; color: #a22b2b;")

    def set_monitoring_paused(self, paused: bool) -> None:
        self._monitoring_paused = bool(paused)
        if not self._training_active_ui:
            self._pause_button.setEnabled(True)
        self._pause_button.setText("Resume monitoring" if paused else "Pause monitoring")
        self._pause_button.setStyleSheet(
            "font-weight: bold; background: #d59a00; color: black;" if paused else ""
        )

    def set_training_started(self, label_name: str) -> None:
        self._training_active_ui = True
        self._progress.setRange(0, 6)
        self._progress.setValue(0)
        self._progress.setFormat(f"{label_name}: waiting for six shoulder points")
        self._state.setText(f"ADDING {label_name} TRAINING EXAMPLE")
        self._state.setStyleSheet("font-weight: bold; color: #c07b00;")
        self._pause_button.setEnabled(False)
        self._good_button.setEnabled(False)
        self._bad_button.setEnabled(False)
        self._review_button.setEnabled(False)
        self._video.set_overlay("")
        self._video.clear_selection_points()

    def set_training_points_required(self, message: str, selected: int, target: int) -> None:
        self._progress.setRange(0, target)
        self._progress.setValue(selected)
        self._progress.setFormat(f"Shoulder points: {selected}/{target}")
        self._video.set_selection_enabled(True, target)
        self._video.set_overlay(message)

    def set_training_recording_progress(self, current: int, target: int, message: str) -> None:
        self._progress.setRange(0, target)
        self._progress.setValue(current)
        self._progress.setFormat(f"Recording: {current}/{target}")
        self._video.set_selection_enabled(False)
        self._video.set_overlay(message)

    def set_training_sample_saved(
        self,
        label_name: str,
        good: int,
        bad: int,
        model_ready: bool,
    ) -> None:
        self._training_active_ui = False
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setFormat(f"{label_name} sample saved and merged")
        self._good_button.setEnabled(True)
        self._bad_button.setEnabled(True)
        self._review_button.setEnabled(True)
        self._pause_button.setEnabled(True)
        self._video.set_selection_enabled(False)
        self._video.clear_selection_points()
        self._video.set_overlay(
            f"{label_name} SAMPLE SAVED\nDataset: {good} GOOD / {bad} BAD. You can add more examples whenever you want."
        )
        self.set_dataset_stats(good, bad, model_ready)

    def set_training_failed(self, message: str) -> None:
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
        # Commit any number still being edited before shutdown. This makes the
        # last visible threshold survive even when the user types a value and
        # immediately closes the window without pressing Enter or Tab.
        self._threshold_spin.interpretText()
        self.bad_score_threshold_changed.emit(float(self._threshold_spin.value()) / 100.0)
        self.closing.emit()
        super().closeEvent(event)
