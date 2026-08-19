from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QTimer, Signal
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

from nd_posture_guard.ui.video_widget import VideoWidget
from nd_posture_guard.vision.tracked_shoulders import TrackedShoulders


class MainWindow(QMainWindow):
    calibrate_requested = Signal()
    calibration_point_selected = Signal(float, float)
    clear_reference_requested = Signal()
    shoulder_drop_changed = Signal(float)
    test_alert_requested = Signal()
    closing = Signal()

    def __init__(self, title: str, shoulder_drop_percent: float) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1060, 850)
        self._video = VideoWidget()
        self._video.shoulder_point_clicked.connect(self._on_shoulder_point_clicked)

        self._calibration_state = QLabel("NOT CALIBRATED")
        self._set_state_style("#9b2c2c")
        self._diagnostics = QLabel("TRACKER: --   SHOULDERS: --")
        self._diagnostics.setStyleSheet("font-weight: bold;")
        self._status = QLabel("Starting camera...")
        self._status.setWordWrap(True)

        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(18)
        self._progress.setValue(0)
        self._progress.setFormat("Calibration not started")
        self._progress.setTextVisible(True)
        self._progress.setMinimumHeight(24)

        self._drop_spin = QDoubleSpinBox()
        self._drop_spin.setRange(2.0, 30.0)
        self._drop_spin.setDecimals(1)
        self._drop_spin.setSingleStep(0.5)
        self._drop_spin.setSuffix(" %")
        self._drop_spin.setValue(shoulder_drop_percent)
        self._drop_spin.valueChanged.connect(self.shoulder_drop_changed.emit)

        self._calibrate_button = QPushButton("Calibrate shoulders")
        self._calibrate_button.clicked.connect(self._on_calibrate_clicked)
        self._clear_button = QPushButton("Clear calibration")
        self._clear_button.clicked.connect(self.clear_reference_requested.emit)
        self._test_alert_button = QPushButton("Test one long beep")
        self._test_alert_button.clicked.connect(self.test_alert_requested.emit)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Shoulder-drop gate:"))
        controls.addWidget(self._drop_spin)
        controls.addStretch(1)
        controls.addWidget(self._test_alert_button)
        controls.addWidget(self._calibrate_button)
        controls.addWidget(self._clear_button)

        legend = QLabel(
            "HEAD AND FACE ARE COMPLETELY IGNORED. In each pose you mark 6 shoulder-line points: "
            "3 on the LEFT shoulder from the neck outward, then 3 on the RIGHT shoulder in the same order. "
            "Repeat for CENTER, a very small LEFT turn, and a very small RIGHT turn: 18 points total. "
            "The same six shoulder anchors are tracked directly; no new face/background feature is added. "
            "Green = calibrated CENTER anchors | Cyan = valid current anchors | Gray = temporarily lost anchor | Red = slouch."
        )
        legend.setWordWrap(True)

        state_row = QHBoxLayout()
        state_row.addWidget(self._calibration_state)
        state_row.addWidget(self._diagnostics, 1)

        layout = QVBoxLayout()
        layout.addWidget(self._video, 1)
        layout.addLayout(state_row)
        layout.addWidget(self._progress)
        layout.addWidget(legend)
        layout.addLayout(controls)
        layout.addWidget(self._status)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def _on_calibrate_clicked(self) -> None:
        self._video.set_selection_enabled(False)
        self._calibrate_button.setEnabled(False)
        self._calibrate_button.setText("Please wait...")
        self.calibrate_requested.emit()

    def _on_shoulder_point_clicked(self, x: float, y: float) -> None:
        self.calibration_point_selected.emit(x, y)

    def show_frame(
        self,
        frame: NDArray[np.uint8],
        shoulders: TrackedShoulders | None,
        reference_shoulders: TrackedShoulders | None,
        posture_state: str,
        shoulder_drop_percent: float,
        shoulder_width_change_percent: float,
        shoulder_tilt_change_degrees: float,
        warning_text: str | None,
        tracker_ready: bool,
        shoulders_detected: bool,
        tracker_confidence: float,
        calibration_active: bool,
        calibration_current: int,
        calibration_target: int,
        calibration_message: str,
    ) -> None:
        valid_anchor_count = shoulders.valid_anchor_count if shoulders is not None else 0
        self._diagnostics.setText(
            f"TRACKER: {'OK' if tracker_ready else 'NO'}   "
            f"SHOULDERS: {'OK' if shoulders_detected else 'NO'}   "
            f"ANCHORS: {valid_anchor_count}/6   "
            f"MATCH: {tracker_confidence:.2f}"
        )
        if shoulders_detected:
            self._diagnostics.setStyleSheet("font-weight: bold; color: #16803a;")
        elif tracker_ready:
            self._diagnostics.setStyleSheet("font-weight: bold; color: #c07b00;")
        else:
            self._diagnostics.setStyleSheet("font-weight: bold; color: #a22b2b;")
        self._video.set_frame(
            frame,
            shoulders,
            reference_shoulders,
            posture_state,
            shoulder_drop_percent,
            shoulder_width_change_percent,
            shoulder_tilt_change_degrees,
            warning_text,
            tracker_ready,
            shoulders_detected,
            tracker_confidence,
            calibration_active,
            calibration_current,
            calibration_target,
            calibration_message,
        )

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_calibration_started(self, target: int) -> None:
        self._progress.setMaximum(max(1, target))
        self._progress.setValue(0)
        self._progress.setFormat(f"0/{target} — preparing CENTER")
        self._calibrate_button.setText("Preparing image...")
        self._calibrate_button.setEnabled(False)
        self._calibration_state.setText("CALIBRATING SHOULDERS")
        self._set_state_style("#d59a00", black_text=True)
        self._video.set_selection_enabled(False)
        self._video.set_calibration_overlay("CENTER\nPreparing a frozen image...")

    def set_calibration_progress(self, current: int, target: int, message: str) -> None:
        self._progress.setMaximum(max(1, target))
        self._progress.setValue(min(current, target))
        self._progress.setFormat(f"{current}/{target} — {message}")
        self._video.set_calibration_overlay(message)

    def set_calibration_points_required(self, message: str) -> None:
        self._calibrate_button.setText("Click the 6 requested shoulder points")
        self._calibrate_button.setEnabled(False)
        self._video.set_selection_enabled(True, 6)
        self._video.set_calibration_overlay(message)

    def set_calibration_action_required(self, message: str) -> None:
        self._video.set_selection_enabled(False)
        self._video.clear_selection_points()
        self._progress.setFormat(message)
        self._calibrate_button.setText("I am in position — Continue")
        self._calibrate_button.setEnabled(True)
        self._video.set_calibration_overlay(message)

    def set_calibration_complete(self) -> None:
        self._video.set_selection_enabled(False)
        self._video.clear_selection_points()
        self._progress.setValue(self._progress.maximum())
        self._progress.setFormat("18-point shoulder calibration complete")
        self._calibrate_button.setText("Recalibrate shoulders")
        self._calibrate_button.setEnabled(True)
        self._calibration_state.setText("CALIBRATED — SHOULDERS ONLY")
        self._set_state_style("#16803a")
        self._video.set_calibration_overlay(
            "CALIBRATED\nReturn to your normal centered pose. The six shoulder anchors are tracked directly. If fewer than four anchors remain reliable, warning is paused instead of guessing."
        )
        QTimer.singleShot(3000, lambda: self._video.set_calibration_overlay(None))

    def set_calibration_failed(self, message: str) -> None:
        self._video.set_selection_enabled(False)
        self._progress.setFormat(f"Calibration problem — {message}")
        self._calibrate_button.setText("Restart calibration")
        self._calibrate_button.setEnabled(True)
        self._calibration_state.setText("CALIBRATION NEEDS ATTENTION")
        self._set_state_style("#9b2c2c")
        self._video.set_calibration_overlay(f"CALIBRATION PROBLEM\n{message}")

    def set_calibration_cleared(self) -> None:
        self._video.set_selection_enabled(False)
        self._video.clear_selection_points()
        self._progress.setValue(0)
        self._progress.setFormat("Calibration not started")
        self._calibrate_button.setText("Calibrate shoulders")
        self._calibrate_button.setEnabled(True)
        self._calibration_state.setText("NOT CALIBRATED")
        self._set_state_style("#9b2c2c")
        self._video.set_calibration_overlay("NOT CALIBRATED")

    def _set_state_style(self, background: str, black_text: bool = False) -> None:
        foreground = "black" if black_text else "white"
        self._calibration_state.setStyleSheet(
            f"font-weight: bold; font-size: 15px; color: {foreground}; "
            f"background: {background}; padding: 6px 10px; border-radius: 4px;"
        )

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.closing.emit()
        super().closeEvent(event)
