from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nd_posture_guard.data.training_sample_summary import TrainingSampleSummary


class TrainingDataReviewDialog(QDialog):
    """Review saved training clips newest-first and delete unwanted samples."""

    delete_sample_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review training videos")
        self.resize(1050, 700)

        self._samples: list[TrainingSampleSummary] = []
        self._capture: cv2.VideoCapture | None = None
        self._current_video_path: Path | None = None

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._select_row)

        self._preview = QLabel("Select a training sample")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(640, 480)
        self._preview.setStyleSheet("background: #111; color: #ddd;")

        self._details = QLabel("")
        self._details.setWordWrap(True)

        self._delete_button = QPushButton("Delete selected training sample")
        self._delete_button.setEnabled(False)
        self._delete_button.clicked.connect(self._delete_selected)

        self._close_button = QPushButton("Close")
        self._close_button.clicked.connect(self.close)

        right = QVBoxLayout()
        right.addWidget(self._preview, 1)
        right.addWidget(self._details)
        buttons = QHBoxLayout()
        buttons.addWidget(self._delete_button)
        buttons.addStretch(1)
        buttons.addWidget(self._close_button)
        right.addLayout(buttons)

        root = QHBoxLayout()
        root.addWidget(self._list, 1)
        right_widget = QWidget()
        right_widget.setLayout(right)
        root.addWidget(right_widget, 3)
        self.setLayout(root)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._next_frame)

    def set_samples(self, samples: list[TrainingSampleSummary]) -> None:
        selected_id = self._selected_sample_id()
        self._stop_video()
        self._samples = list(samples)
        self._list.clear()

        selected_row = -1
        for index, sample in enumerate(self._samples):
            label = "GOOD" if sample.label == 0 else "BAD"
            created = sample.created_at_utc.replace("T", " ").replace("+00:00", " UTC")
            item = QListWidgetItem(f"{label}   {created}\n{sample.sample_id}")
            item.setData(Qt.ItemDataRole.UserRole, sample.sample_id)
            if sample.label == 1:
                item.setToolTip("BAD training example — should cause a posture warning")
            else:
                item.setToolTip("GOOD training example — should not cause a posture warning")
            self._list.addItem(item)
            if sample.sample_id == selected_id:
                selected_row = index

        if self._samples:
            self._list.setCurrentRow(selected_row if selected_row >= 0 else 0)
        else:
            self._preview.setText("No saved training videos")
            self._details.setText("")
            self._delete_button.setEnabled(False)

    def _selected_sample_id(self) -> str | None:
        row = self._list.currentRow()
        if 0 <= row < len(self._samples):
            return self._samples[row].sample_id
        return None

    def _select_row(self, row: int) -> None:
        self._stop_video()
        if not (0 <= row < len(self._samples)):
            self._delete_button.setEnabled(False)
            return

        sample = self._samples[row]
        self._delete_button.setEnabled(True)
        label = "GOOD" if sample.label == 0 else "BAD"
        self._details.setText(
            f"Label: {label}    Frames: {sample.frame_count}\n"
            f"Saved: {sample.created_at_utc}\n"
            f"Sample: {sample.sample_id}"
        )

        if sample.video_path is not None and sample.video_path.is_file():
            self._current_video_path = sample.video_path
            self._capture = cv2.VideoCapture(str(sample.video_path))
            if self._capture.isOpened():
                self._timer.start()
                self._next_frame()
                return
            self._stop_video()

        if sample.anchor_frame_path is not None and sample.anchor_frame_path.is_file():
            frame = cv2.imread(str(sample.anchor_frame_path))
            if frame is not None:
                self._show_bgr_frame(frame)
                return

        self._preview.setText("Preview is not available for this sample")

    def _next_frame(self) -> None:
        if self._capture is None or not self._capture.isOpened():
            return
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
        if ok and frame is not None:
            self._show_bgr_frame(frame)

    def _show_bgr_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self._preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(pixmap)

    def _delete_selected(self) -> None:
        row = self._list.currentRow()
        if not (0 <= row < len(self._samples)):
            return
        sample = self._samples[row]
        label = "GOOD" if sample.label == 0 else "BAD"
        answer = QMessageBox.question(
            self,
            "Delete training sample",
            f"Delete this {label} training video and its saved training data?\n\n{sample.sample_id}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._stop_video()
        self._delete_button.setEnabled(False)
        self.delete_sample_requested.emit(sample.sample_id)

    def _stop_video(self) -> None:
        self._timer.stop()
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._current_video_path = None

    def closeEvent(self, event) -> None:
        self._stop_video()
        super().closeEvent(event)
