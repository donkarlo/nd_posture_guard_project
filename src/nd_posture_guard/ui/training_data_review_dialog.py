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
    """Review saved clips and remove samples without stale-list or reopen side effects."""

    delete_sample_requested = Signal(str)
    closed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review training videos")
        self.resize(1050, 700)
        self._samples: list[TrainingSampleSummary] = []
        self._pending_delete_ids: set[str] = set()
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
        """Replace the list without allowing an in-flight deletion to reappear."""
        selected_id = self._selected_sample_id()
        self._stop_video()

        incoming_ids = {sample.sample_id for sample in samples}
        # A stale worker response can still contain a sample just removed in the UI.
        # Keep hiding that id until a later authoritative refresh proves it is gone.
        self._pending_delete_ids.intersection_update(incoming_ids)
        self._samples = [
            sample for sample in samples if sample.sample_id not in self._pending_delete_ids
        ]

        self._list.blockSignals(True)
        self._list.clear()
        selected_row = -1
        for index, sample in enumerate(self._samples):
            created = sample.created_at_utc.replace("T", " ").replace("+00:00", " UTC")
            suffix = "" if sample.geometry_compatible else "   [legacy 6-point]"
            item = QListWidgetItem(f"{sample.label_name}{suffix}   {created}\n{sample.sample_id}")
            item.setData(Qt.ItemDataRole.UserRole, sample.sample_id)
            item.setToolTip(
                "Usable by the current seven-point geometry model"
                if sample.geometry_compatible
                else "Preserved older sample; reviewable/deletable but not used by the seven-point model"
            )
            self._list.addItem(item)
            if sample.sample_id == selected_id:
                selected_row = index
        self._list.blockSignals(False)

        if self._samples:
            self._list.setCurrentRow(selected_row if selected_row >= 0 else 0)
        else:
            self._show_empty_state()

    def _selected_sample_id(self) -> str | None:
        """Return the currently selected sample id if the row is valid."""
        row = self._list.currentRow()
        if 0 <= row < len(self._samples):
            return self._samples[row].sample_id
        return None

    def _select_row(self, row: int) -> None:
        """Start previewing the selected video or anchor image."""
        self._stop_video()
        if not (0 <= row < len(self._samples)):
            self._delete_button.setEnabled(False)
            return
        sample = self._samples[row]
        self._delete_button.setEnabled(True)
        compatibility = (
            "seven-point geometry sample — USED by current model"
            if sample.geometry_compatible
            else "legacy six-point sample — preserved, NOT used by current model"
        )
        self._details.setText(
            f"Label: {sample.label_name}    Frames: {sample.frame_count}\n"
            f"Saved: {sample.created_at_utc}\n"
            f"Sample: {sample.sample_id}\n{compatibility}"
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
        self._preview.setPixmap(QPixmap())
        self._preview.setText("Preview is not available for this sample")

    def _next_frame(self) -> None:
        """Advance the looping preview by one frame."""
        if self._capture is None or not self._capture.isOpened():
            return
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
        if ok and frame is not None:
            self._show_bgr_frame(frame)

    def _show_bgr_frame(self, frame) -> None:
        """Render one OpenCV BGR frame into the Qt preview label."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self._preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setText("")
        self._preview.setPixmap(pixmap)

    def _delete_selected(self) -> None:
        """Hide the row immediately and suppress stale refreshes while deletion is in flight."""
        row = self._list.currentRow()
        if not (0 <= row < len(self._samples)):
            return
        sample = self._samples[row]
        answer = QMessageBox.question(
            self,
            "Delete training sample",
            f"Delete this {sample.label_name} training video and all data for it?\n\n{sample.sample_id}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._stop_video()
        self._pending_delete_ids.add(sample.sample_id)
        self._samples.pop(row)
        self._list.blockSignals(True)
        self._list.takeItem(row)
        self._list.blockSignals(False)
        if self._samples:
            self._list.setCurrentRow(min(row, len(self._samples) - 1))
        else:
            self._show_empty_state()
        self.delete_sample_requested.emit(sample.sample_id)

    def _show_empty_state(self) -> None:
        """Reset preview controls when no samples remain."""
        self._stop_video()
        self._preview.setPixmap(QPixmap())
        self._preview.setText("No saved training videos")
        self._details.setText("")
        self._delete_button.setEnabled(False)

    def _stop_video(self) -> None:
        """Stop the preview timer and release its native VideoCapture promptly."""
        self._timer.stop()
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._current_video_path = None

    def closeEvent(self, event) -> None:
        """Release preview resources and tell MainWindow that the user closed review."""
        self._stop_video()
        self._pending_delete_ids.clear()
        self.closed.emit()
        super().closeEvent(event)
