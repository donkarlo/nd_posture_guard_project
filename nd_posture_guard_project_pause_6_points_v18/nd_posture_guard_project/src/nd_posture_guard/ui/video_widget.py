from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPaintEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from nd_posture_guard.vision.tracked_shoulders import TrackedShoulders


class VideoWidget(QWidget):
    shoulder_point_clicked = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 480)
        self._image: QImage | None = None
        self._frame_width = 1
        self._frame_height = 1
        self._shoulders: TrackedShoulders | None = None
        self._reference_shoulders: TrackedShoulders | None = None
        self._posture_state = "not_calibrated"
        self._shoulder_drop_percent = 0.0
        self._shoulder_width_change_percent = 0.0
        self._shoulder_tilt_change_degrees = 0.0
        self._warning_text: str | None = None
        self._tracker_ready = False
        self._shoulders_detected = False
        self._tracker_confidence = 0.0
        self._calibration_active = False
        self._calibration_current = 0
        self._calibration_target = 0
        self._calibration_message = ""
        self._forced_overlay: str | None = "NOT CALIBRATED"
        self._selection_enabled = False
        self._selection_target = 2
        self._selection_points: list[tuple[float, float]] = []

    def set_calibration_overlay(self, text: str | None) -> None:
        self._forced_overlay = text
        self.update()

    def set_selection_enabled(self, enabled: bool, target_points: int = 2) -> None:
        self._selection_enabled = enabled
        self._selection_target = max(1, int(target_points))
        if enabled:
            self._selection_points = []
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        self.update()

    def clear_selection_points(self) -> None:
        self._selection_points = []
        self.update()

    def set_frame(
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        self._image = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()
        self._frame_width = width
        self._frame_height = height
        self._shoulders = shoulders
        self._reference_shoulders = reference_shoulders
        self._posture_state = posture_state
        self._shoulder_drop_percent = shoulder_drop_percent
        self._shoulder_width_change_percent = shoulder_width_change_percent
        self._shoulder_tilt_change_degrees = shoulder_tilt_change_degrees
        self._warning_text = warning_text
        self._tracker_ready = tracker_ready
        self._shoulders_detected = shoulders_detected
        self._tracker_confidence = tracker_confidence
        self._calibration_active = calibration_active
        self._calibration_current = calibration_current
        self._calibration_target = calibration_target
        self._calibration_message = calibration_message
        if calibration_active:
            self._forced_overlay = None
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._selection_enabled or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        draw_rect = self._image_rect()
        if draw_rect is None or not draw_rect.contains(event.position().toPoint()):
            return
        position = event.position()
        x = (position.x() - draw_rect.left()) / max(draw_rect.width(), 1) * self._frame_width
        y = (position.y() - draw_rect.top()) / max(draw_rect.height(), 1) * self._frame_height
        point = (float(x), float(y))
        self._selection_points.append(point)
        self.shoulder_point_clicked.emit(point[0], point[1])
        if len(self._selection_points) >= self._selection_target:
            self._selection_enabled = False
            self.unsetCursor()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        draw_rect = self._image_rect()
        if self._image is None or draw_rect is None:
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for camera...")
            return
        painter.drawImage(draw_rect, self._image)
        self._draw_reference(painter, draw_rect)
        self._draw_current(painter, draw_rect)
        self._draw_selected_points(painter, draw_rect)
        self._draw_diagnostics(painter, draw_rect)
        self._draw_calibration_overlay(painter, draw_rect)
        self._draw_metrics(painter, draw_rect)
        if self._warning_text:
            painter.setPen(QColor(255, 70, 70))
            font = painter.font()
            font.setPointSize(28)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                draw_rect.adjusted(0, 14, 0, 0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                self._warning_text,
            )

    def _draw_reference(self, painter: QPainter, draw_rect: QRect) -> None:
        if self._reference_shoulders is None:
            return
        color = QColor(70, 220, 100)
        painter.save()
        painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
        anchor_count = len(self._reference_shoulders.anchors)
        if anchor_count >= 4 and anchor_count % 2 == 0:
            side = anchor_count // 2
            valid = (True,) * anchor_count
            self._draw_anchor_side(
                painter, draw_rect, self._reference_shoulders.anchors[:side], valid[:side], color, 6
            )
            self._draw_anchor_side(
                painter, draw_rect, self._reference_shoulders.anchors[side:], valid[side:], color, 6
            )
        else:
            left = self._to_widget(draw_rect, self._reference_shoulders.left)
            right = self._to_widget(draw_rect, self._reference_shoulders.right)
            painter.drawLine(left, right)
            painter.drawEllipse(left, 7, 7)
            painter.drawEllipse(right, 7, 7)
        painter.restore()

    def _draw_current(self, painter: QPainter, draw_rect: QRect) -> None:
        if self._shoulders is None:
            return
        color = QColor(245, 70, 70) if self._posture_state == "slouch" else QColor(80, 220, 220)
        painter.save()
        painter.setPen(QPen(color, 3))
        anchor_count = len(self._shoulders.anchors)
        if anchor_count >= 4 and anchor_count % 2 == 0:
            side = anchor_count // 2
            valid = self._shoulders.valid_anchors or ((True,) * anchor_count)
            self._draw_anchor_side(painter, draw_rect, self._shoulders.anchors[:side], valid[:side], color, 7)
            self._draw_anchor_side(painter, draw_rect, self._shoulders.anchors[side:], valid[side:], color, 7)
        else:
            left = self._to_widget(draw_rect, self._shoulders.left)
            right = self._to_widget(draw_rect, self._shoulders.right)
            painter.drawLine(left, right)
            painter.drawEllipse(left, 8, 8)
            painter.drawEllipse(right, 8, 8)
        painter.restore()

    def _draw_anchor_side(
        self,
        painter: QPainter,
        draw_rect: QRect,
        anchors: tuple[tuple[float, float], ...],
        valid: tuple[bool, ...],
        color: QColor,
        radius: int,
    ) -> None:
        widget_points = [self._to_widget(draw_rect, point) for point in anchors]
        valid_points = [point for point, is_valid in zip(widget_points, valid) if is_valid]
        painter.setPen(QPen(color, 3))
        if len(valid_points) >= 2:
            for first, second in zip(valid_points, valid_points[1:]):
                painter.drawLine(first, second)
        for point, is_valid in zip(widget_points, valid):
            if is_valid:
                painter.setPen(QPen(color, 3))
            else:
                painter.setPen(QPen(QColor(150, 150, 150), 2, Qt.PenStyle.DotLine))
            painter.drawEllipse(point, radius, radius)

    def _draw_selected_points(self, painter: QPainter, draw_rect: QRect) -> None:
        if not self._selection_points:
            return
        painter.save()
        painter.setPen(QPen(QColor(255, 80, 220), 3))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(11)
        painter.setFont(font)
        for index, point in enumerate(self._selection_points, start=1):
            widget_point = self._to_widget(draw_rect, point)
            painter.drawEllipse(widget_point, 9, 9)
            painter.drawText(widget_point + QPoint(12, -10), str(index))
        painter.restore()

    def _draw_diagnostics(self, painter: QPainter, draw_rect: QRect) -> None:
        labels = [("TRACKER", self._tracker_ready), ("SHOULDERS", self._shoulders_detected)]
        x = draw_rect.right() - 245
        y = draw_rect.top() + 18
        painter.save()
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        for name, ok in labels:
            width = 110 if name == "TRACKER" else 125
            rect = QRect(x, y, width, 26)
            painter.fillRect(rect, QColor(20, 120, 55, 210) if ok else QColor(150, 45, 45, 210))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{name}: {'OK' if ok else 'NO'}")
            x += width + 5
        if self._tracker_ready and not self._calibration_active:
            confidence_rect = QRect(draw_rect.right() - 245, y + 30, 240, 22)
            painter.fillRect(confidence_rect, QColor(20, 20, 20, 165))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                confidence_rect,
                Qt.AlignmentFlag.AlignCenter,
                f"match confidence: {self._tracker_confidence:.2f}",
            )
        painter.restore()

    def _draw_calibration_overlay(self, painter: QPainter, draw_rect: QRect) -> None:
        if self._calibration_active:
            title = f"MANUAL SHOULDER CALIBRATION {self._calibration_current}/{self._calibration_target}"
            detail = self._calibration_message
            background = QColor(246, 195, 68, 225)
            foreground = QColor(15, 15, 15)
        elif self._forced_overlay:
            parts = self._forced_overlay.split("\n", 1)
            title = parts[0]
            detail = parts[1] if len(parts) > 1 else ""
            if title == "CALIBRATED":
                background = QColor(22, 128, 58, 220)
            elif "FAILED" in title:
                background = QColor(155, 44, 44, 225)
            else:
                background = QColor(40, 40, 40, 210)
            foreground = QColor(255, 255, 255)
        else:
            return
        box = QRect(draw_rect.left() + 18, draw_rect.top() + 18, min(790, draw_rect.width() - 36), 104)
        painter.fillRect(box, background)
        painter.setPen(foreground)
        font = painter.font()
        font.setBold(True)
        font.setPointSize(17)
        painter.setFont(font)
        painter.drawText(box.adjusted(12, 6, -12, -52), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
        if detail:
            font.setPointSize(10)
            font.setBold(False)
            painter.setFont(font)
            painter.drawText(
                box.adjusted(12, 42, -12, -7),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
                detail,
            )

    def _draw_metrics(self, painter: QPainter, draw_rect: QRect) -> None:
        if self._posture_state == "not_calibrated":
            return
        painter.save()
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QColor(245, 245, 245))
        text = (
            f"Shoulder drop {self._shoulder_drop_percent:+.1f}%   "
            f"Width change {self._shoulder_width_change_percent:+.1f}%   "
            f"Tilt change {self._shoulder_tilt_change_degrees:.1f}°"
        )
        painter.drawText(
            draw_rect.adjusted(12, 0, -12, -10),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            text,
        )
        painter.restore()

    def _to_widget(self, draw_rect: QRect, point: tuple[float, float]) -> QPoint:
        return QPoint(
            draw_rect.left() + round(point[0] / max(self._frame_width, 1) * draw_rect.width()),
            draw_rect.top() + round(point[1] / max(self._frame_height, 1) * draw_rect.height()),
        )

    def _image_rect(self) -> QRect | None:
        if self._image is None or self._image.width() <= 0 or self._image.height() <= 0:
            return None
        image_ratio = self._image.width() / self._image.height()
        widget_ratio = self.width() / max(self.height(), 1)
        if widget_ratio > image_ratio:
            height = self.height()
            width = round(height * image_ratio)
            return QRect((self.width() - width) // 2, 0, width, height)
        width = self.width()
        height = round(width / image_ratio)
        return QRect(0, (self.height() - height) // 2, width, height)
