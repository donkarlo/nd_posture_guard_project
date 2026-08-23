from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPaintEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget


class VideoWidget(QWidget):
    """Display the camera image, seven-point geometry and interactive training overlays."""

    geometry_point_clicked = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 480)
        self._image: QImage | None = None
        self._frame_width = 1
        self._frame_height = 1
        self._selection_enabled = False
        self._selection_target = 0
        self._selection_points: list[tuple[float, float]] = []
        self._geometry_points: tuple[tuple[float, float], ...] | None = None
        self._tracking_confidence = 0.0
        self._posture_state = "not_trained"
        self._bad_score = 0.0
        self._confidence = 0.0
        self._warning_text: str | None = None
        self._training_active = False
        self._training_current = 0
        self._training_total = 0
        self._training_message = ""
        self._forced_overlay = ""

    def set_frame(
        self,
        frame: NDArray[np.uint8],
        geometry_points: tuple[tuple[float, float], ...] | None,
        tracking_confidence: float,
        posture_state: str,
        bad_score: float,
        confidence: float,
        warning_text: str | None,
        training_active: bool,
        training_current: int,
        training_total: int,
        training_message: str,
    ) -> None:
        """Replace the displayed frame and its posture/geometry metadata."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        self._image = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()
        self._frame_width = width
        self._frame_height = height
        self._geometry_points = geometry_points
        self._tracking_confidence = float(tracking_confidence)
        self._posture_state = posture_state
        self._bad_score = float(bad_score)
        self._confidence = float(confidence)
        self._warning_text = warning_text
        self._training_active = training_active
        self._training_current = training_current
        self._training_total = training_total
        self._training_message = training_message
        self.update()

    @property
    def selection_enabled(self) -> bool:
        """Return whether landmark clicks are currently accepted."""
        return self._selection_enabled

    def set_selection_enabled(self, enabled: bool, target: int = 7) -> None:
        """Enable or disable seven-point landmark selection."""
        was_enabled = self._selection_enabled
        self._selection_enabled = bool(enabled)
        self._selection_target = int(target)
        if enabled:
            if not was_enabled:
                self._selection_points = []
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        self.update()

    def clear_selection_points(self) -> None:
        """Remove all temporary training clicks."""
        self._selection_points = []
        self.update()

    def set_overlay(self, text: str) -> None:
        """Set a non-training informational overlay."""
        self._forced_overlay = text
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Translate a click from widget space into camera-frame coordinates."""
        if not self._selection_enabled or event.button() != Qt.MouseButton.LeftButton:
            return
        draw_rect = self._image_rect()
        if draw_rect is None or not draw_rect.contains(event.position().toPoint()):
            return
        position = event.position()
        x = (position.x() - draw_rect.left()) / max(draw_rect.width(), 1) * self._frame_width
        y = (position.y() - draw_rect.top()) / max(draw_rect.height(), 1) * self._frame_height
        point = (float(x), float(y))
        self._selection_points.append(point)
        self.geometry_point_clicked.emit(point[0], point[1])
        if len(self._selection_points) >= self._selection_target:
            self._selection_enabled = False
            self.unsetCursor()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint camera, tracked geometry, training geometry and status overlays."""
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        draw_rect = self._image_rect()
        if self._image is None or draw_rect is None:
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for camera...")
            return
        painter.drawImage(draw_rect, self._image)
        self._draw_tracked_geometry(painter, draw_rect)
        self._draw_selected_geometry(painter, draw_rect)
        self._draw_state(painter, draw_rect)
        self._draw_overlay(painter, draw_rect)
        if self._warning_text:
            painter.setPen(QColor(255, 70, 70))
            font = painter.font()
            font.setPointSize(30)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                draw_rect.adjusted(0, 14, 0, 0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                self._warning_text,
            )

    def _draw_tracked_geometry(self, painter: QPainter, draw_rect: QRect) -> None:
        """Draw the best face triangle and both shoulder lines used by classification."""
        if self._geometry_points is None or len(self._geometry_points) != 7 or self._training_active:
            return
        if self._tracking_confidence < 0.32:
            color = QColor(235, 170, 45)
        elif self._posture_state == "bad":
            color = QColor(240, 70, 70)
        else:
            color = QColor(75, 225, 125)
        self._draw_geometry(painter, draw_rect, self._geometry_points, color, 3, True)

    def _draw_selected_geometry(self, painter: QPainter, draw_rect: QRect) -> None:
        """Draw the triangle/lines progressively while the user clicks training landmarks."""
        if not self._selection_points:
            return
        painter.save()
        color = QColor(255, 80, 220)
        points = self._selection_points
        mapped = [self._to_widget(draw_rect, point) for point in points]
        painter.setPen(QPen(color, 3))
        if len(mapped) >= 2:
            painter.drawLine(mapped[0], mapped[1])
        if len(mapped) >= 3:
            painter.drawLine(mapped[1], mapped[2])
            painter.drawLine(mapped[2], mapped[0])
        if len(mapped) >= 5:
            painter.drawLine(mapped[3], mapped[4])
        if len(mapped) >= 7:
            painter.drawLine(mapped[5], mapped[6])
        font = painter.font()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        for index, widget_point in enumerate(mapped, start=1):
            painter.drawEllipse(widget_point, 7, 7)
            painter.drawText(widget_point + QPoint(9, -7), str(index))
        painter.restore()

    def _draw_geometry(
        self,
        painter: QPainter,
        draw_rect: QRect,
        points: tuple[tuple[float, float], ...],
        color: QColor,
        width: int,
        show_points: bool,
    ) -> None:
        """Render one complete 3-point face triangle and two 2-point shoulder segments."""
        mapped = [self._to_widget(draw_rect, point) for point in points]
        painter.save()
        painter.setPen(QPen(color, width))
        painter.drawLine(mapped[0], mapped[1])
        painter.drawLine(mapped[1], mapped[2])
        painter.drawLine(mapped[2], mapped[0])
        painter.drawLine(mapped[3], mapped[4])
        painter.drawLine(mapped[5], mapped[6])
        if show_points:
            for point in mapped:
                painter.drawEllipse(point, 5, 5)
        painter.restore()

    def _draw_state(self, painter: QPainter, draw_rect: QRect) -> None:
        """Draw classification plus tracker confidence in the upper-right corner."""
        if self._training_active:
            return
        if self._posture_state == "bad":
            text, color = f"BAD {self._bad_score * 100:.0f}%", QColor(160, 40, 40, 220)
        elif self._posture_state == "good":
            text, color = f"GOOD {self._bad_score * 100:.0f}% bad", QColor(20, 120, 55, 220)
        elif self._posture_state == "unknown":
            text, color = "TRACKING UNCERTAIN — no beep", QColor(190, 125, 0, 220)
        elif self._posture_state == "paused":
            text, color = "PAUSED — no beep", QColor(190, 125, 0, 220)
        else:
            text, color = "GEOMETRY MODEL NOT TRAINED", QColor(150, 45, 45, 220)
        box = QRect(draw_rect.right() - 300, draw_rect.top() + 18, 285, 30)
        painter.fillRect(box, color)
        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        if self._posture_state in {"good", "bad", "unknown", "paused"}:
            small = QRect(draw_rect.right() - 300, draw_rect.top() + 52, 285, 22)
            painter.fillRect(small, QColor(20, 20, 20, 170))
            painter.drawText(
                small,
                Qt.AlignmentFlag.AlignCenter,
                f"tracking {self._tracking_confidence:.2f}   class confidence {self._confidence:.2f}",
            )

    def _draw_overlay(self, painter: QPainter, draw_rect: QRect) -> None:
        """Draw training instructions or the latest model message."""
        text = self._training_message if self._training_active else self._forced_overlay
        if not text:
            return
        box = QRect(draw_rect.left() + 18, draw_rect.top() + 18, min(800, draw_rect.width() - 36), 105)
        painter.fillRect(
            box,
            QColor(246, 195, 68, 225) if self._training_active else QColor(30, 30, 30, 210),
        )
        painter.setPen(
            QColor(20, 20, 20) if self._training_active else QColor(255, 255, 255)
        )
        font = painter.font()
        font.setBold(True)
        font.setPointSize(13)
        painter.setFont(font)
        title = (
            f"TRAINING LANDMARKS {self._training_current}/{self._training_total}"
            if self._training_active
            else "POSTURE MODEL"
        )
        painter.drawText(
            box.adjusted(12, 5, -12, -58),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            title,
        )
        font.setBold(False)
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(
            box.adjusted(12, 38, -12, -7),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
            text,
        )

    def _to_widget(self, draw_rect: QRect, point: tuple[float, float]) -> QPoint:
        """Map one camera-space point to the aspect-fit image rectangle."""
        return QPoint(
            draw_rect.left() + round(point[0] / max(self._frame_width, 1) * draw_rect.width()),
            draw_rect.top() + round(point[1] / max(self._frame_height, 1) * draw_rect.height()),
        )

    def _image_rect(self) -> QRect | None:
        """Return the aspect-preserving rectangle occupied by the camera image."""
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
