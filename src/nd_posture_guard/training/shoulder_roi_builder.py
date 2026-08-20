from __future__ import annotations


class ShoulderRoiBuilder:
    def __init__(self, horizontal_padding_ratio: float, vertical_padding_ratio: float) -> None:
        self._horizontal_padding_ratio = max(0.05, float(horizontal_padding_ratio))
        self._vertical_padding_ratio = max(0.20, float(vertical_padding_ratio))

    def build(
        self,
        points: tuple[tuple[float, float], ...],
        frame_width: int,
        frame_height: int,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        if len(points) != 6:
            raise ValueError("Exactly six shoulder points are required.")
        left = points[:3]
        right = points[3:]
        self._validate(left, right, frame_width, frame_height)

        left_inner, left_mid, left_outer = left
        right_inner, right_mid, right_outer = right
        shoulder_span = max(120.0, right_outer[0] - left_outer[0])
        hpad = shoulder_span * self._horizontal_padding_ratio
        vpad = shoulder_span * self._vertical_padding_ratio

        # Deliberately exclude most of the central neck/face area. The model sees
        # the middle-to-outer shoulder band on each side, not the head.
        left_x1 = left_outer[0] - hpad
        left_x2 = left_mid[0] + (0.60 * hpad)
        right_x1 = right_mid[0] - (0.60 * hpad)
        right_x2 = right_outer[0] + hpad

        shoulder_y_values = [point[1] for point in points]
        top = min(shoulder_y_values) - (0.10 * vpad)
        bottom = max(shoulder_y_values) + vpad

        left_roi = self._clip_rect(left_x1, top, left_x2, bottom, frame_width, frame_height)
        right_roi = self._clip_rect(right_x1, top, right_x2, bottom, frame_width, frame_height)
        if left_roi[2] - left_roi[0] < 40 or right_roi[2] - right_roi[0] < 40:
            raise ValueError("Shoulder regions are too narrow. Mark the shoulder points farther apart.")
        if left_roi[3] - left_roi[1] < 50:
            raise ValueError("Shoulder region is too short vertically. Mark the visible shoulder line again.")
        return left_roi, right_roi

    @staticmethod
    def _validate(
        left: tuple[tuple[float, float], ...],
        right: tuple[tuple[float, float], ...],
        frame_width: int,
        frame_height: int,
    ) -> None:
        for x, y in (*left, *right):
            if not (0 <= x < frame_width and 0 <= y < frame_height):
                raise ValueError("A selected shoulder point is outside the camera image.")

        left_x = [point[0] for point in left]
        right_x = [point[0] for point in right]
        if not (left_x[0] > left_x[1] > left_x[2]):
            raise ValueError("LEFT shoulder points must go from the neck-side outward.")
        if not (right_x[0] < right_x[1] < right_x[2]):
            raise ValueError("RIGHT shoulder points must go from the neck-side outward.")
        if right_x[0] - left_x[0] < frame_width * 0.04:
            raise ValueError("The inner shoulder points are too close together.")

    @staticmethod
    def _clip_rect(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        left = max(0, min(frame_width - 2, round(min(x1, x2))))
        right = max(left + 1, min(frame_width, round(max(x1, x2))))
        top = max(0, min(frame_height - 2, round(min(y1, y2))))
        bottom = max(top + 1, min(frame_height, round(max(y1, y2))))
        return left, top, right, bottom
