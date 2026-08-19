from __future__ import annotations

from nd_posture_guard.posture.posture_reference import PostureReference
from nd_posture_guard.posture.shoulder_calibration_pose import ShoulderCalibrationPose
from nd_posture_guard.posture.shoulder_calibration_profile import ShoulderCalibrationProfile


class ShoulderCalibrationSession:
    STAGES = ("CENTER", "LEFT", "RIGHT")
    POINTS_PER_SHOULDER = 3
    POINTS_PER_POSE = 6

    def __init__(self) -> None:
        self._active = False
        self._stage_index = 0
        self._waiting_for_turn = False
        self._points: list[tuple[float, float]] = []
        self._poses: list[ShoulderCalibrationPose] = []
        self._frame_width = 0
        self._frame_height = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def waiting_for_turn(self) -> bool:
        return self._waiting_for_turn

    @property
    def stage_name(self) -> str:
        return self.STAGES[min(self._stage_index, len(self.STAGES) - 1)]

    @property
    def point_count(self) -> int:
        return len(self._poses) * self.POINTS_PER_POSE + len(self._points)

    @property
    def target_points(self) -> int:
        return len(self.STAGES) * self.POINTS_PER_POSE

    @property
    def current_points(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._points)

    @property
    def complete(self) -> bool:
        return len(self._poses) == len(self.STAGES)

    @property
    def next_point_message(self) -> str:
        index = len(self._points)
        stage = self.stage_name
        prompts = (
            "LEFT shoulder 1/3 — click the point next to the neck.",
            "LEFT shoulder 2/3 — click the middle of the shoulder line.",
            "LEFT shoulder 3/3 — click the outer shoulder tip.",
            "RIGHT shoulder 1/3 — click the point next to the neck.",
            "RIGHT shoulder 2/3 — click the middle of the shoulder line.",
            "RIGHT shoulder 3/3 — click the outer shoulder tip.",
        )
        return f"{stage}: {prompts[min(index, len(prompts) - 1)]}"

    @property
    def poses(self) -> tuple[ShoulderCalibrationPose, ...]:
        return tuple(self._poses)

    def start(self) -> None:
        self.clear()
        self._active = True

    def clear(self) -> None:
        self._active = False
        self._stage_index = 0
        self._waiting_for_turn = False
        self._points = []
        self._poses = []
        self._frame_width = 0
        self._frame_height = 0

    def continue_after_turn(self) -> bool:
        if not self._active or not self._waiting_for_turn or self.complete:
            return False
        self._waiting_for_turn = False
        self._points = []
        return True

    def add_point(
        self,
        point: tuple[float, float],
        frame_width: int,
        frame_height: int,
    ) -> ShoulderCalibrationPose | None:
        if not self._active or self._waiting_for_turn or self.complete:
            return None
        x, y = point
        if not (0 <= x < frame_width and 0 <= y < frame_height):
            return None
        self._frame_width = int(frame_width)
        self._frame_height = int(frame_height)
        candidate_points = [*self._points, (float(x), float(y))]
        if len(candidate_points) < self.POINTS_PER_POSE:
            self._points = candidate_points
            return None

        left_points = tuple(candidate_points[:3])
        right_points = tuple(candidate_points[3:6])
        assert len(left_points) == 3 and len(right_points) == 3
        pose = ShoulderCalibrationPose(
            left_points=(left_points[0], left_points[1], left_points[2]),
            right_points=(right_points[0], right_points[1], right_points[2]),
        )
        try:
            self._validate_pose(pose, frame_width, frame_height)
        except ValueError:
            # Never leave six rejected clicks stuck in the session.
            # The current stage can only restart from point 1/6.
            self._points = []
            raise
        self._poses.append(pose)
        self._points = []
        if len(self._poses) < len(self.STAGES):
            self._stage_index += 1
            self._waiting_for_turn = True
        return pose

    def build_reference(self) -> PostureReference:
        if not self.complete:
            raise RuntimeError("Calibration needs CENTER, slight LEFT, and slight RIGHT six-point shoulder poses.")
        profiles = tuple(
            ShoulderCalibrationProfile.from_shoulders(
                pose.shoulders,
                self._frame_width,
                self._frame_height,
            )
            for pose in self._poses
        )
        return PostureReference(
            shoulders=self._poses[0].shoulders,
            profiles=profiles,
            frame_width=self._frame_width,
            frame_height=self._frame_height,
        )

    @staticmethod
    def _validate_pose(
        pose: ShoulderCalibrationPose,
        frame_width: int,
        frame_height: int,
    ) -> None:
        shoulders = pose.shoulders
        minimum_width = max(70.0, frame_width * 0.12)
        if shoulders.width < minimum_width:
            raise ValueError("The two shoulders are too close together. Mark both shoulder lines again.")

        left_x = [point[0] for point in pose.left_points]
        right_x = [point[0] for point in pose.right_points]
        minimum_span = max(12.0, frame_width * 0.018)
        if left_x[0] - left_x[2] < minimum_span:
            raise ValueError(
                "On the LEFT shoulder, click from the neck outward: neck-side point, middle point, then outer tip."
            )
        if right_x[2] - right_x[0] < minimum_span:
            raise ValueError(
                "On the RIGHT shoulder, click from the neck outward: neck-side point, middle point, then outer tip."
            )

        max_vertical_span = frame_height * 0.18
        for points, name in ((pose.left_points, "LEFT"), (pose.right_points, "RIGHT")):
            ys = [point[1] for point in points]
            if max(ys) - min(ys) > max_vertical_span:
                raise ValueError(f"The three {name} shoulder points are too far apart vertically. Please mark the shoulder line again.")
