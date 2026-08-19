from __future__ import annotations

from nd_posture_guard.posture.posture_evaluation import PostureEvaluation
from nd_posture_guard.posture.posture_reference import PostureReference
from nd_posture_guard.posture.shoulder_calibration_profile import ShoulderCalibrationProfile
from nd_posture_guard.vision.tracked_shoulders import TrackedShoulders


class PostureEvaluator:
    STATE_NOT_CALIBRATED = "not_calibrated"
    STATE_OK = "ok"
    STATE_SLOUCH = "slouch"
    STATE_POSE_LOST = "pose_lost"

    def __init__(
        self,
        shoulder_drop_trigger_percent: float,
        shoulder_width_trigger_percent: float,
        shoulder_tilt_trigger_degrees: float,
        required_bad_frames: int,
    ) -> None:
        self._shoulder_drop_trigger_percent = max(1.0, float(shoulder_drop_trigger_percent))
        self._shoulder_width_trigger_percent = max(1.0, float(shoulder_width_trigger_percent))
        self._shoulder_tilt_trigger_degrees = max(1.0, float(shoulder_tilt_trigger_degrees))
        self._required_bad_frames = max(1, int(required_bad_frames))
        self._reference: PostureReference | None = None
        self._bad_frames = 0

    @property
    def calibrated(self) -> bool:
        return self._reference is not None

    @property
    def reference(self) -> PostureReference | None:
        return self._reference

    @property
    def shoulder_drop_trigger_percent(self) -> float:
        return self._shoulder_drop_trigger_percent

    def set_shoulder_drop_trigger_percent(self, value: float) -> None:
        self._shoulder_drop_trigger_percent = max(1.0, min(float(value), 40.0))
        self._bad_frames = 0

    def calibrate(self, reference: PostureReference) -> None:
        if reference.shoulders.width < 55.0:
            raise ValueError("Shoulders are too close together for reliable calibration.")
        if len(reference.profiles) < 3:
            raise ValueError("Center, slight-left, and slight-right calibration are required.")
        self._reference = reference
        self._bad_frames = 0

    def clear(self) -> None:
        self._reference = None
        self._bad_frames = 0

    def reset_tracking(self) -> None:
        self._bad_frames = 0

    def evaluate(
        self,
        shoulders: TrackedShoulders | None,
        frame_width: int,
        frame_height: int,
    ) -> PostureEvaluation:
        if self._reference is None:
            return self._empty(self.STATE_NOT_CALIBRATED)
        if shoulders is None:
            self._bad_frames = 0
            return self._empty(self.STATE_POSE_LOST)

        current = ShoulderCalibrationProfile.from_shoulders(
            shoulders, frame_width, frame_height
        )
        matched = self._closest_profile(current)

        reference_width_pixels = max(matched.width_ratio * frame_width, 1.0)
        current_center_y = current.center_y_ratio * frame_height
        reference_center_y = matched.center_y_ratio * frame_height
        shoulder_drop = (current_center_y - reference_center_y) / reference_width_pixels * 100.0
        width_change = (matched.width_ratio - current.width_ratio) / max(matched.width_ratio, 1e-6) * 100.0
        tilt_change = abs(current.tilt_degrees - matched.tilt_degrees)

        strong_drop = shoulder_drop >= self._shoulder_drop_trigger_percent
        moderate_drop = shoulder_drop >= self._shoulder_drop_trigger_percent * 0.55
        geometry_changed = (
            abs(width_change) >= self._shoulder_width_trigger_percent
            or tilt_change >= self._shoulder_tilt_trigger_degrees
        )
        slouch = strong_drop or (moderate_drop and geometry_changed)

        if slouch:
            self._bad_frames += 1
            state = self.STATE_SLOUCH
        else:
            self._bad_frames = 0
            state = self.STATE_OK

        return PostureEvaluation(
            state=state,
            should_alert=state == self.STATE_SLOUCH and self._bad_frames >= self._required_bad_frames,
            shoulder_drop_percent=shoulder_drop,
            shoulder_width_change_percent=width_change,
            shoulder_tilt_change_degrees=tilt_change,
        )

    def _closest_profile(self, current: ShoulderCalibrationProfile) -> ShoulderCalibrationProfile:
        assert self._reference is not None

        def score(profile: ShoulderCalibrationProfile) -> float:
            width_change = abs(current.width_ratio - profile.width_ratio) / max(profile.width_ratio, 1e-6)
            tilt_change = abs(current.tilt_degrees - profile.tilt_degrees) / 12.0
            return width_change + tilt_change

        return min(self._reference.profiles, key=score)

    @staticmethod
    def _empty(state: str) -> PostureEvaluation:
        return PostureEvaluation(
            state=state,
            should_alert=False,
            shoulder_drop_percent=0.0,
            shoulder_width_change_percent=0.0,
            shoulder_tilt_change_degrees=0.0,
        )
