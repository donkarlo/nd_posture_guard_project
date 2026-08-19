from __future__ import annotations

from dataclasses import dataclass

from nd_posture_guard.posture.shoulder_calibration_profile import ShoulderCalibrationProfile
from nd_posture_guard.vision.tracked_shoulders import TrackedShoulders


@dataclass(frozen=True, slots=True)
class PostureReference:
    shoulders: TrackedShoulders
    profiles: tuple[ShoulderCalibrationProfile, ...]
    frame_width: int
    frame_height: int
