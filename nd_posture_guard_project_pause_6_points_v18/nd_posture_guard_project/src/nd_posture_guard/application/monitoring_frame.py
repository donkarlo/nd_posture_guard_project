from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.vision.tracked_shoulders import TrackedShoulders


@dataclass(slots=True)
class MonitoringFrame:
    frame: NDArray[np.uint8]
    shoulders: TrackedShoulders | None
    reference_shoulders: TrackedShoulders | None
    posture_state: str
    shoulder_drop_percent: float
    shoulder_width_change_percent: float
    shoulder_tilt_change_degrees: float
    warning_text: str | None
    tracker_ready: bool
    shoulders_detected: bool
    tracker_confidence: float
    calibration_active: bool
    calibration_current: int
    calibration_target: int
    calibration_message: str
    monitoring_paused: bool
