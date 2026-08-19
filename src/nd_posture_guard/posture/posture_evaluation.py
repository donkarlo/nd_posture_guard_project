from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PostureEvaluation:
    state: str
    should_alert: bool
    shoulder_drop_percent: float
    shoulder_width_change_percent: float
    shoulder_tilt_change_degrees: float
