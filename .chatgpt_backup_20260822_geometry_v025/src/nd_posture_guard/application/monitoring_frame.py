from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class MonitoringFrame:
    frame: NDArray[np.uint8]
    posture_state: str
    bad_score: float
    confidence: float
    nearest_distance: float
    warning_text: str | None
    model_ready: bool
    left_roi: tuple[int, int, int, int] | None
    right_roi: tuple[int, int, int, int] | None
    training_active: bool
    training_stage_current: int
    training_stage_total: int
    training_message: str
    monitoring_paused: bool
