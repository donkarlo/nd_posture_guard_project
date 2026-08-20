from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class ShoulderTrainingProfile:
    left_roi: tuple[int, int, int, int]
    right_roi: tuple[int, int, int, int]
    features: NDArray[np.float32]
    labels: NDArray[np.int8]
    novelty_threshold: float
