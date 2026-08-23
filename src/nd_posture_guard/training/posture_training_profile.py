from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class PostureTrainingProfile:
    """Runtime geometry classifier data plus point templates for landmark tracking."""

    reference_points_normalized: tuple[tuple[float, float], ...]
    point_templates: tuple[tuple[NDArray[np.uint8], ...], ...]
    features: NDArray[np.float32]
    labels: NDArray[np.int8]
