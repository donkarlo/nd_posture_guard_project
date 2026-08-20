from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PostureClassification:
    state: str
    bad_score: float
    confidence: float
    nearest_distance: float
    should_alert: bool
