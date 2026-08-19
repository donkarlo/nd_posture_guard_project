from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrackedShoulders:
    left: tuple[float, float]
    right: tuple[float, float]
    anchors: tuple[tuple[float, float], ...] = ()
    valid_anchors: tuple[bool, ...] = ()

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.left[0] + self.right[0]) / 2.0,
            (self.left[1] + self.right[1]) / 2.0,
        )

    @property
    def width(self) -> float:
        dx = self.right[0] - self.left[0]
        dy = self.right[1] - self.left[1]
        return (dx * dx + dy * dy) ** 0.5

    @property
    def valid_anchor_count(self) -> int:
        if not self.valid_anchors:
            return len(self.anchors)
        return sum(1 for value in self.valid_anchors if value)
