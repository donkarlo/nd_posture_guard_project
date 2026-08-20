from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingSampleSummary:
    sample_id: str
    label: int
    label_name: str
    created_at_utc: str
    frame_count: int
    sample_dir: Path
    video_path: Path | None
    anchor_frame_path: Path | None
