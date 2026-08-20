from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AppSettings:
    title: str
    warning_text: str
    camera_device: str
    camera_width: int
    camera_height: int
    camera_fps: int
    mirror_camera: bool
    detection_interval_ms: int
    training_frames_per_sample: int
    training_roi_horizontal_padding_ratio: float
    training_roi_vertical_padding_ratio: float
    training_novelty_multiplier: float
    classifier_bad_score_threshold: float
    classifier_k_neighbors: int
    required_bad_frames: int
    alert_cooldown_seconds: float
    alert_sound_path: str
    alert_volume_percent: int
    training_data_root: str
