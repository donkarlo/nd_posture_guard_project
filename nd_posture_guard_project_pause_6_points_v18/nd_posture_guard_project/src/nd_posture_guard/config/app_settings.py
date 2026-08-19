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
    shoulder_anchor_template_size_px: int
    shoulder_anchor_search_radius_px: int
    shoulder_anchor_recovery_search_radius_px: int
    shoulder_anchor_minimum_match_confidence: float
    shoulder_anchor_minimum_valid_anchors: int
    shoulder_anchor_maximum_group_motion_residual_px: float
    shoulder_anchor_optical_flow_window_px: int
    shoulder_anchor_optical_flow_pyramid_levels: int
    shoulder_anchor_forward_backward_error_px: float
    shoulder_drop_trigger_percent: float
    shoulder_width_trigger_percent: float
    shoulder_tilt_trigger_degrees: float
    required_bad_frames: int
    alert_cooldown_seconds: float
    alert_sound_path: str
    alert_volume_percent: int
