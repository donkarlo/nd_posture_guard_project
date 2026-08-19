from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nd_posture_guard.config.app_settings import AppSettings


class SettingsRepository:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> AppSettings:
        raw = self._read_raw()
        app = raw.get("app", {})
        camera = raw.get("camera", {})
        vision = raw.get("vision", {})
        monitoring = raw.get("monitoring", {})
        alerts = raw.get("alerts", {})
        return AppSettings(
            title=str(app.get("title", "ND Posture Guard")),
            warning_text=str(app.get("warning_text", "صاف بشین")),
            camera_device=str(camera.get("device", "auto")),
            camera_width=int(camera.get("width", 640)),
            camera_height=int(camera.get("height", 480)),
            camera_fps=int(camera.get("fps", 30)),
            mirror_camera=bool(camera.get("mirror", True)),
            detection_interval_ms=int(vision.get("detection_interval_ms", 100)),
            shoulder_anchor_template_size_px=int(vision.get("shoulder_anchor_template_size_px", 25)),
            shoulder_anchor_search_radius_px=int(vision.get("shoulder_anchor_search_radius_px", 42)),
            shoulder_anchor_minimum_match_confidence=float(
                vision.get("shoulder_anchor_minimum_match_confidence", 0.45)
            ),
            shoulder_anchor_minimum_valid_anchors=int(
                vision.get("shoulder_anchor_minimum_valid_anchors", 4)
            ),
            shoulder_anchor_maximum_group_motion_residual_px=float(
                vision.get("shoulder_anchor_maximum_group_motion_residual_px", 16.0)
            ),
            shoulder_drop_trigger_percent=float(monitoring.get("shoulder_drop_trigger_percent", 7.0)),
            shoulder_width_trigger_percent=float(monitoring.get("shoulder_width_trigger_percent", 12.0)),
            shoulder_tilt_trigger_degrees=float(monitoring.get("shoulder_tilt_trigger_degrees", 12.0)),
            required_bad_frames=int(monitoring.get("required_bad_frames", 5)),
            alert_cooldown_seconds=float(monitoring.get("alert_cooldown_seconds", 3.0)),
            alert_sound_path=str(alerts.get("sound_path", "assets/saf_beshin.wav")),
            alert_volume_percent=int(alerts.get("volume_percent", 120)),
        )

    def save_shoulder_drop_trigger_percent(self, value: float) -> None:
        raw = self._read_raw()
        raw.setdefault("monitoring", {})["shoulder_drop_trigger_percent"] = float(value)
        self._write_raw(raw)

    def _read_raw(self) -> dict[str, Any]:
        if not self._path.exists():
            raise FileNotFoundError(f"Settings file not found: {self._path}")
        with self._path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("settings.yaml must contain a YAML mapping at the top level.")
        return loaded

    def _write_raw(self, raw: dict[str, Any]) -> None:
        with self._path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle, allow_unicode=True, sort_keys=False)
