from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import yaml

from nd_posture_guard.config.app_settings import AppSettings


class SettingsRepository:
    DEFAULT_BAD_SCORE_THRESHOLD = 0.50
    RUNTIME_SCHEMA_VERSION = 2

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
        training = raw.get("training", {})
        monitoring = raw.get("monitoring", {})
        alerts = raw.get("alerts", {})
        data = raw.get("data", {})
        data_root = Path(
            str(data.get("root", "/home/donkarlo/Dropbox/repo/data/nd_posture_guard_project"))
        ).expanduser()

        project_default = self._validated_threshold(
            monitoring.get("bad_score_threshold", self.DEFAULT_BAD_SCORE_THRESHOLD),
            self.DEFAULT_BAD_SCORE_THRESHOLD,
        )
        threshold = self._load_persistent_threshold(data_root, project_default)

        return AppSettings(
            title=str(app.get("title", "ND Posture Guard")),
            warning_text=str(app.get("warning_text", "صاف بشین")),
            camera_device=str(camera.get("device", "auto")),
            camera_width=int(camera.get("width", 640)),
            camera_height=int(camera.get("height", 480)),
            camera_fps=int(camera.get("fps", 30)),
            mirror_camera=bool(camera.get("mirror", True)),
            detection_interval_ms=int(vision.get("detection_interval_ms", 100)),
            training_frames_per_sample=int(training.get("frames_per_sample", 30)),
            training_roi_horizontal_padding_ratio=float(training.get("roi_horizontal_padding_ratio", 0.14)),
            training_roi_vertical_padding_ratio=float(training.get("roi_vertical_padding_ratio", 0.55)),
            training_novelty_multiplier=float(training.get("novelty_multiplier", 4.0)),
            classifier_bad_score_threshold=float(threshold),
            classifier_k_neighbors=int(monitoring.get("k_neighbors", 3)),
            required_bad_frames=int(monitoring.get("required_bad_frames", 5)),
            alert_cooldown_seconds=float(monitoring.get("alert_cooldown_seconds", 3.0)),
            alert_sound_path=str(alerts.get("sound_path", "assets/one_long_beep.wav")),
            alert_volume_percent=int(alerts.get("volume_percent", 100)),
            training_data_root=str(data_root),
        )

    def save_bad_score_threshold(self, value: float) -> None:
        """Persist the GUI threshold atomically outside the application tree.

        The project settings file is intentionally not rewritten. A new ZIP can be
        unpacked over the application without replacing the user's runtime choice.
        """
        value = self._validated_threshold(value, self.DEFAULT_BAD_SCORE_THRESHOLD)
        raw = self._read_raw()
        data_root = Path(
            str(raw.get("data", {}).get("root", "/home/donkarlo/Dropbox/repo/data/nd_posture_guard_project"))
        ).expanduser()
        persistent_path = self._persistent_path(data_root)
        payload = {
            "runtime_schema_version": self.RUNTIME_SCHEMA_VERSION,
            "monitoring": {"bad_score_threshold": float(value)},
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_write_yaml(persistent_path, payload)

    def _load_persistent_threshold(self, data_root: Path, project_default: float) -> float:
        persistent_path = self._persistent_path(data_root)
        if not persistent_path.exists():
            return project_default

        try:
            persistent = self._read_yaml_mapping(persistent_path)
        except (OSError, ValueError, yaml.YAMLError):
            return project_default

        persistent_monitoring = persistent.get("monitoring", {})
        if not isinstance(persistent_monitoring, dict):
            return project_default

        threshold = self._validated_threshold(
            persistent_monitoring.get("bad_score_threshold", project_default),
            project_default,
        )

        # v0.22 shipped 0.62 as an arbitrary default. Old runtime files had no
        # schema marker, so an untouched 0.62 is migrated once to the mathematically
        # meaningful 0.50 GOOD/BAD boundary. Any other old user value is preserved.
        schema = int(persistent.get("runtime_schema_version", 0) or 0)
        if schema < self.RUNTIME_SCHEMA_VERSION and abs(threshold - 0.62) < 1e-9:
            threshold = self.DEFAULT_BAD_SCORE_THRESHOLD
            try:
                self.save_bad_score_threshold(threshold)
            except OSError:
                pass
        return threshold

    def _read_raw(self) -> dict[str, Any]:
        if not self._path.exists():
            raise FileNotFoundError(f"Settings file not found: {self._path}")
        return self._read_yaml_mapping(self._path)

    @staticmethod
    def _persistent_path(data_root: Path) -> Path:
        return data_root / "runtime" / "runtime_settings.yaml"

    @staticmethod
    def _read_yaml_mapping(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a YAML mapping at the top level.")
        return loaded

    @staticmethod
    def _validated_threshold(value: object, fallback: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        if not 0.50 <= numeric <= 0.95:
            return float(fallback)
        return numeric

    @staticmethod
    def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
