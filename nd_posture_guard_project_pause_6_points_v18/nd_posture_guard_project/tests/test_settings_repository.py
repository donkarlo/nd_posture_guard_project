from pathlib import Path

from nd_posture_guard.config.settings_repository import SettingsRepository


def test_shoulder_drop_tolerance_is_persisted(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        """
app: {title: ND Posture Guard}
camera: {}
vision: {}
monitoring:
  shoulder_drop_trigger_percent: 7.0
alerts: {}
""",
        encoding="utf-8",
    )
    repository = SettingsRepository(settings_path)
    repository.save_shoulder_drop_trigger_percent(4.5)
    assert repository.load().shoulder_drop_trigger_percent == 4.5
