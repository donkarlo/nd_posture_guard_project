from pathlib import Path

from nd_posture_guard.config.settings_repository import SettingsRepository


def test_bad_threshold_is_persisted_outside_project(tmp_path: Path) -> None:
    data_root = tmp_path / "persistent-data"
    path = tmp_path / "settings.yaml"
    path.write_text(
        f'data:\n  root: "{data_root}"\nmonitoring:\n  bad_score_threshold: 0.62\n',
        encoding="utf-8",
    )
    repository = SettingsRepository(path)
    repository.save_bad_score_threshold(0.74)
    assert abs(repository.load().classifier_bad_score_threshold - 0.74) < 1e-9
    assert (data_root / "runtime" / "runtime_settings.yaml").is_file()


def test_persistent_threshold_overrides_new_project_default(tmp_path: Path) -> None:
    data_root = tmp_path / "persistent-data"
    path = tmp_path / "settings.yaml"
    path.write_text(
        f'data:\n  root: "{data_root}"\nmonitoring:\n  bad_score_threshold: 0.60\n',
        encoding="utf-8",
    )
    first = SettingsRepository(path)
    first.save_bad_score_threshold(0.81)
    # Simulate replacing the project with a newer settings.yaml default.
    path.write_text(
        f'data:\n  root: "{data_root}"\nmonitoring:\n  bad_score_threshold: 0.55\n',
        encoding="utf-8",
    )
    assert abs(SettingsRepository(path).load().classifier_bad_score_threshold - 0.81) < 1e-9


def test_legacy_62_default_is_migrated_to_natural_50_boundary(tmp_path: Path) -> None:
    data_root = tmp_path / "persistent-data"
    path = tmp_path / "settings.yaml"
    path.write_text(
        f'data:\n  root: "{data_root}"\nmonitoring:\n  bad_score_threshold: 0.50\n',
        encoding="utf-8",
    )
    runtime = data_root / "runtime" / "runtime_settings.yaml"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("monitoring:\n  bad_score_threshold: 0.62\n", encoding="utf-8")

    loaded = SettingsRepository(path).load()
    assert abs(loaded.classifier_bad_score_threshold - 0.50) < 1e-9
    text = runtime.read_text(encoding="utf-8")
    assert "runtime_schema_version: 2" in text
    assert "bad_score_threshold: 0.5" in text


def test_project_settings_file_is_not_rewritten_when_threshold_changes(tmp_path: Path) -> None:
    data_root = tmp_path / "persistent-data"
    path = tmp_path / "settings.yaml"
    original = f'data:\n  root: "{data_root}"\nmonitoring:\n  bad_score_threshold: 0.50\n'
    path.write_text(original, encoding="utf-8")
    SettingsRepository(path).save_bad_score_threshold(0.57)
    assert path.read_text(encoding="utf-8") == original
    assert abs(SettingsRepository(path).load().classifier_bad_score_threshold - 0.57) < 1e-9
