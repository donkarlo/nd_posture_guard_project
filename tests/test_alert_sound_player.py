from pathlib import Path
from types import SimpleNamespace

from nd_posture_guard.alerts.alert_sound_player import AlertSoundPlayer


def test_audio_fallback_continues_after_failed_backend(tmp_path: Path, monkeypatch) -> None:
    sound = tmp_path / "beep.wav"
    sound.write_bytes(b"RIFFdummy")
    player = AlertSoundPlayer(sound, 100)

    monkeypatch.setattr(
        "nd_posture_guard.alerts.alert_sound_player.shutil.which",
        lambda name: f"/usr/bin/{name}" if name in {"paplay", "pw-play"} else None,
    )
    calls: list[str] = []

    def fake_run(command, **kwargs):
        calls.append(command[0])
        return SimpleNamespace(returncode=1 if command[0].endswith("paplay") else 0)

    monkeypatch.setattr("nd_posture_guard.alerts.alert_sound_player.subprocess.run", fake_run)
    player._play_with_fallbacks()
    assert calls == ["/usr/bin/paplay", "/usr/bin/pw-play"]
