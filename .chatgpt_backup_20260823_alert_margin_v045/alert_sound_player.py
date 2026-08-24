from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from threading import Lock, Thread


class AlertSoundPlayer:
    """Play the warning WAV through the first audio backend that actually succeeds.

    Older versions treated a successful ``Popen`` as successful playback. On Ubuntu
    a present-but-broken ``paplay`` can exit immediately, which prevented all later
    fallbacks from being attempted. This implementation checks the real return code
    in a background thread and then tries PipeWire, ALSA, and ffplay in order.
    """

    def __init__(self, sound_path: Path, volume_percent: int) -> None:
        self._sound_path = sound_path
        self._volume_percent = max(0, min(int(volume_percent), 150))
        self._lock = Lock()
        self._playing = False

    @property
    def sound_path(self) -> Path:
        return self._sound_path

    def play(self) -> bool:
        if not self._sound_path.is_file():
            return False
        with self._lock:
            if self._playing:
                return True
            self._playing = True
        Thread(target=self._play_with_fallbacks, name="posture-alert-audio", daemon=True).start()
        return True

    def _play_with_fallbacks(self) -> None:
        try:
            for command in self._commands():
                try:
                    completed = subprocess.run(
                        command,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=4.0,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    continue
                if completed.returncode == 0:
                    return
        finally:
            with self._lock:
                self._playing = False

    def _commands(self) -> list[list[str]]:
        path = str(self._sound_path.resolve())
        commands: list[list[str]] = []

        paplay = shutil.which("paplay")
        if paplay is not None:
            pulse_volume = round(65536 * min(self._volume_percent, 100) / 100.0)
            commands.append([paplay, f"--volume={pulse_volume}", path])

        pw_play = shutil.which("pw-play")
        if pw_play is not None:
            commands.append([pw_play, path])

        aplay = shutil.which("aplay")
        if aplay is not None:
            commands.append([aplay, "-q", path])

        ffplay = shutil.which("ffplay")
        if ffplay is not None:
            gain = max(0.0, self._volume_percent / 100.0)
            commands.append(
                [
                    ffplay,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-af",
                    f"volume={gain:.2f}",
                    path,
                ]
            )

        return commands
