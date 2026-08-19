from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer


class AlertSoundPlayer:
    def __init__(self, sound_path: Path, volume_percent: int) -> None:
        self._sound_path = sound_path
        self._volume_percent = max(0, min(int(volume_percent), 150))
        self._process: subprocess.Popen[bytes] | None = None

        self._audio_output = QAudioOutput()
        self._audio_output.setVolume(min(self._volume_percent / 100.0, 1.0))
        self._player = QMediaPlayer()
        self._player.setAudioOutput(self._audio_output)
        if self._sound_path.is_file():
            self._player.setSource(QUrl.fromLocalFile(str(self._sound_path.resolve())))

    @property
    def sound_path(self) -> Path:
        return self._sound_path

    def play(self) -> bool:
        if not self._sound_path.is_file():
            return False

        self._stop_previous_external_player()
        suffix = self._sound_path.suffix.lower()

        # WAV + paplay is the most reliable path on Ubuntu/PulseAudio/PipeWire.
        if suffix == ".wav" and self._play_with_paplay():
            return True

        # ALSA fallback. It does not provide per-command gain, but is widely available.
        if suffix == ".wav" and self._play_with_aplay():
            return True

        # ffplay handles both WAV and MP3 and avoids Qt multimedia codec issues.
        if self._play_with_ffplay():
            return True

        # Final cross-platform fallback.
        self._player.setPosition(0)
        self._player.play()
        return True

    def _play_with_paplay(self) -> bool:
        paplay = shutil.which("paplay")
        if paplay is None:
            return False

        # paplay's normal volume is 65536. Do not pass >100% values to the CLI;
        # the Google WAV is pre-amplified during conversion instead.
        pulse_volume = round(65536 * min(self._volume_percent, 100) / 100.0)
        try:
            self._process = subprocess.Popen(
                [paplay, f"--volume={pulse_volume}", str(self._sound_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except OSError:
            return False

    def _play_with_aplay(self) -> bool:
        aplay = shutil.which("aplay")
        if aplay is None:
            return False
        try:
            self._process = subprocess.Popen(
                [aplay, "-q", str(self._sound_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except OSError:
            return False

    def _play_with_ffplay(self) -> bool:
        ffplay = shutil.which("ffplay")
        if ffplay is None:
            return False
        gain = max(0.0, self._volume_percent / 100.0)
        try:
            self._process = subprocess.Popen(
                [
                    ffplay,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-af",
                    f"volume={gain:.2f}",
                    str(self._sound_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except OSError:
            return False

    def _stop_previous_external_player(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        try:
            self._process.terminate()
        except OSError:
            pass
