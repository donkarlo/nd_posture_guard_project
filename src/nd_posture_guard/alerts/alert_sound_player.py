from __future__ import annotations

from pathlib import Path
import math
import shutil
import struct
import subprocess
from threading import Lock, Thread
import wave


class AlertSoundPlayer:
    """Play a gentle two-beep warning asynchronously with audio fallbacks."""

    SAMPLE_RATE = 44100
    BEEP_FREQUENCY_HZ = 740.0
    BEEP_SECONDS = 0.085
    GAP_SECONDS = 0.070
    FADE_SECONDS = 0.012
    AMPLITUDE = 0.14

    def __init__(self, sound_path: Path, volume_percent: int) -> None:
        self._fallback_sound_path = sound_path
        self._volume_percent = max(0, min(int(volume_percent), 150))
        self._lock = Lock()
        self._playing = False
        self._sound_path = self._prepare_double_beep()

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
        Thread(
            target=self._play_with_fallbacks,
            name="posture-alert-audio",
            daemon=True,
        ).start()
        return True

    def _prepare_double_beep(self) -> Path:
        cache_path = Path.home() / ".cache" / "nd_posture_guard" / "double_soft_beep.wav"
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.is_file() and cache_path.stat().st_size > 1000:
                return cache_path
            self._write_double_beep(cache_path)
            return cache_path
        except OSError:
            return self._fallback_sound_path

    def _write_double_beep(self, path: Path) -> None:
        beep_frames = max(1, round(self.SAMPLE_RATE * self.BEEP_SECONDS))
        gap_frames = max(1, round(self.SAMPLE_RATE * self.GAP_SECONDS))
        fade_frames = max(1, round(self.SAMPLE_RATE * self.FADE_SECONDS))
        samples: list[int] = []

        for beep_index in range(2):
            for frame_index in range(beep_frames):
                envelope = 1.0
                if frame_index < fade_frames:
                    envelope = frame_index / fade_frames
                elif frame_index >= beep_frames - fade_frames:
                    envelope = (beep_frames - 1 - frame_index) / fade_frames
                envelope = max(0.0, min(1.0, envelope))
                phase = (
                    2.0
                    * math.pi
                    * self.BEEP_FREQUENCY_HZ
                    * frame_index
                    / self.SAMPLE_RATE
                )
                value = self.AMPLITUDE * envelope * math.sin(phase)
                samples.append(int(max(-1.0, min(1.0, value)) * 32767))

            if beep_index == 0:
                samples.extend([0] * gap_frames)

        temporary = path.with_suffix(".part")
        with wave.open(str(temporary), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(
                b"".join(struct.pack("<h", sample) for sample in samples)
            )
        temporary.replace(path)

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
            pulse_volume = round(
                65536 * min(self._volume_percent, 100) / 100.0
            )
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
