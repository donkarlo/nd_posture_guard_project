from __future__ import annotations

from pathlib import Path
import math
import shutil
import struct
import subprocess
from threading import Lock, Thread
import wave


class AlertSoundPlayer:
    """Play a soft modern two-note warning asynchronously with audio fallbacks."""

    SAMPLE_RATE = 44100
    NOTE_FREQUENCIES_HZ = (659.25, 783.99)
    NOTE_SECONDS = 0.130
    GAP_SECONDS = 0.060
    ATTACK_SECONDS = 0.010
    RELEASE_SECONDS = 0.055
    AMPLITUDE = 0.18
    SECOND_HARMONIC_MIX = 0.10
    THIRD_HARMONIC_MIX = 0.025

    def __init__(self, sound_path: Path, volume_percent: int) -> None:
        self._fallback_sound_path = sound_path
        self._volume_percent = max(0, min(int(volume_percent), 150))
        self._lock = Lock()
        self._playing = False
        self._sound_path = self._prepare_modern_double_chime()

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

    def _prepare_modern_double_chime(self) -> Path:
        cache_path = (
            Path.home()
            / ".cache"
            / "nd_posture_guard"
            / "modern_double_chime_v1.wav"
        )
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.is_file() and cache_path.stat().st_size > 1000:
                return cache_path
            self._write_modern_double_chime(cache_path)
            return cache_path
        except OSError:
            return self._fallback_sound_path

    def _write_modern_double_chime(self, path: Path) -> None:
        note_frames = max(1, round(self.SAMPLE_RATE * self.NOTE_SECONDS))
        gap_frames = max(1, round(self.SAMPLE_RATE * self.GAP_SECONDS))
        attack_frames = max(1, round(self.SAMPLE_RATE * self.ATTACK_SECONDS))
        release_frames = max(1, round(self.SAMPLE_RATE * self.RELEASE_SECONDS))
        harmonic_normalizer = (
            1.0 + self.SECOND_HARMONIC_MIX + self.THIRD_HARMONIC_MIX
        )
        samples: list[int] = []

        for note_index, frequency_hz in enumerate(self.NOTE_FREQUENCIES_HZ):
            for frame_index in range(note_frames):
                attack_position = min(1.0, frame_index / attack_frames)
                release_position = min(
                    1.0,
                    max(0, note_frames - 1 - frame_index) / release_frames,
                )
                attack_envelope = math.sin(
                    0.5 * math.pi * attack_position
                ) ** 2
                release_envelope = math.sin(
                    0.5 * math.pi * release_position
                ) ** 2
                envelope = attack_envelope * release_envelope

                time_seconds = frame_index / self.SAMPLE_RATE
                phase = 2.0 * math.pi * frequency_hz * time_seconds
                tone = (
                    math.sin(phase)
                    + self.SECOND_HARMONIC_MIX * math.sin(2.0 * phase)
                    + self.THIRD_HARMONIC_MIX * math.sin(3.0 * phase)
                ) / harmonic_normalizer
                value = self.AMPLITUDE * envelope * tone
                samples.append(int(max(-1.0, min(1.0, value)) * 32767))

            if note_index == 0:
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
