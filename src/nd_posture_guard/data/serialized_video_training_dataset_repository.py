from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
import os
import shutil
import subprocess
from threading import Event, Thread
import tempfile

import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.data.isolated_training_dataset_repository import (
    IsolatedTrainingDatasetRepository,
)


class SerializedVideoTrainingDatasetRepository(IsolatedTrainingDatasetRepository):
    """Encode review videos through one low-priority ffmpeg worker.

    The training/persistence path remains non-blocking, but video encoding is
    deliberately serialized so multiple training samples can never spawn multiple
    concurrent ffmpeg processes. Frames are streamed one at a time instead of
    being concatenated into one large in-memory byte buffer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._video_jobs: Queue[
            tuple[Path, tuple[NDArray[np.uint8], ...], int, int, str]
        ] = Queue()
        self._video_stop = Event()
        self._video_worker = Thread(
            target=self._video_encode_loop,
            name="posture-video-encoder",
            daemon=True,
        )
        self._video_worker.start()

    def _encode_review_video_with_ffmpeg(
        self,
        sample_dir: Path,
        frames: tuple[NDArray[np.uint8], ...],
        width: int,
        height: int,
        sample_id: str,
    ) -> None:
        """Queue one clip; return immediately so save_sample never waits for ffmpeg."""
        self._video_jobs.put((sample_dir, frames, int(width), int(height), sample_id))

    def _video_encode_loop(self) -> None:
        """Run exactly one ffmpeg process at a time for all queued samples."""
        while not self._video_stop.is_set() or not self._video_jobs.empty():
            try:
                job = self._video_jobs.get(timeout=0.25)
            except Empty:
                continue
            try:
                self._encode_one_clip(*job)
            finally:
                self._video_jobs.task_done()

    def _encode_one_clip(
        self,
        sample_dir: Path,
        frames: tuple[NDArray[np.uint8], ...],
        width: int,
        height: int,
        sample_id: str,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None or not frames or not sample_dir.is_dir():
            return

        expected_shape = (height, width)
        for frame in frames:
            if (
                frame.shape[:2] != expected_shape
                or frame.ndim != 3
                or frame.shape[2] != 3
                or frame.dtype != np.uint8
            ):
                return

        temp_path: Path | None = None
        staged_path = sample_dir / ".clip.avi.part"
        final_path = sample_dir / "clip.avi"
        process: subprocess.Popen[bytes] | None = None

        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"nd_posture_guard_{sample_id}_",
                suffix=".avi",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)

            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                f"{self._recording_fps:g}",
                "-i",
                "pipe:0",
                "-an",
                "-threads",
                "1",
                "-c:v",
                "mjpeg",
                "-q:v",
                "5",
                "-f",
                "avi",
                str(temp_path),
            ]

            nice = shutil.which("nice")
            if nice is not None and os.name == "posix":
                command = [nice, "-n", "10", *command]

            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if process.stdin is None:
                return

            try:
                for frame in frames:
                    process.stdin.write(memoryview(np.ascontiguousarray(frame)))
                process.stdin.close()
                process.stdin = None
            except (BrokenPipeError, OSError):
                return

            duration = len(frames) / max(self._recording_fps, 1.0)
            timeout_seconds = max(10.0, duration * 4.0 + 5.0)
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
                return

            if return_code != 0:
                return
            if temp_path is None or not temp_path.is_file() or temp_path.stat().st_size <= 0:
                return
            if not sample_dir.is_dir():
                return

            shutil.copyfile(temp_path, staged_path)
            staged_path.replace(final_path)
        except (OSError, subprocess.SubprocessError, MemoryError):
            return
        finally:
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                    process.wait(timeout=1.0)
                except (OSError, subprocess.SubprocessError):
                    pass
            try:
                if staged_path.exists():
                    staged_path.unlink()
            except OSError:
                pass
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
