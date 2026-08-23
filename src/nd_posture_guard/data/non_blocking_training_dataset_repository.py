from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from threading import Thread
import tempfile

import cv2
import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.data.training_dataset_repository import TrainingDatasetRepository
from nd_posture_guard.model.posture_geometry import PostureGeometry


class NonBlockingTrainingDatasetRepository(TrainingDatasetRepository):
    """Persist model-critical data immediately and encode review video outside OpenCV video I/O."""

    def save_sample(
        self,
        label: int,
        points: tuple[tuple[float, float], ...],
        anchor_frame: NDArray[np.uint8],
        frames: tuple[NDArray[np.uint8], ...],
        feature: NDArray[np.float32],
    ) -> Path:
        """Save training state synchronously, then encode the review clip with external ffmpeg."""
        if label not in (0, 1):
            raise ValueError("Training label must be 0 (GOOD) or 1 (BAD).")
        if len(points) != 7:
            raise ValueError("Exactly seven face/shoulder points are required for a sample.")
        if not frames:
            raise ValueError("A training sample must contain a short raw camera clip.")

        vector = np.asarray(feature, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("A training sample must contain one geometry feature vector.")

        height, width = anchor_frame.shape[:2]
        geometry = PostureGeometry.from_points(points)
        if not geometry.is_plausible(width, height):
            raise ValueError("The saved seven-point posture geometry is not plausible.")

        label_name = "good" if label == 0 else "bad"
        sample_id = self._new_sample_id()
        sample_dir = self._root / "samples" / label_name / sample_id
        sample_dir.mkdir(parents=True, exist_ok=False)

        try:
            np.save(sample_dir / "features.npy", vector.reshape(1, -1))
            if not cv2.imwrite(str(sample_dir / "anchor_frame.jpg"), anchor_frame):
                raise OSError("Could not save the training anchor frame.")

            metadata = {
                "sample_id": sample_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "data_schema_version": self.DATA_SCHEMA_VERSION,
                "geometry_schema": self.GEOMETRY_SCHEMA,
                "feature_length": int(vector.size),
                "label": int(label),
                "label_name": label_name,
                "frame_width": int(width),
                "frame_height": int(height),
                "frame_count": int(len(frames)),
                "points_normalized": [
                    [float(x) / max(width, 1), float(y) / max(height, 1)] for x, y in points
                ],
                "point_order": [
                    "left_image_eye_center",
                    "right_image_eye_center",
                    "chin_bottom",
                    "left_image_shoulder_inner",
                    "left_image_shoulder_outer",
                    "right_image_shoulder_inner",
                    "right_image_shoulder_outer",
                ],
                "media": {
                    "anchor_frame": "anchor_frame.jpg",
                    "frames_archive": None,
                    "video": "clip.avi",
                    "video_encoding": "background_external_ffmpeg",
                },
            }
            (sample_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            shutil.rmtree(sample_dir, ignore_errors=True)
            raise

        self._update_manifest()

        # The camera uses OpenCV VideoCapture. Do not create any OpenCV VideoWriter
        # while capture is live: some Linux/OpenCV backends can serialize or deadlock
        # capture and writer operations. ffmpeg runs in a separate OS process instead.
        Thread(
            target=self._encode_review_video_with_ffmpeg,
            args=(sample_dir, frames, width, height, sample_id),
            name=f"posture-ffmpeg-{sample_id}",
            daemon=True,
        ).start()
        return sample_dir

    def _encode_review_video_with_ffmpeg(
        self,
        sample_dir: Path,
        frames: tuple[NDArray[np.uint8], ...],
        width: int,
        height: int,
        sample_id: str,
    ) -> None:
        """Encode raw BGR frames via an external ffmpeg process, never cv2.VideoWriter."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return

        temp_path: Path | None = None
        staged_path = sample_dir / ".clip.avi.part"
        final_path = sample_dir / "clip.avi"
        try:
            expected_shape = (height, width)
            if any(frame.shape[:2] != expected_shape for frame in frames):
                return
            if any(frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 for frame in frames):
                return

            with tempfile.NamedTemporaryFile(
                prefix=f"nd_posture_guard_{sample_id}_",
                suffix=".avi",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)

            raw_video = b"".join(frame.tobytes(order="C") for frame in frames)
            duration = len(frames) / max(self._recording_fps, 1.0)
            timeout_seconds = max(10.0, duration * 4.0 + 5.0)
            result = subprocess.run(
                [
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
                    "-c:v",
                    "mjpeg",
                    "-q:v",
                    "5",
                    "-f",
                    "avi",
                    str(temp_path),
                ],
                input=raw_video,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=timeout_seconds,
            )
            if result.returncode != 0:
                return
            if not temp_path.is_file() or temp_path.stat().st_size <= 0:
                return
            if not sample_dir.is_dir():
                return

            shutil.copyfile(temp_path, staged_path)
            staged_path.replace(final_path)
        except (OSError, subprocess.SubprocessError, MemoryError):
            pass
        finally:
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
