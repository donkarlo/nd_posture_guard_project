from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from threading import Thread
import tempfile

import cv2
import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.data.training_dataset_repository import TrainingDatasetRepository
from nd_posture_guard.model.posture_geometry import PostureGeometry


class NonBlockingTrainingDatasetRepository(TrainingDatasetRepository):
    """Persist model-critical data immediately and encode review video off the camera worker."""

    def save_sample(
        self,
        label: int,
        points: tuple[tuple[float, float], ...],
        anchor_frame: NDArray[np.uint8],
        frames: tuple[NDArray[np.uint8], ...],
        feature: NDArray[np.float32],
    ) -> Path:
        """Save compact training state synchronously, then encode the review clip in background."""
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
                    "video_encoding": "background",
                },
            }
            (sample_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            shutil.rmtree(sample_dir, ignore_errors=True)
            raise

        self._update_manifest()

        Thread(
            target=self._encode_review_video,
            args=(sample_dir, frames, width, height, sample_id),
            name=f"posture-video-{sample_id}",
            daemon=True,
        ).start()
        return sample_dir

    def _encode_review_video(
        self,
        sample_dir: Path,
        frames: tuple[NDArray[np.uint8], ...],
        width: int,
        height: int,
        sample_id: str,
    ) -> None:
        """Encode locally first, then atomically publish the completed AVI into Dropbox."""
        temp_path: Path | None = None
        staged_path = sample_dir / ".clip.avi.part"
        final_path = sample_dir / "clip.avi"
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"nd_posture_guard_{sample_id}_",
                suffix=".avi",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)

            writer = cv2.VideoWriter(
                str(temp_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self._recording_fps,
                (width, height),
            )
            if not writer.isOpened():
                return
            try:
                for frame in frames:
                    output = frame
                    if frame.shape[1] != width or frame.shape[0] != height:
                        output = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    writer.write(output)
            finally:
                writer.release()

            if not temp_path.is_file() or temp_path.stat().st_size <= 0:
                return
            if not sample_dir.is_dir():
                return

            shutil.copyfile(temp_path, staged_path)
            os.replace(staged_path, final_path)
        except (OSError, cv2.error):
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
