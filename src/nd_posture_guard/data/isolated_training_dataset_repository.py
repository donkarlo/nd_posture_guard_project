from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from threading import RLock, Thread

import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QImage

from nd_posture_guard.data.non_blocking_training_dataset_repository import (
    NonBlockingTrainingDatasetRepository,
)
from nd_posture_guard.model.posture_geometry import PostureGeometry
from nd_posture_guard.training.posture_training_profile import PostureTrainingProfile
from nd_posture_guard.vision.posture_geometry_feature_extractor import (
    PostureGeometryFeatureExtractor,
)


class IsolatedTrainingDatasetRepository(NonBlockingTrainingDatasetRepository):
    """Persist training data without using OpenCV image codecs.

    Camera capture is OpenCV/V4L2 based. Keeping cv2.imwrite/cv2.imread out of
    persistence/profile building avoids native codec/capture lock contention.
    Review-video encoding remains in the inherited external-ffmpeg process.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._repository_lock = RLock()
        super().__init__(*args, **kwargs)

    def save_sample(
        self,
        label: int,
        points: tuple[tuple[float, float], ...],
        anchor_frame: NDArray[np.uint8],
        frames: tuple[NDArray[np.uint8], ...],
        feature: NDArray[np.float32],
    ) -> Path:
        with self._repository_lock:
            return self._save_sample_locked(label, points, anchor_frame, frames, feature)

    def _save_sample_locked(
        self,
        label: int,
        points: tuple[tuple[float, float], ...],
        anchor_frame: NDArray[np.uint8],
        frames: tuple[NDArray[np.uint8], ...],
        feature: NDArray[np.float32],
    ) -> Path:
        if label not in (0, 1):
            raise ValueError("Training label must be 0 (GOOD) or 1 (BAD).")
        if len(points) != 7:
            raise ValueError("Exactly seven face/shoulder points are required for a sample.")
        if not frames:
            raise ValueError("A training sample must contain a short raw camera clip.")

        vector = np.asarray(feature, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("A training sample must contain one geometry feature vector.")

        anchor = np.asarray(anchor_frame, dtype=np.uint8)
        if anchor.ndim != 3 or anchor.shape[2] < 3:
            raise ValueError("Training anchor frame must be a BGR color image.")
        height, width = anchor.shape[:2]
        geometry = PostureGeometry.from_points(points)
        if not geometry.is_plausible(width, height):
            raise ValueError("The saved seven-point posture geometry is not plausible.")

        label_name = "good" if label == 0 else "bad"
        sample_id = self._new_sample_id()
        sample_dir = self._root / "samples" / label_name / sample_id
        sample_dir.mkdir(parents=True, exist_ok=False)

        try:
            np.save(sample_dir / "features.npy", vector.reshape(1, -1))
            self._save_anchor_without_opencv(sample_dir / "anchor_frame.jpg", anchor)
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
                    "anchor_encoding": "qt_qimage_jpeg",
                },
            }
            (sample_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._update_manifest()
        except Exception:
            shutil.rmtree(sample_dir, ignore_errors=True)
            raise

        Thread(
            target=self._encode_review_video_with_ffmpeg,
            args=(sample_dir, frames, width, height, sample_id),
            name=f"posture-ffmpeg-{sample_id}",
            daemon=True,
        ).start()
        return sample_dir

    @staticmethod
    def _save_anchor_without_opencv(path: Path, frame: NDArray[np.uint8]) -> None:
        rgb = np.ascontiguousarray(frame[:, :, :3][:, :, ::-1])
        height, width = rgb.shape[:2]
        image = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        if image.isNull() or not image.save(str(path), "JPG", 90):
            raise OSError("Could not save the training anchor frame with Qt image encoder.")

    def build_profile(
        self,
        feature_extractor: PostureGeometryFeatureExtractor,
    ) -> PostureTrainingProfile | None:
        """Build the model from geometry only; no image decoding or template matching."""
        with self._repository_lock:
            samples = self._load_geometry_samples()
            if not samples:
                return None

            features: list[NDArray[np.float32]] = []
            labels: list[int] = []
            normalized_point_sets: list[tuple[tuple[float, float], ...]] = []
            good_point_sets: list[tuple[tuple[float, float], ...]] = []

            for entry in reversed(samples):
                width = int(entry["frame_width"])
                height = int(entry["frame_height"])
                normalized = entry["points_normalized"]
                assert isinstance(normalized, tuple)
                points = tuple((float(x) * width, float(y) * height) for x, y in normalized)
                geometry = PostureGeometry.from_points(points)
                if not geometry.is_plausible(width, height):
                    continue
                try:
                    vector = feature_extractor.extract(geometry, width, height)
                except ValueError:
                    continue
                features.append(vector)
                label = int(entry["label"])
                labels.append(label)
                normalized_point_sets.append(normalized)
                if label == 0:
                    good_point_sets.append(normalized)

            if not features:
                return None

            reference_source = good_point_sets if good_point_sets else normalized_point_sets
            reference_array = np.asarray(reference_source, dtype=np.float64)
            median_points = np.median(reference_array, axis=0)
            reference = tuple((float(point[0]), float(point[1])) for point in median_points)

            placeholders = tuple(
                (np.zeros((1, 1), dtype=np.uint8),) for _ in range(7)
            )
            return PostureTrainingProfile(
                reference_points_normalized=reference,
                point_templates=placeholders,
                features=np.vstack(features).astype(np.float32),
                labels=np.asarray(labels, dtype=np.int8),
            )
