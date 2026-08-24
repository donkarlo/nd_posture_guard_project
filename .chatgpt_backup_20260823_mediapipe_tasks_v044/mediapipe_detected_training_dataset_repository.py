from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QImage

from nd_posture_guard.data.serialized_video_training_dataset_repository import (
    SerializedVideoTrainingDatasetRepository,
)
from nd_posture_guard.training.posture_training_profile import PostureTrainingProfile
from nd_posture_guard.vision.mediapipe_posture_geometry_detector import (
    MediaPipePostureGeometryDetector,
)
from nd_posture_guard.vision.posture_geometry_feature_extractor import (
    PostureGeometryFeatureExtractor,
)


class MediaPipeDetectedTrainingDatasetRepository(
    SerializedVideoTrainingDatasetRepository
):
    """Rebuild profile features from each saved anchor using fresh detections."""

    def build_profile(
        self,
        feature_extractor: PostureGeometryFeatureExtractor,
    ) -> PostureTrainingProfile | None:
        with self._repository_lock:
            samples = self._load_geometry_samples()
            if not samples:
                return None

            detector = MediaPipePostureGeometryDetector()
            try:
                features: list[NDArray[np.float32]] = []
                labels: list[int] = []
                normalized_sets: list[tuple[tuple[float, float], ...]] = []
                good_sets: list[tuple[tuple[float, float], ...]] = []

                for entry in reversed(samples):
                    anchor = self._load_anchor_without_opencv(Path(entry["anchor_path"]))
                    if anchor is None:
                        continue
                    geometry = detector.detect(anchor)
                    if geometry is None:
                        continue

                    height, width = anchor.shape[:2]
                    try:
                        vector = feature_extractor.extract(geometry, width, height)
                    except ValueError:
                        continue

                    normalized = tuple(
                        (
                            float(x) / max(width, 1),
                            float(y) / max(height, 1),
                        )
                        for x, y in geometry.points
                    )
                    label = int(entry["label"])
                    features.append(vector)
                    labels.append(label)
                    normalized_sets.append(normalized)
                    if label == 0:
                        good_sets.append(normalized)
            finally:
                detector.close()

            if not features or set(labels) != {0, 1}:
                return None

            reference_source = good_sets if good_sets else normalized_sets
            reference_array = np.asarray(reference_source, dtype=np.float64)
            median_points = np.median(reference_array, axis=0)
            reference = tuple(
                (float(point[0]), float(point[1])) for point in median_points
            )
            placeholders = tuple(
                (np.zeros((1, 1), dtype=np.uint8),) for _ in range(7)
            )
            return PostureTrainingProfile(
                reference_points_normalized=reference,
                point_templates=placeholders,
                features=np.vstack(features).astype(np.float32),
                labels=np.asarray(labels, dtype=np.int8),
            )

    @staticmethod
    def _load_anchor_without_opencv(path: Path) -> NDArray[np.uint8] | None:
        image = QImage(str(path))
        if image.isNull():
            return None
        image = image.convertToFormat(QImage.Format.Format_RGB888)
        width = image.width()
        height = image.height()
        if width <= 0 or height <= 0:
            return None

        bytes_per_line = image.bytesPerLine()
        raw = np.frombuffer(
            image.bits(),
            dtype=np.uint8,
            count=image.sizeInBytes(),
        ).reshape(height, bytes_per_line)
        rgb = raw[:, : width * 3].reshape(height, width, 3).copy()
        return np.ascontiguousarray(rgb[:, :, ::-1])
