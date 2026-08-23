from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from nd_posture_guard.data.non_blocking_training_dataset_repository import (
    NonBlockingTrainingDatasetRepository,
)
from nd_posture_guard.training.posture_training_profile import PostureTrainingProfile
from nd_posture_guard.vision.posture_geometry_feature_extractor import (
    PostureGeometryFeatureExtractor,
)


class StableTrainingDatasetRepository(NonBlockingTrainingDatasetRepository):
    """Build landmark tracking from any usable sample; classification still needs both classes."""

    def build_profile(
        self,
        feature_extractor: PostureGeometryFeatureExtractor,
    ) -> PostureTrainingProfile | None:
        samples = self._load_geometry_samples()
        if not samples:
            return None

        features: list[np.ndarray] = []
        labels: list[int] = []
        normalized_point_sets: list[tuple[tuple[float, float], ...]] = []
        good_point_sets: list[tuple[tuple[float, float], ...]] = []
        templates_by_point: list[list[np.ndarray]] = [[] for _ in range(7)]

        for entry in reversed(samples):
            width = int(entry["frame_width"])
            height = int(entry["frame_height"])
            normalized = entry["points_normalized"]
            assert isinstance(normalized, tuple)
            points = tuple((float(x) * width, float(y) * height) for x, y in normalized)

            from nd_posture_guard.model.posture_geometry import PostureGeometry

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

            anchor_path = Path(entry["anchor_path"])
            anchor = cv2.imread(str(anchor_path), cv2.IMREAD_COLOR)
            if anchor is None:
                continue
            gray = self._to_gray(anchor)
            anchor_h, anchor_w = gray.shape[:2]
            for index, (nx, ny) in enumerate(normalized):
                if len(templates_by_point[index]) >= self.MAX_TEMPLATES_PER_POINT:
                    continue
                template = self._extract_template(gray, nx * anchor_w, ny * anchor_h)
                if template is not None:
                    templates_by_point[index].append(template)

        if not features or any(not templates for templates in templates_by_point):
            return None

        reference_source = good_point_sets if good_point_sets else normalized_point_sets
        reference_array = np.asarray(reference_source, dtype=np.float64)
        median_points = np.median(reference_array, axis=0)
        reference = tuple((float(point[0]), float(point[1])) for point in median_points)

        return PostureTrainingProfile(
            reference_points_normalized=reference,
            point_templates=tuple(tuple(values) for values in templates_by_point),
            features=np.vstack(features).astype(np.float32),
            labels=np.asarray(labels, dtype=np.int8),
        )
