from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.model.posture_classification import PostureClassification
from nd_posture_guard.training.shoulder_training_profile import ShoulderTrainingProfile


class ExamplePostureClassifier:
    """Balanced personal GOOD/BAD classifier with a supervised posture axis.

    Nearest-neighbour distance alone is easily dominated by irrelevant shirt
    texture in a high-dimensional image descriptor. This classifier additionally
    learns a diagonal Fisher-style direction from GOOD toward BAD. Feature
    dimensions that consistently differ between the user's GOOD and BAD samples
    therefore receive more weight than dimensions that merely fluctuate.
    """

    def __init__(self, bad_score_threshold: float, required_bad_frames: int, k_neighbors: int) -> None:
        self._bad_score_threshold = float(bad_score_threshold)
        self._required_bad_frames = max(1, int(required_bad_frames))
        self._k_neighbors = max(1, int(k_neighbors))
        self._profile: ShoulderTrainingProfile | None = None
        self._bad_frames = 0
        self._score_history: deque[float] = deque(maxlen=3)
        self._direction: NDArray[np.float32] | None = None
        self._projection_boundary = 0.0
        self._projection_scale = 1.0

    @property
    def ready(self) -> bool:
        return self._profile is not None

    @property
    def profile(self) -> ShoulderTrainingProfile | None:
        return self._profile

    def set_profile(self, profile: ShoulderTrainingProfile) -> None:
        if profile.features.ndim != 2 or profile.features.shape[0] != len(profile.labels):
            raise ValueError("Invalid shoulder training profile.")
        if len(np.unique(profile.labels)) < 2:
            raise ValueError("Training data must contain both good and bad posture examples.")
        self._profile = profile
        self._prepare_posture_axis(profile.features, profile.labels)
        self._bad_frames = 0
        self._score_history.clear()

    def clear(self) -> None:
        self._profile = None
        self._direction = None
        self._bad_frames = 0
        self._score_history.clear()

    def set_bad_score_threshold(self, value: float) -> None:
        self._bad_score_threshold = float(value)
        self._bad_frames = 0
        self._score_history.clear()

    def classify(self, feature: NDArray[np.float32]) -> PostureClassification:
        if self._profile is None:
            return PostureClassification("not_trained", 0.0, 0.0, 0.0, False)

        vector = np.asarray(feature, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self._profile.features.shape[1]:
            raise ValueError("Posture feature length does not match the loaded training profile.")

        distances = np.linalg.norm(self._profile.features - vector, axis=1)
        nearest_distance = float(np.min(distances))
        good_distances = distances[self._profile.labels == 0]
        bad_distances = distances[self._profile.labels == 1]
        if good_distances.size == 0 or bad_distances.size == 0:
            return PostureClassification("not_trained", 0.0, 0.0, nearest_distance, False)

        good_distance = self._class_distance(good_distances)
        bad_distance = self._class_distance(bad_distances)
        distance_score = self._distance_bad_score(good_distance, bad_distance)

        projection_score = self._projection_bad_score(vector)
        raw_score = (
            distance_score
            if projection_score is None
            else (0.25 * distance_score) + (0.75 * projection_score)
        )

        self._score_history.append(float(raw_score))
        bad_score = float(np.median(np.asarray(self._score_history, dtype=np.float32)))
        confidence = min(1.0, abs(bad_score - 0.5) * 2.0)

        # 50% is the natural GOOD/BAD decision boundary. The user-adjustable
        # threshold controls the beep, not whether the classifier is allowed to
        # call a frame BAD. This makes diagnostics interpretable.
        posture_bad = bad_score > 0.5
        warning_bad = bad_score > self._bad_score_threshold
        self._bad_frames = self._bad_frames + 1 if warning_bad else 0

        return PostureClassification(
            state="bad" if posture_bad else "good",
            bad_score=bad_score,
            confidence=confidence,
            nearest_distance=nearest_distance,
            should_alert=warning_bad and self._bad_frames >= self._required_bad_frames,
        )

    def _prepare_posture_axis(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int8],
    ) -> None:
        good = features[labels == 0].astype(np.float64)
        bad = features[labels == 1].astype(np.float64)
        good_mean = np.mean(good, axis=0)
        bad_mean = np.mean(bad, axis=0)
        difference = bad_mean - good_mean

        good_var = np.var(good, axis=0) if len(good) > 1 else np.zeros_like(good_mean)
        bad_var = np.var(bad, axis=0) if len(bad) > 1 else np.zeros_like(bad_mean)
        pooled_var = 0.5 * (good_var + bad_var)
        positive = pooled_var[pooled_var > 1e-10]
        variance_floor = float(np.median(positive)) * 0.50 if positive.size else 1e-4
        variance_floor = max(variance_floor, 1e-5)

        direction = difference / np.sqrt(pooled_var + variance_floor)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            self._direction = None
            self._projection_boundary = 0.0
            self._projection_scale = 1.0
            return
        direction = direction / norm

        good_projection = good @ direction
        bad_projection = bad @ direction
        good_center = float(np.median(good_projection))
        bad_center = float(np.median(bad_projection))
        if bad_center < good_center:
            direction = -direction
            good_projection = -good_projection
            bad_projection = -bad_projection
            good_center = -good_center
            bad_center = -bad_center

        separation = bad_center - good_center
        if separation <= 1e-6:
            self._direction = None
            return

        within_projection = np.concatenate(
            [good_projection - good_center, bad_projection - bad_center]
        )
        robust_noise = float(np.percentile(np.abs(within_projection), 75.0))
        # At the class medians this produces scores roughly 0.1 and 0.9 when the
        # classes are well separated, but widens smoothly for noisier examples.
        scale = max(separation * 0.22, robust_noise * 1.5, 1e-4)

        self._direction = direction.astype(np.float32)
        self._projection_boundary = 0.5 * (good_center + bad_center)
        self._projection_scale = float(scale)

    def _projection_bad_score(self, vector: NDArray[np.float32]) -> float | None:
        if self._direction is None:
            return None
        projection = float(np.dot(vector, self._direction))
        z = (projection - self._projection_boundary) / self._projection_scale
        z = max(-12.0, min(12.0, z))
        return float(1.0 / (1.0 + np.exp(-z)))

    @staticmethod
    def _distance_bad_score(good_distance: float, bad_distance: float) -> float:
        good_sq = good_distance * good_distance
        bad_sq = bad_distance * bad_distance
        denominator = good_sq + bad_sq
        return 0.5 if denominator <= 1e-12 else float(good_sq / denominator)

    def _class_distance(self, distances: NDArray[np.float32]) -> float:
        count = min(max(1, self._k_neighbors), int(distances.size), 3)
        if count == distances.size:
            selected = distances
        else:
            selected = np.partition(distances, count - 1)[:count]
        return float(np.mean(selected))
