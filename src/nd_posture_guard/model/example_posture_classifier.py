from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.model.posture_classification import PostureClassification
from nd_posture_guard.training.posture_training_profile import PostureTrainingProfile


class ExamplePostureClassifier:
    """Classify normalized seven-point geometry against the user's GOOD/BAD examples."""

    def __init__(self, bad_score_threshold: float, required_bad_frames: int, k_neighbors: int) -> None:
        self._bad_score_threshold = float(bad_score_threshold)
        self._required_bad_frames = max(1, int(required_bad_frames))
        self._k_neighbors = max(1, int(k_neighbors))
        self._profile: PostureTrainingProfile | None = None
        self._scaled_features: NDArray[np.float32] | None = None
        self._feature_center: NDArray[np.float32] | None = None
        self._feature_scale: NDArray[np.float32] | None = None
        self._direction: NDArray[np.float32] | None = None
        self._projection_boundary = 0.0
        self._projection_scale = 1.0
        self._bad_frames = 0
        self._score_history: deque[float] = deque(maxlen=5)

    @property
    def ready(self) -> bool:
        """Return whether both GOOD and BAD geometry examples are loaded."""
        return self._profile is not None and self._scaled_features is not None

    @property
    def profile(self) -> PostureTrainingProfile | None:
        """Return the currently loaded personal geometry profile."""
        return self._profile

    def set_profile(self, profile: PostureTrainingProfile) -> None:
        """Fit robust feature scaling and the supervised GOOD-to-BAD posture axis."""
        features = np.asarray(profile.features, dtype=np.float32)
        labels = np.asarray(profile.labels, dtype=np.int8)
        if features.ndim != 2 or features.shape[0] != labels.shape[0]:
            raise ValueError("Invalid posture geometry training profile.")
        if features.shape[0] < 2 or set(int(value) for value in np.unique(labels)) != {0, 1}:
            raise ValueError("Training data must contain both GOOD and BAD geometry examples.")

        self._profile = profile
        self._feature_center, self._feature_scale = self._robust_scaling(features)
        self._scaled_features = self._scale(features)
        self._prepare_posture_axis(self._scaled_features, labels)
        self._bad_frames = 0
        self._score_history.clear()

    def clear(self) -> None:
        """Clear the active personal model and temporal state."""
        self._profile = None
        self._scaled_features = None
        self._feature_center = None
        self._feature_scale = None
        self._direction = None
        self._bad_frames = 0
        self._score_history.clear()

    def set_bad_score_threshold(self, value: float) -> None:
        """Update the alert threshold without retraining the geometry classifier."""
        self._bad_score_threshold = float(value)
        self._bad_frames = 0
        self._score_history.clear()

    def classify(self, feature: NDArray[np.float32]) -> PostureClassification:
        """Classify one tracked geometry feature and apply temporal alert hysteresis."""
        if not self.ready or self._profile is None or self._scaled_features is None:
            return PostureClassification("not_trained", 0.0, 0.0, 0.0, False)

        vector = np.asarray(feature, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self._profile.features.shape[1]:
            raise ValueError("Posture geometry feature length does not match the loaded profile.")
        scaled = self._scale(vector.reshape(1, -1))[0]
        distances = np.linalg.norm(self._scaled_features - scaled, axis=1)
        labels = self._profile.labels
        good_distances = distances[labels == 0]
        bad_distances = distances[labels == 1]
        if good_distances.size == 0 or bad_distances.size == 0:
            return PostureClassification("not_trained", 0.0, 0.0, 0.0, False)

        good_distance = self._class_distance(good_distances)
        bad_distance = self._class_distance(bad_distances)
        nearest_distance = float(min(good_distance, bad_distance))
        distance_score = self._distance_bad_score(good_distance, bad_distance)
        projection_score = self._projection_bad_score(scaled)

        # Geometry has only a few interpretable dimensions, so the supervised
        # posture axis is useful without allowing it to completely override the
        # nearest examples when the user has only one sample per class.
        raw_score = (
            distance_score
            if projection_score is None
            else (0.40 * distance_score) + (0.60 * projection_score)
        )
        self._score_history.append(float(raw_score))
        bad_score = float(np.median(np.asarray(self._score_history, dtype=np.float32)))
        confidence = float(np.clip(abs(bad_score - 0.5) * 2.0, 0.0, 1.0))

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

    @staticmethod
    def _robust_scaling(
        features: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Estimate a robust per-dimension location and minimum-safe scale."""
        center = np.median(features, axis=0).astype(np.float32)
        deviations = np.abs(features - center)
        mad = np.median(deviations, axis=0).astype(np.float32)
        scale = 1.4826 * mad

        # With one GOOD and one BAD example MAD may be exactly half their
        # difference; with identical dimensions it can be zero. A small geometry
        # floor prevents numerical explosions while still making millimetric jitter
        # much less important than a visible head/shoulder change.
        scale = np.maximum(scale, 0.012).astype(np.float32)
        return center, scale

    def _scale(self, values: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply the fitted robust normalization to one or more feature vectors."""
        if self._feature_center is None or self._feature_scale is None:
            raise RuntimeError("Geometry feature scaling has not been fitted.")
        return ((values - self._feature_center) / self._feature_scale).astype(np.float32)

    def _prepare_posture_axis(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int8],
    ) -> None:
        """Learn a low-variance direction from the user's GOOD class toward BAD."""
        good = features[labels == 0].astype(np.float64)
        bad = features[labels == 1].astype(np.float64)
        good_center = np.median(good, axis=0)
        bad_center = np.median(bad, axis=0)
        difference = bad_center - good_center

        good_var = np.var(good, axis=0) if len(good) > 1 else np.zeros_like(good_center)
        bad_var = np.var(bad, axis=0) if len(bad) > 1 else np.zeros_like(bad_center)
        pooled_var = 0.5 * (good_var + bad_var)
        positive = pooled_var[pooled_var > 1e-8]
        variance_floor = float(np.median(positive)) * 0.40 if positive.size else 0.20
        variance_floor = max(variance_floor, 0.08)

        direction = difference / np.sqrt(pooled_var + variance_floor)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            self._direction = None
            return
        direction /= norm

        good_projection = good @ direction
        bad_projection = bad @ direction
        good_scalar = float(np.median(good_projection))
        bad_scalar = float(np.median(bad_projection))
        if bad_scalar < good_scalar:
            direction = -direction
            good_projection = -good_projection
            bad_projection = -bad_projection
            good_scalar = -good_scalar
            bad_scalar = -bad_scalar

        separation = bad_scalar - good_scalar
        if separation <= 1e-6:
            self._direction = None
            return

        residuals = np.concatenate(
            [good_projection - good_scalar, bad_projection - bad_scalar]
        )
        noise = float(np.percentile(np.abs(residuals), 75.0)) if residuals.size else 0.0
        self._direction = direction.astype(np.float32)
        self._projection_boundary = 0.5 * (good_scalar + bad_scalar)
        self._projection_scale = max(separation * 0.24, noise * 1.5, 0.12)

    def _projection_bad_score(self, vector: NDArray[np.float32]) -> float | None:
        """Convert position on the learned GOOD-to-BAD axis into a probability-like score."""
        if self._direction is None:
            return None
        projection = float(np.dot(vector, self._direction))
        z = (projection - self._projection_boundary) / self._projection_scale
        z = max(-12.0, min(12.0, z))
        return float(1.0 / (1.0 + np.exp(-z)))

    @staticmethod
    def _distance_bad_score(good_distance: float, bad_distance: float) -> float:
        """Map relative GOOD/BAD distances to a bounded BAD score."""
        good_sq = good_distance * good_distance
        bad_sq = bad_distance * bad_distance
        denominator = good_sq + bad_sq
        return 0.5 if denominator <= 1e-12 else float(good_sq / denominator)

    def _class_distance(self, distances: NDArray[np.float32]) -> float:
        """Average only the nearest few examples of one class."""
        count = min(max(1, self._k_neighbors), int(distances.size), 3)
        selected = distances if count == distances.size else np.partition(distances, count - 1)[:count]
        return float(np.mean(selected))
