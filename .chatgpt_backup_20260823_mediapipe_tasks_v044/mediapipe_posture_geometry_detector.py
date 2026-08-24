from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.model.posture_geometry import PostureGeometry
from nd_posture_guard.training.posture_training_profile import PostureTrainingProfile


class MediaPipePostureGeometryDetector:
    """Detect all seven posture landmarks independently on every frame.

    This deliberately does not propagate image points with optical flow. MediaPipe
    is run in static-image mode so every frame must produce a fresh face + pose
    detection. A missing detection returns None instead of reusing stale geometry.
    """

    FACE_MIN_CONFIDENCE = 0.50
    POSE_MIN_CONFIDENCE = 0.50
    SHOULDER_VISIBILITY_MIN = 0.45
    SHOULDER_INNER_FACTOR = 0.35

    def __init__(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required for drift-free landmark detection. "
                "Install it with: python -m pip install 'mediapipe>=1.0,<2'"
            ) from exc

        try:
            face_mesh_module = mp.solutions.face_mesh
            pose_module = mp.solutions.pose
        except AttributeError:
            try:
                from mediapipe.python.solutions import face_mesh as face_mesh_module
                from mediapipe.python.solutions import pose as pose_module
            except ImportError as exc:
                raise RuntimeError(
                    "This MediaPipe installation does not expose FaceMesh/Pose solutions."
                ) from exc

        self._pose_module = pose_module
        self._face_mesh = face_mesh_module.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.FACE_MIN_CONFIDENCE,
            min_tracking_confidence=0.99,
        )
        self._pose = pose_module.Pose(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=False,
            enable_segmentation=False,
            min_detection_confidence=self.POSE_MIN_CONFIDENCE,
            min_tracking_confidence=0.99,
        )
        self._profile: PostureTrainingProfile | None = None

    def set_profile(self, profile: PostureTrainingProfile | None) -> None:
        self._profile = profile

    def reset(self) -> None:
        return

    def track(self, frame: NDArray[np.uint8]) -> PostureGeometry | None:
        return self.detect(frame)

    def detect(self, frame: NDArray[np.uint8]) -> PostureGeometry | None:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
            return None

        height, width = image.shape[:2]
        rgb = np.ascontiguousarray(image[:, :, :3][:, :, ::-1])

        face_result = self._face_mesh.process(rgb)
        pose_result = self._pose.process(rgb)
        if not face_result.multi_face_landmarks or pose_result.pose_landmarks is None:
            return None

        face_landmarks = face_result.multi_face_landmarks[0].landmark
        pose_landmarks = pose_result.pose_landmarks.landmark
        if len(face_landmarks) <= 152 or len(pose_landmarks) <= 12:
            return None

        eye_a, eye_b = self._eye_centers(face_landmarks, width, height)
        chin_landmark = face_landmarks[152]
        chin = self._pixel_point(chin_landmark.x, chin_landmark.y, width, height)

        left_shoulder = pose_landmarks[
            self._pose_module.PoseLandmark.LEFT_SHOULDER.value
        ]
        right_shoulder = pose_landmarks[
            self._pose_module.PoseLandmark.RIGHT_SHOULDER.value
        ]
        visibility = min(
            float(getattr(left_shoulder, "visibility", 1.0)),
            float(getattr(right_shoulder, "visibility", 1.0)),
        )
        if visibility < self.SHOULDER_VISIBILITY_MIN:
            return None

        shoulder_a = np.asarray(
            self._pixel_point(left_shoulder.x, left_shoulder.y, width, height),
            dtype=np.float32,
        )
        shoulder_b = np.asarray(
            self._pixel_point(right_shoulder.x, right_shoulder.y, width, height),
            dtype=np.float32,
        )
        shoulder_center = 0.5 * (shoulder_a + shoulder_b)

        shoulder_segments = []
        for shoulder in (shoulder_a, shoulder_b):
            inner = shoulder_center + self.SHOULDER_INNER_FACTOR * (
                shoulder - shoulder_center
            )
            shoulder_segments.append(
                (
                    (float(inner[0]), float(inner[1])),
                    (float(shoulder[0]), float(shoulder[1])),
                )
            )

        eye_points = sorted((eye_a, eye_b), key=lambda point: point[0])
        shoulder_segments.sort(
            key=lambda segment: 0.5 * (segment[0][0] + segment[1][0])
        )
        raw_points = (
            eye_points[0],
            eye_points[1],
            chin,
            shoulder_segments[0][0],
            shoulder_segments[0][1],
            shoulder_segments[1][0],
            shoulder_segments[1][1],
        )
        confidence = float(np.clip(0.80 + 0.20 * visibility, 0.0, 1.0))
        geometry = PostureGeometry.from_points(raw_points, confidence)
        if not geometry.is_plausible(width, height):
            return None
        return geometry

    def close(self) -> None:
        for solution in (self._face_mesh, self._pose):
            close = getattr(solution, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _pixel_point(
        normalized_x: float,
        normalized_y: float,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        x = float(np.clip(normalized_x, 0.0, 1.0)) * max(width - 1, 1)
        y = float(np.clip(normalized_y, 0.0, 1.0)) * max(height - 1, 1)
        return x, y

    def _eye_centers(
        self,
        landmarks,
        width: int,
        height: int,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if len(landmarks) >= 478:
            groups = (
                (468, 469, 470, 471, 472),
                (473, 474, 475, 476, 477),
            )
        else:
            groups = (
                (33, 133, 159, 145),
                (362, 263, 386, 374),
            )

        centers: list[tuple[float, float]] = []
        for group in groups:
            xs = [float(landmarks[index].x) for index in group]
            ys = [float(landmarks[index].y) for index in group]
            centers.append(
                self._pixel_point(
                    float(np.mean(xs)),
                    float(np.mean(ys)),
                    width,
                    height,
                )
            )
        return centers[0], centers[1]
