from __future__ import annotations

from pathlib import Path
import shutil
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.model.posture_geometry import PostureGeometry
from nd_posture_guard.training.posture_training_profile import PostureTrainingProfile


class MediaPipePostureGeometryDetector:
    """Detect seven posture landmarks independently on every frame with MediaPipe Tasks."""

    FACE_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )
    POSE_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )

    FACE_MIN_CONFIDENCE = 0.50
    POSE_MIN_CONFIDENCE = 0.50
    SHOULDER_VISIBILITY_MIN = 0.45
    SHOULDER_INNER_FACTOR = 0.35
    MIN_MODEL_BYTES = 100_000

    def __init__(self, model_dir: Path) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required for drift-free landmark detection. "
                "Install it with: python -m pip install 'mediapipe>=0.10.35,<1'"
            ) from exc

        if not hasattr(mp, "tasks") or not hasattr(mp.tasks, "vision"):
            raise RuntimeError(
                "This MediaPipe installation does not expose the Tasks Vision API."
            )

        self._mp = mp
        self._profile: PostureTrainingProfile | None = None
        self._model_dir = Path(model_dir).expanduser()
        self._model_dir.mkdir(parents=True, exist_ok=True)

        face_model = self._ensure_model(
            self._model_dir / "face_landmarker.task",
            self.FACE_MODEL_URL,
        )
        pose_model = self._ensure_model(
            self._model_dir / "pose_landmarker_lite.task",
            self.POSE_MODEL_URL,
        )

        base_options = mp.tasks.BaseOptions
        vision = mp.tasks.vision

        face_options = vision.FaceLandmarkerOptions(
            base_options=base_options(model_asset_path=str(face_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self.FACE_MIN_CONFIDENCE,
            min_face_presence_confidence=self.FACE_MIN_CONFIDENCE,
            min_tracking_confidence=0.99,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=str(pose_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.POSE_MIN_CONFIDENCE,
            min_pose_presence_confidence=self.POSE_MIN_CONFIDENCE,
            min_tracking_confidence=0.99,
            output_segmentation_masks=False,
        )

        self._face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        self._pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

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
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )

        face_result = self._face_landmarker.detect(mp_image)
        pose_result = self._pose_landmarker.detect(mp_image)
        if not face_result.face_landmarks or not pose_result.pose_landmarks:
            return None

        face_landmarks = face_result.face_landmarks[0]
        pose_landmarks = pose_result.pose_landmarks[0]
        if len(face_landmarks) <= 152 or len(pose_landmarks) <= 12:
            return None

        eye_a, eye_b = self._eye_centers(face_landmarks, width, height)
        chin_landmark = face_landmarks[152]
        chin = self._pixel_point(
            float(chin_landmark.x),
            float(chin_landmark.y),
            width,
            height,
        )

        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        visibility = min(
            self._landmark_confidence(left_shoulder),
            self._landmark_confidence(right_shoulder),
        )
        if visibility < self.SHOULDER_VISIBILITY_MIN:
            return None

        shoulder_a = np.asarray(
            self._pixel_point(
                float(left_shoulder.x),
                float(left_shoulder.y),
                width,
                height,
            ),
            dtype=np.float32,
        )
        shoulder_b = np.asarray(
            self._pixel_point(
                float(right_shoulder.x),
                float(right_shoulder.y),
                width,
                height,
            ),
            dtype=np.float32,
        )
        shoulder_center = 0.5 * (shoulder_a + shoulder_b)

        shoulder_segments: list[
            tuple[tuple[float, float], tuple[float, float]]
        ] = []
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
        for task in (self._face_landmarker, self._pose_landmarker):
            close = getattr(task, "close", None)
            if callable(close):
                close()

    def _ensure_model(self, path: Path, url: str) -> Path:
        if path.is_file() and path.stat().st_size >= self.MIN_MODEL_BYTES:
            return path

        temporary = path.with_suffix(path.suffix + ".part")
        temporary.unlink(missing_ok=True)
        request = Request(url, headers={"User-Agent": "nd-posture-guard/1.0"})
        try:
            with urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size < self.MIN_MODEL_BYTES:
                raise RuntimeError(
                    f"Downloaded MediaPipe model is unexpectedly small: {url}"
                )
            temporary.replace(path)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download required MediaPipe model: {url}"
            ) from exc
        return path

    @staticmethod
    def _landmark_confidence(landmark) -> float:
        values: list[float] = []
        for attribute in ("visibility", "presence"):
            value = getattr(landmark, attribute, None)
            if value is not None:
                values.append(float(value))
        return min(values) if values else 1.0

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
