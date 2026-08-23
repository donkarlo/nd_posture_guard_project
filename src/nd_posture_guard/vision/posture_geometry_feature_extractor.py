from __future__ import annotations


import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.model.posture_geometry import PostureGeometry


class PostureGeometryFeatureExtractor:
    """Convert seven posture landmarks into translation/scale-normalized geometry features."""

    SCHEMA = "face_triangle_shoulder_lines_geometry_v1"
    FEATURE_LENGTH = 28

    @property
    def feature_length(self) -> int:
        return self.FEATURE_LENGTH

    def extract(
        self,
        geometry: PostureGeometry,
        frame_width: int,
        frame_height: int,
    ) -> NDArray[np.float32]:
        if not geometry.is_plausible(frame_width, frame_height):
            raise ValueError("The detected face/shoulder geometry is not plausible.")

        left_eye = np.asarray(geometry.left_eye, dtype=np.float64)
        right_eye = np.asarray(geometry.right_eye, dtype=np.float64)
        chin = np.asarray(geometry.chin, dtype=np.float64)
        left_inner = np.asarray(geometry.left_shoulder_inner, dtype=np.float64)
        left_outer = np.asarray(geometry.left_shoulder_outer, dtype=np.float64)
        right_inner = np.asarray(geometry.right_shoulder_inner, dtype=np.float64)
        right_outer = np.asarray(geometry.right_shoulder_outer, dtype=np.float64)

        shoulder_center = 0.5 * (left_inner + right_inner)
        shoulder_span = float(np.linalg.norm(right_outer - left_outer))
        shoulder_span = max(shoulder_span, float(frame_width) * 0.10, 1.0)

        points = np.vstack(
            [left_eye, right_eye, chin, left_inner, left_outer, right_inner, right_outer]
        )
        normalized_points = ((points - shoulder_center) / shoulder_span).reshape(-1)

        eye_mid = 0.5 * (left_eye + right_eye)
        eye_vector = (right_eye - left_eye) / shoulder_span
        chin_vector = (chin - eye_mid) / shoulder_span
        left_shoulder_vector = (left_outer - left_inner) / shoulder_span
        right_shoulder_vector = (right_outer - right_inner) / shoulder_span

        eye_width = float(np.linalg.norm(right_eye - left_eye) / shoulder_span)
        face_height = float(np.linalg.norm(chin - eye_mid) / shoulder_span)
        outer_tilt = float((right_outer[1] - left_outer[1]) / shoulder_span)
        inner_tilt = float((right_inner[1] - left_inner[1]) / shoulder_span)
        eye_to_shoulders = float((shoulder_center[1] - eye_mid[1]) / shoulder_span)
        chin_to_shoulders = float((shoulder_center[1] - chin[1]) / shoulder_span)

        vector = np.concatenate(
            [
                normalized_points,
                eye_vector,
                chin_vector,
                left_shoulder_vector,
                right_shoulder_vector,
                np.asarray(
                    [
                        eye_width,
                        face_height,
                        outer_tilt,
                        inner_tilt,
                        eye_to_shoulders,
                        chin_to_shoulders,
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        if vector.shape[0] != self.FEATURE_LENGTH:
            raise RuntimeError(f"Unexpected geometry feature length: {vector.shape[0]}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Geometry produced non-finite posture features.")
        return vector.astype(np.float32)
