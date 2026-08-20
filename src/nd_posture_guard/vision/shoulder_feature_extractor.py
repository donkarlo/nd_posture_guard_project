from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


class ShoulderFeatureExtractor:
    """Extract a posture-sensitive fixed-camera shoulder descriptor.

    Earlier versions relied mostly on a normalized HOG-like texture descriptor.
    That descriptor was good at recognizing the shirt/shoulder appearance but was
    too invariant to the vertical movement that matters for slouch detection.

    This version keeps the texture descriptor, but adds two strongly spatial
    components for each shoulder:

    * a low-resolution map of vertical edge energy, and
    * a row-by-row edge profile.

    Because the analysis ROIs are fixed in camera coordinates, moving the shoulder
    downward or forward moves these spatial features inside the descriptor instead
    of being normalized away. No point is tracked during monitoring.
    """

    RUNTIME_SCHEMA = "shoulder_fixed_camera_spatial_v4"

    def __init__(
        self,
        width: int = 96,
        height: int = 80,
        cell_size: int = 8,
        orientation_bins: int = 9,
        spatial_width: int = 24,
        spatial_height: int = 20,
    ) -> None:
        self._size = (int(width), int(height))
        self._cell_size = max(4, int(cell_size))
        self._orientation_bins = max(6, int(orientation_bins))
        self._spatial_size = (max(8, int(spatial_width)), max(8, int(spatial_height)))

        if self._size[0] % self._cell_size != 0 or self._size[1] % self._cell_size != 0:
            raise ValueError("Feature image dimensions must be divisible by cell_size.")

    @property
    def feature_length(self) -> int:
        cells_x = self._size[0] // self._cell_size
        cells_y = self._size[1] // self._cell_size
        hog_like = cells_x * cells_y * (self._orientation_bins + 2)
        spatial_map = self._spatial_size[0] * self._spatial_size[1]
        row_profile = self._size[1]
        return 2 * (hog_like + spatial_map + row_profile)

    def extract(
        self,
        frame: NDArray[np.uint8],
        left_roi: tuple[int, int, int, int],
        right_roi: tuple[int, int, int, int],
    ) -> NDArray[np.float32]:
        left = self._descriptor(self._crop(frame, left_roi))
        right = self._descriptor(self._crop(frame, right_roi))
        feature = np.concatenate([left, right]).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(feature))
        if norm > 1e-6:
            feature = feature / norm
        return feature

    def _descriptor(self, crop: NDArray[np.uint8]) -> NDArray[np.float32]:
        resized = cv2.resize(crop, self._size, interpolation=cv2.INTER_AREA)
        gray = self._contrast_normalize(self._to_gray(resized))

        gradient_y, gradient_x = np.gradient(gray)
        magnitude = np.hypot(gradient_x, gradient_y)
        orientation = (np.degrees(np.arctan2(gradient_y, gradient_x)) + 180.0) % 180.0

        hog_like = self._hog_like(gray, magnitude, orientation)

        # A shoulder boundary is predominantly expressed by change in the vertical
        # image direction. Blur only suppresses shirt texture; coordinates remain
        # fixed, so a dropped shoulder changes the map location strongly.
        vertical_edge = np.abs(gradient_y).astype(np.float32)
        vertical_edge = cv2.GaussianBlur(vertical_edge, (5, 5), 0)

        spatial = cv2.resize(vertical_edge, self._spatial_size, interpolation=cv2.INTER_AREA)
        spatial = spatial.reshape(-1).astype(np.float32)
        spatial_norm = float(np.linalg.norm(spatial))
        if spatial_norm > 1e-6:
            spatial /= spatial_norm

        row_profile = np.mean(vertical_edge, axis=1).astype(np.float32)
        row_profile = np.convolve(row_profile, np.ones(5, dtype=np.float32) / 5.0, mode="same")
        row_norm = float(np.linalg.norm(row_profile))
        if row_norm > 1e-6:
            row_profile /= row_norm

        # Spatial location is deliberately weighted more strongly than texture.
        vector = np.concatenate(
            [
                0.75 * hog_like,
                4.0 * spatial,
                6.0 * row_profile,
            ]
        ).astype(np.float32)
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm > 1e-6:
            vector /= vector_norm
        return vector

    def _hog_like(
        self,
        gray: NDArray[np.float32],
        magnitude: NDArray[np.float32],
        orientation: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        cells_y = self._size[1] // self._cell_size
        cells_x = self._size[0] // self._cell_size
        descriptor = np.zeros(
            (cells_y, cells_x, self._orientation_bins + 2),
            dtype=np.float32,
        )

        for cell_y in range(cells_y):
            y1 = cell_y * self._cell_size
            y2 = y1 + self._cell_size
            for cell_x in range(cells_x):
                x1 = cell_x * self._cell_size
                x2 = x1 + self._cell_size
                cell_orientation = orientation[y1:y2, x1:x2].reshape(-1)
                cell_magnitude = magnitude[y1:y2, x1:x2].reshape(-1)
                histogram, _ = np.histogram(
                    cell_orientation,
                    bins=self._orientation_bins,
                    range=(0.0, 180.0),
                    weights=cell_magnitude,
                )
                cell_gray = gray[y1:y2, x1:x2]
                descriptor[cell_y, cell_x, : self._orientation_bins] = histogram.astype(np.float32)
                descriptor[cell_y, cell_x, self._orientation_bins] = float(np.mean(cell_gray))
                descriptor[cell_y, cell_x, self._orientation_bins + 1] = float(np.std(cell_gray))

        vector = descriptor.reshape(-1)
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm > 1e-6:
            vector = vector / vector_norm
        return vector.astype(np.float32, copy=False)

    @staticmethod
    def _to_gray(image: NDArray[np.uint8]) -> NDArray[np.float32]:
        if image.ndim == 2:
            return image.astype(np.float32) / 255.0
        if image.ndim != 3 or image.shape[2] < 3:
            raise RuntimeError("Unexpected camera image format.")
        blue = image[..., 0].astype(np.float32)
        green = image[..., 1].astype(np.float32)
        red = image[..., 2].astype(np.float32)
        gray = 0.114 * blue + 0.587 * green + 0.299 * red
        return gray / 255.0

    @staticmethod
    def _contrast_normalize(gray: NDArray[np.float32]) -> NDArray[np.float32]:
        low = float(np.percentile(gray, 2.0))
        high = float(np.percentile(gray, 98.0))
        if high - low < 1e-6:
            return np.clip(gray, 0.0, 1.0)
        return np.clip((gray - low) / (high - low), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _crop(
        frame: NDArray[np.uint8],
        roi: tuple[int, int, int, int],
    ) -> NDArray[np.uint8]:
        x1, y1, x2, y2 = roi
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, x1))
        x2 = max(x1 + 1, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(y1 + 1, min(height, y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise RuntimeError("The learned shoulder region is outside the current camera frame.")
        return crop
