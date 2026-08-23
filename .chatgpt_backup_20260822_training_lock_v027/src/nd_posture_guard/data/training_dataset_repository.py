from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

import cv2
import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.data.training_sample_summary import TrainingSampleSummary
from nd_posture_guard.model.posture_geometry import PostureGeometry
from nd_posture_guard.training.posture_training_profile import PostureTrainingProfile
from nd_posture_guard.vision.posture_geometry_feature_extractor import PostureGeometryFeatureExtractor


class TrainingDatasetRepository:
    """Persist raw clips while building the lightweight seven-point personal model."""

    DATA_SCHEMA_VERSION = 2
    GEOMETRY_SCHEMA = PostureGeometryFeatureExtractor.SCHEMA
    TEMPLATE_RADIUS = 14
    MAX_TEMPLATES_PER_POINT = 8

    def __init__(self, root: Path, novelty_multiplier: float, recording_fps: float) -> None:
        self._root = root.expanduser()
        # Kept in the constructor for settings/backward API compatibility. Geometry
        # v1 does not need high-dimensional novelty-distance tuning.
        self._novelty_multiplier = max(1.0, float(novelty_multiplier))
        self._recording_fps = max(1.0, float(recording_fps))
        self._ensure_layout()

    @property
    def root(self) -> Path:
        """Return the persistent dataset root."""
        return self._root

    def sample_counts(self) -> tuple[int, int]:
        """Return counts of samples usable by the current seven-point model."""
        samples = self._load_geometry_samples()
        good = sum(1 for entry in samples if int(entry["label"]) == 0)
        bad = sum(1 for entry in samples if int(entry["label"]) == 1)
        return good, bad

    def total_sample_counts(self) -> tuple[int, int]:
        """Return all persisted GOOD/BAD counts including legacy six-point clips."""
        good = len(tuple((self._root / "samples" / "good").glob("*/metadata.json")))
        bad = len(tuple((self._root / "samples" / "bad").glob("*/metadata.json")))
        return good, bad

    def usable_sample_counts(self) -> tuple[int, int]:
        """Alias current-model sample counts for callers from earlier versions."""
        return self.sample_counts()

    def list_samples(self) -> list[TrainingSampleSummary]:
        """Return every saved sample newest-first, marking geometry compatibility."""
        result: list[TrainingSampleSummary] = []
        for label_name in ("good", "bad"):
            for metadata_path in (self._root / "samples" / label_name).glob("*/metadata.json"):
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    label = int(metadata.get("label", -1))
                    if label not in (0, 1):
                        continue
                    sample_id = str(metadata.get("sample_id", metadata_path.parent.name))
                    media = metadata.get("media", {})
                    if not isinstance(media, dict):
                        media = {}
                    video_name = media.get("video")
                    anchor_name = media.get("anchor_frame")
                    video_path = metadata_path.parent / str(video_name) if video_name else None
                    anchor_path = metadata_path.parent / str(anchor_name) if anchor_name else None
                    points = metadata.get("points_normalized", [])
                    compatible = (
                        int(metadata.get("data_schema_version", -1)) == self.DATA_SCHEMA_VERSION
                        and str(metadata.get("geometry_schema", "")) == self.GEOMETRY_SCHEMA
                        and isinstance(points, list)
                        and len(points) == 7
                    )
                    result.append(
                        TrainingSampleSummary(
                            sample_id=sample_id,
                            label=label,
                            label_name="GOOD" if label == 0 else "BAD",
                            created_at_utc=str(metadata.get("created_at_utc", "")),
                            frame_count=int(metadata.get("frame_count", 0)),
                            sample_dir=metadata_path.parent,
                            video_path=video_path if video_path is not None and video_path.is_file() else None,
                            anchor_frame_path=(
                                anchor_path if anchor_path is not None and anchor_path.is_file() else None
                            ),
                            geometry_compatible=compatible,
                        )
                    )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
        result.sort(key=lambda sample: (sample.created_at_utc, sample.sample_id), reverse=True)
        return result

    def delete_sample(self, sample_id: str) -> bool:
        """Delete exactly one persisted sample directory and refresh the manifest."""
        if not sample_id or Path(sample_id).name != sample_id:
            return False
        for label_name in ("good", "bad"):
            sample_dir = self._root / "samples" / label_name / sample_id
            metadata_path = sample_dir / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if str(metadata.get("sample_id", sample_dir.name)) != sample_id:
                    return False
                shutil.rmtree(sample_dir)
            except (OSError, json.JSONDecodeError):
                return False
            self._update_manifest()
            return True
        return False

    def save_sample(
        self,
        label: int,
        points: tuple[tuple[float, float], ...],
        anchor_frame: NDArray[np.uint8],
        frames: tuple[NDArray[np.uint8], ...],
        feature: NDArray[np.float32],
    ) -> Path:
        """Append one seven-point sample without touching prior GOOD/BAD examples."""
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
            np.savez_compressed(sample_dir / "frames.npz", frames=np.asarray(frames, dtype=np.uint8))
            if not cv2.imwrite(str(sample_dir / "anchor_frame.jpg"), anchor_frame):
                raise OSError("Could not save the training anchor frame.")
            video_name = self._write_video(sample_dir, frames, width, height)
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
                    "frames_archive": "frames.npz",
                    "video": video_name,
                },
            }
            (sample_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            shutil.rmtree(sample_dir, ignore_errors=True)
            raise
        self._update_manifest()
        return sample_dir

    def build_profile(
        self,
        feature_extractor: PostureGeometryFeatureExtractor,
    ) -> PostureTrainingProfile | None:
        """Build the runtime model directly from compact geometry metadata and anchor templates."""
        samples = self._load_geometry_samples()
        if not samples or {int(entry["label"]) for entry in samples} != {0, 1}:
            return None

        features: list[NDArray[np.float32]] = []
        labels: list[int] = []
        normalized_point_sets: list[tuple[tuple[float, float], ...]] = []
        good_point_sets: list[tuple[tuple[float, float], ...]] = []
        templates_by_point: list[list[NDArray[np.uint8]]] = [[] for _ in range(7)]

        # Newest templates are kept first so a recently corrected sample naturally
        # replaces stale appearance without deleting historical videos.
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
            labels.append(int(entry["label"]))
            normalized_point_sets.append(normalized)
            if int(entry["label"]) == 0:
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

        if not features or set(labels) != {0, 1}:
            return None
        if any(not templates for templates in templates_by_point):
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

    def _load_geometry_samples(self) -> list[dict[str, object]]:
        """Load only schema-v2 seven-point samples while leaving legacy data untouched."""
        entries: list[dict[str, object]] = []
        for label_name in ("good", "bad"):
            for metadata_path in sorted((self._root / "samples" / label_name).glob("*/metadata.json")):
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if int(metadata.get("data_schema_version", -1)) != self.DATA_SCHEMA_VERSION:
                        continue
                    if str(metadata.get("geometry_schema", "")) != self.GEOMETRY_SCHEMA:
                        continue
                    points = metadata.get("points_normalized", [])
                    if not isinstance(points, list) or len(points) != 7:
                        continue
                    normalized = tuple((float(point[0]), float(point[1])) for point in points)
                    label = int(metadata.get("label", -1))
                    if label not in (0, 1):
                        continue
                    width = int(metadata.get("frame_width", 0))
                    height = int(metadata.get("frame_height", 0))
                    if width <= 0 or height <= 0:
                        continue
                    anchor_path = metadata_path.parent / "anchor_frame.jpg"
                    if not anchor_path.is_file():
                        continue
                    entries.append(
                        {
                            "label": label,
                            "points_normalized": normalized,
                            "frame_width": width,
                            "frame_height": height,
                            "anchor_path": anchor_path,
                        }
                    )
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, IndexError):
                    continue
        return entries

    def _ensure_layout(self) -> None:
        """Create missing persistent folders without replacing existing data."""
        (self._root / "samples" / "good").mkdir(parents=True, exist_ok=True)
        (self._root / "samples" / "bad").mkdir(parents=True, exist_ok=True)
        (self._root / "runtime").mkdir(parents=True, exist_ok=True)
        (self._root / "legacy").mkdir(parents=True, exist_ok=True)
        if not (self._root / "dataset_manifest.json").exists():
            self._update_manifest()

    def _update_manifest(self) -> None:
        """Update lightweight counts while preserving previously seen schema names."""
        usable_good, usable_bad = self.sample_counts()
        total_good, total_bad = self.total_sample_counts()
        manifest_path = self._root / "dataset_manifest.json"
        previous: dict[str, object] = {}
        if manifest_path.exists():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        seen = set(str(value) for value in previous.get("compatible_feature_schemas_seen", []))
        seen.add(self.GEOMETRY_SCHEMA)
        manifest = {
            "data_schema_version": self.DATA_SCHEMA_VERSION,
            "current_runtime_feature_schema": self.GEOMETRY_SCHEMA,
            "compatible_feature_schemas_seen": sorted(seen),
            "merge_policy": "append raw samples; never replace or delete legacy samples automatically",
            "usable_geometry_good_sample_count": usable_good,
            "usable_geometry_bad_sample_count": usable_bad,
            "total_good_sample_count": total_good,
            "total_bad_sample_count": total_bad,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_video(
        self,
        sample_dir: Path,
        frames: tuple[NDArray[np.uint8], ...],
        width: int,
        height: int,
    ) -> str | None:
        """Write the short review clip without making video availability model-critical."""
        path = sample_dir / "clip.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), self._recording_fps, (width, height)
        )
        if not writer.isOpened():
            return None
        try:
            for frame in frames:
                output = frame
                if frame.shape[1] != width or frame.shape[0] != height:
                    output = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(output)
        finally:
            writer.release()
        return path.name if path.is_file() and path.stat().st_size > 0 else None

    def _extract_template(
        self,
        gray: NDArray[np.uint8],
        x: float,
        y: float,
    ) -> NDArray[np.uint8] | None:
        """Cut a fixed-size grayscale appearance patch around one manually clicked landmark."""
        radius = self.TEMPLATE_RADIUS
        height, width = gray.shape[:2]
        cx = int(round(x))
        cy = int(round(y))
        x1, x2 = cx - radius, cx + radius + 1
        y1, y2 = cy - radius, cy + radius + 1
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return None
        patch = gray[y1:y2, x1:x2]
        if patch.shape != (2 * radius + 1, 2 * radius + 1):
            return None
        return patch.copy()

    @staticmethod
    def _to_gray(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convert a stored anchor frame into the tracker's grayscale representation."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.GaussianBlur(gray, (3, 3), 0)

    @staticmethod
    def _new_sample_id() -> str:
        """Create a sortable collision-resistant sample identifier."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        return f"{stamp}_{uuid.uuid4().hex[:8]}"
