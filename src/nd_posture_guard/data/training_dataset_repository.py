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
from nd_posture_guard.training.shoulder_training_profile import ShoulderTrainingProfile
from nd_posture_guard.vision.shoulder_feature_extractor import ShoulderFeatureExtractor


class TrainingDatasetRepository:
    """Append-only storage plus rebuildable personal shoulder training data.

    Raw frames and the six clicked shoulder points are the durable source of truth.
    Runtime descriptors are rebuilt from those raw samples whenever the classifier
    profile is loaded. This allows newer compatible algorithms to reuse old videos
    instead of replacing the user's training dataset.
    """

    DATA_SCHEMA_VERSION = 1
    CAPTURE_FEATURE_SCHEMA = "shoulder_local_numpy_hog_v2"
    RUNTIME_FEATURE_SCHEMA = "shoulder_fixed_camera_spatial_v4"

    def __init__(
        self,
        root: Path,
        novelty_multiplier: float,
        recording_fps: float,
    ) -> None:
        self._root = root.expanduser()
        self._novelty_multiplier = max(1.5, float(novelty_multiplier))
        self._recording_fps = max(1.0, float(recording_fps))
        self._ensure_layout()

    @property
    def root(self) -> Path:
        return self._root

    def sample_counts(self) -> tuple[int, int]:
        good = len(tuple((self._root / "samples" / "good").glob("*/metadata.json")))
        bad = len(tuple((self._root / "samples" / "bad").glob("*/metadata.json")))
        return good, bad

    def usable_sample_counts(self) -> tuple[int, int]:
        samples = self._load_raw_samples()
        good = sum(1 for entry in samples if entry["label"] == 0)
        bad = sum(1 for entry in samples if entry["label"] == 1)
        return good, bad

    def list_samples(self) -> list[TrainingSampleSummary]:
        """Return saved samples newest-first for the review UI."""
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
                        )
                    )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
        result.sort(key=lambda sample: (sample.created_at_utc, sample.sample_id), reverse=True)
        return result

    def delete_sample(self, sample_id: str) -> bool:
        """Delete one exact saved GOOD/BAD sample and refresh the manifest."""
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
        left_roi: tuple[int, int, int, int],
        right_roi: tuple[int, int, int, int],
        anchor_frame: NDArray[np.uint8],
        frames: tuple[NDArray[np.uint8], ...],
        features: NDArray[np.float32],
    ) -> Path:
        if label not in (0, 1):
            raise ValueError("Training label must be 0 (good) or 1 (bad).")
        if len(points) != 6:
            raise ValueError("Exactly six shoulder points are required for a sample.")
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("A training sample must contain descriptor frames.")
        if not frames:
            raise ValueError("A training sample must contain raw camera frames.")

        label_name = "good" if label == 0 else "bad"
        sample_id = self._new_sample_id()
        sample_dir = self._root / "samples" / label_name / sample_id
        sample_dir.mkdir(parents=True, exist_ok=False)

        height, width = anchor_frame.shape[:2]
        np.save(sample_dir / "features.npy", features.astype(np.float32))
        np.savez_compressed(sample_dir / "frames.npz", frames=np.asarray(frames, dtype=np.uint8))
        cv2.imwrite(str(sample_dir / "anchor_frame.jpg"), anchor_frame)
        video_name = self._write_video(sample_dir, frames, width, height)

        metadata = {
            "sample_id": sample_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_schema_version": self.DATA_SCHEMA_VERSION,
            "feature_schema": self.CAPTURE_FEATURE_SCHEMA,
            "runtime_feature_schema": self.RUNTIME_FEATURE_SCHEMA,
            "feature_length": int(features.shape[1]),
            "label": int(label),
            "label_name": label_name,
            "frame_width": int(width),
            "frame_height": int(height),
            "frame_count": int(len(frames)),
            "points_normalized": [
                [float(x) / max(width, 1), float(y) / max(height, 1)] for x, y in points
            ],
            "left_roi_normalized": self._normalize_roi(left_roi, width, height),
            "right_roi_normalized": self._normalize_roi(right_roi, width, height),
            "media": {
                "anchor_frame": "anchor_frame.jpg",
                "frames_archive": "frames.npz",
                "video": video_name,
            },
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._update_manifest()
        return sample_dir

    def import_legacy_profile(
        self,
        path: Path,
        frame_width: int,
        frame_height: int,
        expected_feature_length: int,
    ) -> bool:
        """Preserve the previous v0.19 profile once.

        v0.21 cannot use profile-only legacy data for its fixed-camera geometry
        model because those files do not contain raw frames and six shoulder points.
        They are still imported/preserved so no historical user data is deleted.
        """
        source = path.expanduser()
        marker = self._root / "legacy" / "legacy_shoulder_training_profile_v2_imported.json"
        if marker.exists() or not source.is_file():
            return False
        try:
            with np.load(source, allow_pickle=False) as data:
                features = np.asarray(data["features"], dtype=np.float32)
                labels = np.asarray(data["labels"], dtype=np.int8)
                left_roi = tuple(int(value) for value in data["left_roi"].tolist())
                right_roi = tuple(int(value) for value in data["right_roi"].tolist())
        except (OSError, ValueError, KeyError):
            return False
        if features.ndim != 2 or features.shape[1] != expected_feature_length:
            return False
        if features.shape[0] != labels.shape[0]:
            return False

        imported = 0
        for label in (0, 1):
            class_features = features[labels == label]
            if len(class_features) == 0:
                continue
            label_name = "good" if label == 0 else "bad"
            sample_id = f"legacy_v019_{label_name}_{uuid.uuid4().hex[:8]}"
            sample_dir = self._root / "samples" / label_name / sample_id
            sample_dir.mkdir(parents=True, exist_ok=False)
            np.save(sample_dir / "features.npy", class_features.astype(np.float32))
            metadata = {
                "sample_id": sample_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "data_schema_version": self.DATA_SCHEMA_VERSION,
                "feature_schema": "shoulder_numpy_hog_v2",
                "runtime_feature_schema": None,
                "feature_length": int(features.shape[1]),
                "label": label,
                "label_name": label_name,
                "frame_width": int(frame_width),
                "frame_height": int(frame_height),
                "frame_count": int(len(class_features)),
                "points_normalized": [],
                "left_roi_normalized": self._normalize_roi(left_roi, frame_width, frame_height),
                "right_roi_normalized": self._normalize_roi(right_roi, frame_width, frame_height),
                "media": {"anchor_frame": None, "frames_archive": None, "video": None},
                "source": "legacy shoulder_training_profile_v2.npz",
            }
            (sample_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            imported += 1
        marker.write_text(
            json.dumps(
                {
                    "source": str(source),
                    "imported_at_utc": datetime.now(timezone.utc).isoformat(),
                    "imported_class_groups": imported,
                    "note": "Preserved only; fixed-camera v3 requires raw frames plus six points.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._update_manifest()
        return imported > 0

    def build_profile(
        self,
        frame_width: int,
        frame_height: int,
        feature_extractor: ShoulderFeatureExtractor,
    ) -> ShoulderTrainingProfile | None:
        samples = self._load_raw_samples()
        if not samples or {int(entry["label"]) for entry in samples} != {0, 1}:
            return None

        left_roi_normalized, right_roi_normalized = self._fixed_analysis_rois(samples)
        left_runtime = self._denormalize_roi(left_roi_normalized, frame_width, frame_height)
        right_runtime = self._denormalize_roi(right_roi_normalized, frame_width, frame_height)

        prototypes: list[NDArray[np.float32]] = []
        labels: list[int] = []
        for entry in samples:
            frames = self._load_frames(Path(entry["frames_path"]))
            if frames is None or len(frames) == 0:
                continue
            frame_h, frame_w = frames[0].shape[:2]
            left_roi = self._denormalize_roi(left_roi_normalized, frame_w, frame_h)
            right_roi = self._denormalize_roi(right_roi_normalized, frame_w, frame_h)
            frame_features = np.vstack(
                [feature_extractor.extract(frame, left_roi, right_roi) for frame in frames]
            ).astype(np.float32)
            for prototype in self._chunk_prototypes(frame_features, chunks=3):
                prototypes.append(prototype)
                labels.append(int(entry["label"]))

        if not prototypes or set(labels) != {0, 1}:
            return None
        all_features = np.vstack(prototypes).astype(np.float32)
        all_labels = np.asarray(labels, dtype=np.int8)
        novelty_threshold = self._estimate_novelty_threshold(all_features, all_labels)
        return ShoulderTrainingProfile(
            left_roi=left_runtime,
            right_roi=right_runtime,
            features=all_features,
            labels=all_labels,
            novelty_threshold=novelty_threshold,
        )

    def _load_raw_samples(self) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for label_name in ("good", "bad"):
            for metadata_path in sorted((self._root / "samples" / label_name).glob("*/metadata.json")):
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if int(metadata.get("data_schema_version", -1)) != self.DATA_SCHEMA_VERSION:
                        continue
                    points = metadata.get("points_normalized", [])
                    if not isinstance(points, list) or len(points) != 6:
                        continue
                    normalized_points = tuple((float(point[0]), float(point[1])) for point in points)
                    frames_path = metadata_path.parent / "frames.npz"
                    if not frames_path.is_file():
                        continue
                    label = int(metadata.get("label", -1))
                    if label not in (0, 1):
                        continue
                    entries.append(
                        {
                            "label": label,
                            "points_normalized": normalized_points,
                            "frames_path": frames_path,
                        }
                    )
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, IndexError):
                    continue
        return entries

    @staticmethod
    def _fixed_analysis_rois(
        samples: list[dict[str, object]],
    ) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
        point_sets = np.asarray([entry["points_normalized"] for entry in samples], dtype=np.float64)
        left = point_sets[:, :3, :]
        right = point_sets[:, 3:, :]
        all_y = point_sets[:, :, 1]

        y_min = float(np.min(all_y))
        y_max = float(np.max(all_y))
        y_motion = max(0.02, y_max - y_min)
        top = y_min - max(0.045, 0.35 * y_motion)
        bottom = y_max + max(0.15, 1.20 * y_motion)

        # Fixed camera-space ROIs deliberately do NOT follow each sample's shoulder
        # position. Shoulder drop/forward slouch therefore changes where the edge
        # appears inside the ROI and remains visible to the descriptor.
        left_x1 = float(np.min(left[:, :, 0])) - 0.075
        left_x2 = float(np.max(left[:, :, 0])) + 0.025
        right_x1 = float(np.min(right[:, :, 0])) - 0.025
        right_x2 = float(np.max(right[:, :, 0])) + 0.075

        left_roi = TrainingDatasetRepository._clip_normalized_roi((left_x1, top, left_x2, bottom))
        right_roi = TrainingDatasetRepository._clip_normalized_roi((right_x1, top, right_x2, bottom))
        return left_roi, right_roi

    @staticmethod
    def _chunk_prototypes(features: NDArray[np.float32], chunks: int) -> list[NDArray[np.float32]]:
        count = min(max(1, int(chunks)), len(features))
        result: list[NDArray[np.float32]] = []
        for chunk in np.array_split(features, count):
            prototype = np.median(chunk, axis=0).astype(np.float32)
            norm = float(np.linalg.norm(prototype))
            if norm > 1e-6:
                prototype /= norm
            result.append(prototype)
        return result

    def _estimate_novelty_threshold(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int8],
    ) -> float:
        nearest: list[float] = []
        for index, feature in enumerate(features):
            same_class = np.where(labels == labels[index])[0]
            same_class = same_class[same_class != index]
            if same_class.size == 0:
                continue
            distances = np.linalg.norm(features[same_class] - feature, axis=1)
            nearest.append(float(np.min(distances)))
        if not nearest:
            return 1.5
        base = float(np.percentile(np.asarray(nearest), 99.0))
        # Unit-normalized feature vectors have a theoretical maximum L2 distance
        # of 2. Keep UNKNOWN conservative so bad posture is not silently suppressed.
        return min(2.0, max(1.60, base * self._novelty_multiplier))

    @staticmethod
    def _load_frames(path: Path) -> list[NDArray[np.uint8]] | None:
        try:
            with np.load(path, allow_pickle=False) as data:
                values = np.asarray(data["frames"], dtype=np.uint8)
        except (OSError, ValueError, KeyError):
            return None
        if values.ndim != 4 or values.shape[0] == 0:
            return None
        # Thirty frames per sample are small, but cap future larger archives while
        # preserving the beginning/middle/end of the clip.
        if len(values) > 30:
            indices = np.linspace(0, len(values) - 1, 30, dtype=np.int32)
            values = values[indices]
        return [frame for frame in values]

    def _ensure_layout(self) -> None:
        (self._root / "samples" / "good").mkdir(parents=True, exist_ok=True)
        (self._root / "samples" / "bad").mkdir(parents=True, exist_ok=True)
        (self._root / "runtime").mkdir(parents=True, exist_ok=True)
        (self._root / "legacy").mkdir(parents=True, exist_ok=True)
        manifest = self._root / "dataset_manifest.json"
        if not manifest.exists():
            self._update_manifest()

    def _update_manifest(self) -> None:
        good, bad = self.sample_counts()
        manifest_path = self._root / "dataset_manifest.json"
        previous: dict[str, object] = {}
        if manifest_path.exists():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        versions = set(str(value) for value in previous.get("compatible_feature_schemas_seen", []))
        versions.update({self.CAPTURE_FEATURE_SCHEMA, self.RUNTIME_FEATURE_SCHEMA})
        manifest = {
            "data_schema_version": self.DATA_SCHEMA_VERSION,
            "current_runtime_feature_schema": self.RUNTIME_FEATURE_SCHEMA,
            "capture_feature_schema": self.CAPTURE_FEATURE_SCHEMA,
            "compatible_feature_schemas_seen": sorted(versions),
            "merge_policy": "append raw compatible samples; never replace existing training samples",
            "good_sample_count": good,
            "bad_sample_count": bad,
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
        if not frames:
            return None
        path = sample_dir / "clip.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), self._recording_fps, (width, height)
        )
        if not writer.isOpened():
            return None
        try:
            for frame in frames:
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()
        return path.name if path.is_file() and path.stat().st_size > 0 else None

    @staticmethod
    def _new_sample_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        return f"{stamp}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _normalize_roi(roi: tuple[int, int, int, int], width: int, height: int) -> list[float]:
        x1, y1, x2, y2 = roi
        return [x1 / width, y1 / height, x2 / width, y2 / height]

    @staticmethod
    def _clip_normalized_roi(
        roi: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = roi
        x1 = min(max(0.0, x1), 0.98)
        y1 = min(max(0.0, y1), 0.98)
        x2 = min(max(x1 + 0.02, x2), 1.0)
        y2 = min(max(y1 + 0.05, y2), 1.0)
        return x1, y1, x2, y2

    @staticmethod
    def _denormalize_roi(
        roi: tuple[float, float, float, float], width: int, height: int
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = roi
        left = max(0, min(width - 2, round(x1 * width)))
        top = max(0, min(height - 2, round(y1 * height)))
        right = max(left + 1, min(width, round(x2 * width)))
        bottom = max(top + 1, min(height, round(y2 * height)))
        return left, top, right, bottom
