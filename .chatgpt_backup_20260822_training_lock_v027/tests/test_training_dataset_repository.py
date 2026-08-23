import json
from pathlib import Path

import cv2
import numpy as np

from nd_posture_guard.data.training_dataset_repository import TrainingDatasetRepository
from nd_posture_guard.model.posture_geometry import PostureGeometry
from nd_posture_guard.vision.posture_geometry_feature_extractor import PostureGeometryFeatureExtractor


def _geometry(bad: bool = False) -> PostureGeometry:
    return PostureGeometry.from_points((
        (270, 150), (330, 150), (300, 245 if bad else 220),
        (245, 300 if bad else 280), (150, 325 if bad else 300),
        (355, 300 if bad else 280), (450, 325 if bad else 300),
    ))


def _frame(seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _sample(repo, label, geom, extractor):
    frame = _frame(label + 2)
    repo.save_sample(
        label=label,
        points=geom.points,
        anchor_frame=frame,
        frames=(frame, frame),
        feature=extractor.extract(geom, 640, 480),
    )


def test_geometry_samples_append_build_profile_and_delete(tmp_path: Path) -> None:
    repo = TrainingDatasetRepository(tmp_path, novelty_multiplier=4.5, recording_fps=10.0)
    extractor = PostureGeometryFeatureExtractor()
    _sample(repo, 0, _geometry(False), extractor)
    _sample(repo, 1, _geometry(True), extractor)
    assert repo.sample_counts() == (1, 1)
    profile = repo.build_profile(extractor)
    assert profile is not None
    assert profile.features.shape == (2, extractor.feature_length)
    assert set(profile.labels.tolist()) == {0, 1}
    samples = repo.list_samples()
    assert all(sample.geometry_compatible for sample in samples)
    assert repo.delete_sample(samples[0].sample_id)
    assert len(repo.list_samples()) == 1


def test_legacy_six_point_sample_is_preserved_but_not_used(tmp_path: Path) -> None:
    repo = TrainingDatasetRepository(tmp_path, novelty_multiplier=4.5, recording_fps=10.0)
    extractor = PostureGeometryFeatureExtractor()
    _sample(repo, 0, _geometry(False), extractor)
    _sample(repo, 1, _geometry(True), extractor)

    legacy_dir = tmp_path / "samples" / "good" / "legacy_old"
    legacy_dir.mkdir(parents=True)
    frame = _frame(20)
    cv2.imwrite(str(legacy_dir / "anchor_frame.jpg"), frame)
    metadata = {
        "sample_id": "legacy_old",
        "data_schema_version": 1,
        "label": 0,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "frame_count": 1,
        "points_normalized": [[0.1, 0.1]] * 6,
        "media": {"anchor_frame": "anchor_frame.jpg", "video": None},
    }
    (legacy_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    assert repo.sample_counts() == (1, 1)
    assert repo.total_sample_counts() == (2, 1)
    listed = repo.list_samples()
    legacy = next(sample for sample in listed if sample.sample_id == "legacy_old")
    assert not legacy.geometry_compatible
    assert legacy.sample_dir.exists()
