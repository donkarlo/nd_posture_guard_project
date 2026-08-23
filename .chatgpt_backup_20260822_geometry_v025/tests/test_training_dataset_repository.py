import json
from pathlib import Path

import cv2
import numpy as np

from nd_posture_guard.data.training_dataset_repository import TrainingDatasetRepository
from nd_posture_guard.vision.shoulder_feature_extractor import ShoulderFeatureExtractor


def _frame_with_shoulders(y: int, width: int = 160, height: int = 120) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Shoulder-like bright sloping edges. BAD examples move them downward.
    cv2.line(frame, (18, y + 10), (70, y), (255, 255, 255), 5)
    cv2.line(frame, (90, y), (142, y + 10), (255, 255, 255), 5)
    cv2.rectangle(frame, (45, y + 12), (115, min(height - 1, y + 45)), (80, 80, 80), -1)
    return frame


def _points(y: int) -> tuple[tuple[int, int], ...]:
    return ((70, y), (48, y + 4), (22, y + 9), (90, y), (112, y + 4), (138, y + 9))


def _sample(
    repo: TrainingDatasetRepository,
    label: int,
    y: int,
    feature_extractor: ShoulderFeatureExtractor,
) -> None:
    frame = _frame_with_shoulders(y)
    frames = tuple(frame.copy() for _ in range(9))
    # Capture-time local features intentionally use a moving ROI. The v0.21
    # runtime profile must ignore these and rebuild from raw frames in fixed ROIs.
    local_left = (10, max(0, y - 10), 75, min(120, y + 55))
    local_right = (85, max(0, y - 10), 150, min(120, y + 55))
    features = np.vstack([
        feature_extractor.extract(item, local_left, local_right) for item in frames
    ])
    repo.save_sample(
        label=label,
        points=_points(y),
        left_roi=local_left,
        right_roi=local_right,
        anchor_frame=frame,
        frames=frames,
        features=features,
    )


def test_dataset_directory_is_created_and_samples_append(tmp_path: Path) -> None:
    root = tmp_path / "data" / "nd_posture_guard_project"
    repo = TrainingDatasetRepository(root, novelty_multiplier=3.0, recording_fps=10.0)
    extractor = ShoulderFeatureExtractor()
    assert root.is_dir()
    _sample(repo, 0, 45, extractor)
    _sample(repo, 0, 47, extractor)
    _sample(repo, 1, 75, extractor)
    assert repo.sample_counts() == (2, 1)
    assert repo.usable_sample_counts() == (2, 1)


def test_old_raw_samples_are_rebuilt_into_fixed_camera_profile(tmp_path: Path) -> None:
    repo = TrainingDatasetRepository(tmp_path, novelty_multiplier=4.5, recording_fps=10.0)
    extractor = ShoulderFeatureExtractor()
    _sample(repo, 0, 44, extractor)
    _sample(repo, 0, 47, extractor)
    _sample(repo, 1, 74, extractor)
    _sample(repo, 1, 78, extractor)

    # Simulate v0.20 metadata. v0.21 must use raw frames/points rather than reject
    # the samples merely because their old stored feature schema is v2.
    for metadata_path in tmp_path.glob("samples/*/*/metadata.json"):
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw["feature_schema"] = "shoulder_numpy_hog_v2"
        raw.pop("runtime_feature_schema", None)
        metadata_path.write_text(json.dumps(raw), encoding="utf-8")

    profile = repo.build_profile(160, 120, extractor)
    assert profile is not None
    assert set(profile.labels.tolist()) == {0, 1}
    assert profile.features.shape[1] == extractor.feature_length
    # Fixed analysis ROIs must cover the vertical displacement rather than being
    # recentered separately around GOOD and BAD samples.
    assert profile.left_roi[1] < 44
    assert profile.left_roi[3] > 78


def test_schema_incompatible_raw_sample_is_preserved_but_not_used(tmp_path: Path) -> None:
    repo = TrainingDatasetRepository(tmp_path, novelty_multiplier=4.5, recording_fps=10.0)
    extractor = ShoulderFeatureExtractor()
    _sample(repo, 0, 45, extractor)
    _sample(repo, 1, 75, extractor)
    bad_metadata = next((tmp_path / "samples" / "bad").glob("*/metadata.json"))
    raw = json.loads(bad_metadata.read_text(encoding="utf-8"))
    raw["data_schema_version"] = 999
    bad_metadata.write_text(json.dumps(raw), encoding="utf-8")
    assert bad_metadata.exists()
    assert repo.build_profile(160, 120, extractor) is None


def test_legacy_profile_is_preserved_once_but_requires_raw_frames_for_v3(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy_profile.npz"
    extractor = ShoulderFeatureExtractor()
    np.savez_compressed(
        legacy,
        left_roi=np.asarray([10, 40, 70, 100], dtype=np.int32),
        right_roi=np.asarray([90, 40, 150, 100], dtype=np.int32),
        features=np.vstack([
            np.full((2, extractor.feature_length), 0.1, dtype=np.float32),
            np.full((2, extractor.feature_length), 0.9, dtype=np.float32),
        ]),
        labels=np.asarray([0, 0, 1, 1], dtype=np.int8),
        novelty_threshold=np.asarray([1.0], dtype=np.float32),
    )
    root = tmp_path / "dataset"
    repo = TrainingDatasetRepository(root, novelty_multiplier=3.0, recording_fps=10.0)
    assert repo.import_legacy_profile(legacy, 160, 120, extractor.feature_length)
    assert not repo.import_legacy_profile(legacy, 160, 120, extractor.feature_length)
    assert repo.sample_counts() == (1, 1)
    assert repo.usable_sample_counts() == (0, 0)
    assert repo.build_profile(160, 120, extractor) is None


def test_samples_can_be_listed_newest_first_and_deleted(tmp_path: Path) -> None:
    repo = TrainingDatasetRepository(tmp_path, novelty_multiplier=4.5, recording_fps=10.0)
    extractor = ShoulderFeatureExtractor()
    _sample(repo, 0, 45, extractor)
    _sample(repo, 1, 75, extractor)

    samples = repo.list_samples()
    assert len(samples) == 2
    assert {sample.label for sample in samples} == {0, 1}
    assert all(sample.anchor_frame_path is not None for sample in samples)
    assert all(sample.video_path is not None for sample in samples)

    target = samples[0]
    assert repo.delete_sample(target.sample_id)
    assert not target.sample_dir.exists()
    assert len(repo.list_samples()) == 1
    assert not repo.delete_sample(target.sample_id)


def test_rebuilt_profile_classifies_vertical_bad_posture_as_bad(tmp_path: Path) -> None:
    from nd_posture_guard.model.example_posture_classifier import ExamplePostureClassifier

    repo = TrainingDatasetRepository(tmp_path, novelty_multiplier=4.5, recording_fps=10.0)
    extractor = ShoulderFeatureExtractor()
    _sample(repo, 0, 42, extractor)
    _sample(repo, 0, 46, extractor)
    _sample(repo, 1, 72, extractor)
    _sample(repo, 1, 78, extractor)

    profile = repo.build_profile(160, 120, extractor)
    assert profile is not None
    classifier = ExamplePostureClassifier(0.50, 1, 3)
    classifier.set_profile(profile)

    good_feature = extractor.extract(_frame_with_shoulders(44), profile.left_roi, profile.right_roi)
    bad_feature = extractor.extract(_frame_with_shoulders(76), profile.left_roi, profile.right_roi)

    good_result = classifier.classify(good_feature)
    assert good_result.state == "good"
    classifier.set_profile(profile)
    bad_result = classifier.classify(bad_feature)
    assert bad_result.state == "bad"
    assert bad_result.bad_score > 0.50
    assert bad_result.should_alert
