import numpy as np

from nd_posture_guard.model.example_posture_classifier import ExamplePostureClassifier
from nd_posture_guard.training.posture_training_profile import PostureTrainingProfile


def _profile() -> PostureTrainingProfile:
    good = np.asarray([[0.0, 0.0], [0.04, -0.02], [-0.03, 0.03]], dtype=np.float32)
    bad = np.asarray([[1.0, 1.0], [0.96, 1.03], [1.05, 0.97]], dtype=np.float32)
    return PostureTrainingProfile(
        reference_points_normalized=((0.4, 0.3),) * 7,
        point_templates=tuple((np.zeros((29, 29), dtype=np.uint8),) for _ in range(7)),
        features=np.vstack([good, bad]),
        labels=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8),
    )


def test_classifier_distinguishes_good_and_bad_geometry_examples() -> None:
    classifier = ExamplePostureClassifier(0.50, 1, 3)
    classifier.set_profile(_profile())
    good = classifier.classify(np.asarray([0.02, 0.01], dtype=np.float32))
    assert good.state == "good"
    assert good.bad_score < 0.5

    classifier.set_profile(_profile())
    bad = classifier.classify(np.asarray([1.02, 0.99], dtype=np.float32))
    assert bad.state == "bad"
    assert bad.bad_score > 0.5
    assert bad.should_alert


def test_beep_threshold_does_not_hide_bad_diagnostic_state() -> None:
    classifier = ExamplePostureClassifier(0.95, 1, 3)
    classifier.set_profile(_profile())
    vector = np.asarray([0.8, 0.85], dtype=np.float32)
    result = classifier.classify(vector)
    assert result.state == "bad"
    assert not result.should_alert
    classifier.set_bad_score_threshold(0.50)
    result = classifier.classify(vector)
    assert result.state == "bad"
    assert result.should_alert
