import numpy as np

from nd_posture_guard.model.example_posture_classifier import ExamplePostureClassifier
from nd_posture_guard.training.shoulder_training_profile import ShoulderTrainingProfile


def _profile() -> ShoulderTrainingProfile:
    good = np.asarray([[1.0, 0.0], [0.98, 0.05], [0.97, 0.08]], dtype=np.float32)
    bad = np.asarray([[0.0, 1.0], [0.05, 0.98], [0.08, 0.97]], dtype=np.float32)
    good /= np.linalg.norm(good, axis=1, keepdims=True)
    bad /= np.linalg.norm(bad, axis=1, keepdims=True)
    return ShoulderTrainingProfile(
        (0, 0, 10, 10),
        (20, 0, 30, 10),
        np.vstack([good, bad]),
        np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8),
        2.0,
    )


def test_classifier_distinguishes_good_and_bad_examples() -> None:
    classifier = ExamplePostureClassifier(0.50, 1, 3)
    classifier.set_profile(_profile())
    result = classifier.classify(np.asarray([0.99, 0.04], dtype=np.float32))
    assert result.state == "good"
    assert result.bad_score < 0.5

    classifier = ExamplePostureClassifier(0.50, 1, 3)
    classifier.set_profile(_profile())
    result = classifier.classify(np.asarray([0.04, 0.99], dtype=np.float32))
    assert result.state == "bad"
    assert result.bad_score > 0.5
    assert result.should_alert


def test_beep_threshold_does_not_hide_bad_diagnostic_state() -> None:
    classifier = ExamplePostureClassifier(0.95, 1, 3)
    classifier.set_profile(_profile())
    vector = np.asarray([0.35, 0.94], dtype=np.float32)
    vector /= np.linalg.norm(vector)
    result = classifier.classify(vector)
    assert result.bad_score > 0.5
    assert result.state == "bad"
    assert not result.should_alert

    classifier.set_bad_score_threshold(0.50)
    result = classifier.classify(vector)
    assert result.state == "bad"
    assert result.should_alert
