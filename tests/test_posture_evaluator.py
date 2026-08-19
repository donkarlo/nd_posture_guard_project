from nd_posture_guard.posture.posture_evaluator import PostureEvaluator
from nd_posture_guard.posture.posture_reference import PostureReference
from nd_posture_guard.posture.shoulder_calibration_profile import ShoulderCalibrationProfile
from nd_posture_guard.vision.tracked_shoulders import TrackedShoulders


def test_shoulder_drop_triggers_slouch_without_head_information() -> None:
    center = TrackedShoulders((120, 250), (520, 250))
    profiles = tuple(ShoulderCalibrationProfile.from_shoulders(center, 640, 480) for _ in range(3))
    evaluator = PostureEvaluator(7.0, 12.0, 12.0, 1)
    evaluator.calibrate(PostureReference(center, profiles, 640, 480))
    evaluation = evaluator.evaluate(TrackedShoulders((120, 285), (520, 285)), 640, 480)
    assert evaluation.state == evaluator.STATE_SLOUCH
    assert evaluation.should_alert
