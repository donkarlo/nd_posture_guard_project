import cv2
import numpy as np

from nd_posture_guard.posture.shoulder_calibration_pose import ShoulderCalibrationPose
from nd_posture_guard.vision.shoulder_anchor_tracker import ShoulderAnchorTracker


ANCHORS = np.array(
    [[180, 245], [150, 248], [115, 252], [460, 245], [490, 248], [525, 252]],
    dtype=np.int32,
)


def _pose(offset_y: int = 0) -> ShoulderCalibrationPose:
    points = [(int(x), int(y + offset_y)) for x, y in ANCHORS]
    return ShoulderCalibrationPose(
        left_points=(points[0], points[1], points[2]),
        right_points=(points[3], points[4], points[5]),
    )


def _frame(offset_y: int = 0, obscure: tuple[int, ...] = ()) -> np.ndarray:
    image = np.full((480, 640, 3), 45, dtype=np.uint8)
    rng = np.random.default_rng(12345)
    for index, (x, y) in enumerate(ANCHORS):
        patch = rng.integers(20, 235, size=(31, 31), dtype=np.uint8)
        patch = cv2.GaussianBlur(patch, (3, 3), 0)
        y_center = int(y + offset_y)
        x1, x2 = int(x - 15), int(x + 16)
        y1, y2 = y_center - 15, y_center + 16
        if index in obscure:
            image[y1:y2, x1:x2] = 45
        else:
            image[y1:y2, x1:x2, 0] = patch
            image[y1:y2, x1:x2, 1] = patch
            image[y1:y2, x1:x2, 2] = patch
    return image


def _tracker() -> ShoulderAnchorTracker:
    tracker = ShoulderAnchorTracker(
        template_size_px=25,
        search_radius_px=42,
        minimum_match_confidence=0.45,
        minimum_valid_anchors=4,
        maximum_group_motion_residual_px=16.0,
    )
    for _ in range(3):
        tracker.add_calibration_pose(_frame(), _pose())
    return tracker


def test_six_direct_anchors_follow_downward_shoulder_motion() -> None:
    tracker = _tracker()
    tracked = tracker.track(_frame(offset_y=8))
    assert tracked is not None
    assert tracked.valid_anchor_count == 6
    baseline = _pose().shoulders
    assert 6.0 <= tracked.left[1] - baseline.left[1] <= 10.0
    assert 6.0 <= tracked.right[1] - baseline.right[1] <= 10.0


def test_one_lost_anchor_does_not_break_shoulder_tracking() -> None:
    tracker = _tracker()
    tracked = tracker.track(_frame(offset_y=6, obscure=(1,)))
    assert tracked is not None
    assert 4 <= tracked.valid_anchor_count <= 5


def test_tracking_stops_instead_of_guessing_when_fewer_than_four_anchors_are_valid() -> None:
    tracker = _tracker()
    tracked = tracker.track(_frame(offset_y=6, obscure=(0, 1, 3, 4)))
    assert tracked is None
    assert tracker.valid_anchor_count < 4


def _shifted_pose(offset_x: int = 0, offset_y: int = 0) -> ShoulderCalibrationPose:
    points = [(int(x + offset_x), int(y + offset_y)) for x, y in ANCHORS]
    return ShoulderCalibrationPose(
        left_points=(points[0], points[1], points[2]),
        right_points=(points[3], points[4], points[5]),
    )


def _shifted_frame(offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
    image = np.full((480, 640, 3), 45, dtype=np.uint8)
    rng = np.random.default_rng(12345)
    for x, y in ANCHORS:
        patch = rng.integers(20, 235, size=(31, 31), dtype=np.uint8)
        patch = cv2.GaussianBlur(patch, (3, 3), 0)
        x_center = int(x + offset_x)
        y_center = int(y + offset_y)
        x1, x2 = x_center - 15, x_center + 16
        y1, y2 = y_center - 15, y_center + 16
        image[y1:y2, x1:x2, 0] = patch
        image[y1:y2, x1:x2, 1] = patch
        image[y1:y2, x1:x2, 2] = patch
    return image


def test_tracker_reacquires_from_a_calibrated_turn_outside_normal_search_radius() -> None:
    tracker = ShoulderAnchorTracker(
        template_size_px=25,
        search_radius_px=42,
        minimum_match_confidence=0.45,
        minimum_valid_anchors=4,
        maximum_group_motion_residual_px=16.0,
    )
    tracker.add_calibration_pose(_shifted_frame(0), _shifted_pose(0))
    tracker.add_calibration_pose(_shifted_frame(65), _shifted_pose(65))
    tracker.add_calibration_pose(_shifted_frame(-65), _shifted_pose(-65))

    # Monitoring is seeded from CENTER, so this RIGHT calibration pose is outside
    # the normal 42 px search window and requires the recovery path.
    tracked = tracker.track(_shifted_frame(-65))
    assert tracked is not None
    assert tracked.valid_anchor_count == 6
    assert abs(tracked.left[0] - _shifted_pose(-65).shoulders.left[0]) < 3.0


def test_tracker_recovers_after_a_fully_lost_frame() -> None:
    tracker = _tracker()
    blank = np.full((480, 640, 3), 45, dtype=np.uint8)
    assert tracker.track(blank) is None
    assert tracker.valid_anchor_count == 0

    recovered = tracker.track(_frame(offset_y=10))
    assert recovered is not None
    assert recovered.valid_anchor_count == 6
