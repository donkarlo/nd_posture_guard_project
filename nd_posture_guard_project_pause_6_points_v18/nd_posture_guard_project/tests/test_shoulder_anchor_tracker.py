import cv2
import numpy as np

from nd_posture_guard.posture.shoulder_calibration_pose import ShoulderCalibrationPose
from nd_posture_guard.vision.shoulder_anchor_tracker import ShoulderAnchorTracker


ANCHORS = np.array(
    [
        [190, 245], [150, 249], [110, 253],
        [450, 245], [490, 249], [535, 253],
    ],
    dtype=np.int32,
)


def _pose(offset_x: int = 0, offset_y: int = 0) -> ShoulderCalibrationPose:
    points = [(int(x + offset_x), int(y + offset_y)) for x, y in ANCHORS]
    return ShoulderCalibrationPose(
        left_points=tuple(points[:3]),
        right_points=tuple(points[3:]),
    )


def _frame(offset_x: int = 0, offset_y: int = 0, obscure: tuple[int, ...] = ()) -> np.ndarray:
    image = np.full((480, 640, 3), 45, dtype=np.uint8)
    rng = np.random.default_rng(12345)
    for index, (x, y) in enumerate(ANCHORS):
        patch = rng.integers(20, 235, size=(31, 31), dtype=np.uint8)
        patch = cv2.GaussianBlur(patch, (3, 3), 0)
        x_center = int(x + offset_x)
        y_center = int(y + offset_y)
        x1, x2 = x_center - 15, x_center + 16
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
        search_radius_px=48,
        recovery_search_radius_px=190,
        minimum_match_confidence=0.42,
        minimum_valid_anchors=4,
        maximum_group_motion_residual_px=22.0,
        optical_flow_window_px=41,
        optical_flow_pyramid_levels=4,
        optical_flow_forward_backward_error_px=4.0,
    )
    tracker.add_calibration_pose(_frame(), _pose())
    tracker.add_calibration_pose(_frame(offset_x=55), _pose(offset_x=55))
    tracker.add_calibration_pose(_frame(offset_x=-55), _pose(offset_x=-55))
    return tracker


def test_six_anchors_follow_large_gradual_downward_slouch() -> None:
    tracker = _tracker()
    tracked = tracker.track(_frame())
    assert tracked is not None
    for offset_y in (15, 30, 50, 75):
        tracked = tracker.track(_frame(offset_y=offset_y))
        assert tracked is not None
        assert tracked.valid_anchor_count >= 4
    baseline = _pose().shoulders
    assert 68 <= tracked.left[1] - baseline.left[1] <= 82
    assert 68 <= tracked.right[1] - baseline.right[1] <= 82


def test_tracker_can_continue_when_one_anchor_per_shoulder_is_obscured() -> None:
    tracker = _tracker()
    assert tracker.track(_frame()) is not None
    tracked = tracker.track(_frame(offset_y=10, obscure=(1, 4)))
    assert tracked is not None
    assert tracked.valid_anchor_count >= 4


def test_tracker_recovers_after_a_fully_lost_frame() -> None:
    tracker = _tracker()
    assert tracker.track(_frame()) is not None
    blank = np.full((480, 640, 3), 45, dtype=np.uint8)
    assert tracker.track(blank) is None
    assert tracker.valid_anchor_count == 0

    recovered = tracker.track(_frame(offset_y=70))
    assert recovered is not None
    assert recovered.valid_anchor_count >= 4


def test_tracker_reacquires_a_calibrated_turn() -> None:
    tracker = _tracker()
    tracked = tracker.track(_frame(offset_x=-55))
    assert tracked is not None
    assert tracked.valid_anchor_count >= 4
    expected = _pose(offset_x=-55).shoulders
    assert abs(tracked.left[0] - expected.left[0]) < 5.0


def test_tracker_recovers_from_a_large_abrupt_vertical_move() -> None:
    tracker = _tracker()
    assert tracker.track(_frame()) is not None
    tracked = tracker.track(_frame(offset_y=95))
    assert tracked is not None
    assert tracked.valid_anchor_count >= 4
    assert tracked.left[1] > _pose().shoulders.left[1] + 85


def test_prepare_for_reacquire_keeps_calibration_and_recovers() -> None:
    tracker = _tracker()
    assert tracker.track(_frame(offset_y=25)) is not None
    tracker.prepare_for_reacquire()
    assert tracker.ready
    assert tracker.valid_anchor_count == 0
    recovered = tracker.track(_frame(offset_y=40))
    assert recovered is not None
    assert recovered.valid_anchor_count >= 4
