import pytest

from nd_posture_guard.posture.shoulder_calibration_session import ShoulderCalibrationSession


def _pose_points(shift_x: int = 0, left_y: int = 250, right_y: int = 250):
    return [
        (190 + shift_x, left_y),
        (150 + shift_x, left_y + 2),
        (110 + shift_x, left_y + 4),
        (450 + shift_x, right_y),
        (490 + shift_x, right_y + 2),
        (535 + shift_x, right_y + 4),
    ]


def test_three_manual_poses_use_eighteen_points() -> None:
    session = ShoulderCalibrationSession()
    session.start()
    assert session.target_points == 18
    assert session.points_per_pose == 6
    for points in (_pose_points(), _pose_points(10, 248, 255), _pose_points(-8, 255, 247)):
        pose = None
        for point in points:
            pose = session.add_point(point, 640, 480)
        assert pose is not None
        if not session.complete:
            assert session.waiting_for_turn
            assert session.continue_after_turn()
    assert session.point_count == 18
    reference = session.build_reference()
    assert len(reference.profiles) == 3
    assert reference.shoulders.width > 300
    assert len(reference.shoulders.anchors) == 6


def test_point_prompt_allows_chin_shoulder_contact_when_turned() -> None:
    session = ShoulderCalibrationSession()
    session.start()
    assert "LEFT shoulder 1/3" in session.next_point_message
    assert "neck/shoulder junction" in session.next_point_message
    for point in _pose_points():
        session.add_point(point, 640, 480)
    assert session.waiting_for_turn
    assert session.continue_after_turn()
    assert "LEFT shoulder 1/3" in session.next_point_message
    assert "chin/shoulder contact" in session.next_point_message


def test_invalid_six_point_pose_resets_current_stage_instead_of_sticking() -> None:
    session = ShoulderCalibrationSession()
    session.start()
    valid = _pose_points()
    invalid = [
        (110, 250), (150, 252), (190, 254),
        *valid[3:],
    ]
    for point in invalid[:-1]:
        assert session.add_point(point, 640, 480) is None
    with pytest.raises(ValueError):
        session.add_point(invalid[-1], 640, 480)
    assert session.point_count == 0
    assert session.stage_name == "CENTER"
    assert "LEFT shoulder 1/3" in session.next_point_message
