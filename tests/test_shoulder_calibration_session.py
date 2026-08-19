import pytest

from nd_posture_guard.posture.shoulder_calibration_session import ShoulderCalibrationSession


def _pose_points(shift_x: int = 0, left_y: int = 250, right_y: int = 250):
    return [
        (180 + shift_x, left_y),
        (150 + shift_x, left_y + 2),
        (115 + shift_x, left_y + 4),
        (460 + shift_x, right_y),
        (490 + shift_x, right_y + 2),
        (525 + shift_x, right_y + 4),
    ]


def test_three_manual_poses_use_eighteen_points() -> None:
    session = ShoulderCalibrationSession()
    session.start()
    assert session.target_points == 18
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


def test_point_prompt_walks_from_left_neck_to_right_tip() -> None:
    session = ShoulderCalibrationSession()
    session.start()
    assert "LEFT shoulder 1/3" in session.next_point_message
    for index, point in enumerate(_pose_points()[:5], start=1):
        assert session.add_point(point, 640, 480) is None
        assert session.point_count == index
    assert "RIGHT shoulder 3/3" in session.next_point_message


def test_invalid_six_point_pose_resets_current_stage_instead_of_sticking() -> None:
    session = ShoulderCalibrationSession()
    session.start()
    invalid = [
        (115, 250), (150, 252), (180, 254),  # LEFT is clicked in the wrong order
        (460, 250), (490, 252), (525, 254),
    ]
    for point in invalid[:-1]:
        assert session.add_point(point, 640, 480) is None
    with pytest.raises(ValueError):
        session.add_point(invalid[-1], 640, 480)
    assert session.point_count == 0
    assert session.stage_name == "CENTER"
    assert "LEFT shoulder 1/3" in session.next_point_message
