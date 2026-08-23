import numpy as np

from nd_posture_guard.training.example_training_session import ExampleTrainingSession


def _points():
    return (
        (270, 150), (330, 150), (300, 220),
        (245, 280), (150, 300), (355, 280), (450, 300),
    )


def test_single_training_sample_collects_seven_geometry_points_and_frames() -> None:
    session = ExampleTrainingSession(8)
    session.start(0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    session.begin_point_selection(frame)
    for index, point in enumerate(_points()):
        assert session.add_point(point, 640, 480) is (index == 6)
    assert session.recording
    assert session.geometry.is_plausible(640, 480)
    for index in range(8):
        finished = session.add_frame(frame)
        assert finished is (index == 7)
    assert session.complete
    assert len(session.frames) == 8
    assert session.label == 0


def test_session_supports_bad_label_and_explicit_landmark_messages() -> None:
    session = ExampleTrainingSession(8)
    session.start(1)
    assert session.label_name == "BAD"
    assert session.waiting_for_frame
    assert "eye on the LEFT side of the image" in session.next_point_message
