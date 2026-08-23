import numpy as np

from nd_posture_guard.training.example_training_session import ExampleTrainingSession
from nd_posture_guard.training.shoulder_roi_builder import ShoulderRoiBuilder


def test_single_training_sample_collects_six_points_and_frames() -> None:
    session = ExampleTrainingSession(ShoulderRoiBuilder(0.1, 0.55), 8)
    session.start(0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    session.begin_point_selection(frame)
    points = ((300, 220), (220, 240), (120, 270), (340, 225), (420, 245), (520, 275))
    for point in points:
        session.add_point(point, 640, 480)
    assert session.recording
    for index in range(8):
        finished = session.add_observation(
            np.full(16, index / 100.0, dtype=np.float32), frame
        )
    assert finished
    assert session.complete
    assert session.features.shape == (8, 16)
    assert len(session.frames) == 8
    assert session.label == 0


def test_session_supports_bad_label_without_direction_categories() -> None:
    session = ExampleTrainingSession(ShoulderRoiBuilder(0.1, 0.55), 8)
    session.start(1)
    assert session.label_name == "BAD"
    assert session.waiting_for_frame
