from nd_posture_guard.training.shoulder_roi_builder import ShoulderRoiBuilder


def test_build_rois_excludes_center_and_stays_in_frame() -> None:
    points = ((300, 220), (220, 240), (120, 270), (340, 225), (420, 245), (520, 275))
    left, right = ShoulderRoiBuilder(0.10, 0.55).build(points, 640, 480)
    assert left[0] < left[2] < right[0] < right[2]
    assert 0 <= left[1] < left[3] <= 480
    assert 0 <= right[1] < right[3] <= 480
