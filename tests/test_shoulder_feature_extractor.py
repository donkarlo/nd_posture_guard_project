import numpy as np

from nd_posture_guard.vision.shoulder_feature_extractor import ShoulderFeatureExtractor


def test_feature_extractor_works_without_opencv_hog() -> None:
    extractor = ShoulderFeatureExtractor()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    y = np.arange(100, dtype=np.uint8)[:, None]
    x = np.arange(120, dtype=np.uint8)[None, :]
    frame[80:180, 20:140, 1] = (x + y) % 255
    frame[80:180, 180:300, 2] = (2 * x + y) % 255

    feature = extractor.extract(frame, (20, 80, 140, 180), (180, 80, 300, 180))

    assert feature.shape == (extractor.feature_length,)
    assert np.isfinite(feature).all()
    assert 0.99 <= float(np.linalg.norm(feature)) <= 1.01


def test_spatial_descriptor_changes_when_shoulder_edge_moves_down() -> None:
    import cv2

    extractor = ShoulderFeatureExtractor()
    good = np.zeros((240, 320, 3), dtype=np.uint8)
    bad = np.zeros_like(good)
    cv2.line(good, (20, 100), (140, 90), (255, 255, 255), 5)
    cv2.line(good, (180, 90), (300, 100), (255, 255, 255), 5)
    cv2.line(bad, (20, 145), (140, 135), (255, 255, 255), 5)
    cv2.line(bad, (180, 135), (300, 145), (255, 255, 255), 5)

    rois = ((10, 60, 150, 190), (170, 60, 310, 190))
    good_feature = extractor.extract(good, *rois)
    bad_feature = extractor.extract(bad, *rois)
    distance = float(np.linalg.norm(good_feature - bad_feature))
    assert distance > 0.25
