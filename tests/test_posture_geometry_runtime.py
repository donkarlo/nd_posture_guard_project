import cv2
import numpy as np

from nd_posture_guard.model.posture_geometry import PostureGeometry
from nd_posture_guard.training.posture_training_profile import PostureTrainingProfile
from nd_posture_guard.vision.posture_geometry_feature_extractor import PostureGeometryFeatureExtractor
from nd_posture_guard.vision.posture_geometry_tracker import PostureGeometryTracker


def _geometry(dx=0, dy=0):
    return PostureGeometry.from_points((
        (270+dx,150+dy),(330+dx,150+dy),(300+dx,220+dy),
        (245+dx,280+dy),(150+dx,300+dy),(355+dx,280+dy),(450+dx,300+dy),
    ))


def _frame():
    rng = np.random.default_rng(7)
    gray = rng.integers(0,256,(480,640),dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _templates(frame, points):
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(3,3),0)
    result=[]
    for x,y in points:
        x,y=round(x),round(y)
        result.append((gray[y-14:y+15,x-14:x+15].copy(),))
    return tuple(result)


def test_feature_is_translation_invariant() -> None:
    extractor=PostureGeometryFeatureExtractor()
    a=extractor.extract(_geometry(),640,480)
    b=extractor.extract(_geometry(20,11),640,480)
    assert np.max(np.abs(a-b)) < 1e-5


def test_tracker_follows_translated_frame() -> None:
    frame=_frame(); geom=_geometry(); points=geom.points
    profile=PostureTrainingProfile(
        reference_points_normalized=tuple((x/640,y/480) for x,y in points),
        point_templates=_templates(frame,points),
        features=np.zeros((2,28),dtype=np.float32),
        labels=np.asarray([0,1],dtype=np.int8),
    )
    tracker=PostureGeometryTracker(reanchor_every_frames=4)
    tracker.set_profile(profile)
    initial=tracker.track(frame)
    assert initial is not None and initial.tracking_confidence > 0.8
    shifted=cv2.warpAffine(frame,np.float32([[1,0,6],[0,1,4]]),(640,480),borderMode=cv2.BORDER_REFLECT)
    tracked=tracker.track(shifted)
    expected=np.asarray(points)+np.asarray([6.0,4.0])
    observed=np.asarray(tracked.points)
    assert np.median(np.linalg.norm(observed-expected,axis=1)) < 2.0
