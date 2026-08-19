# ND Posture Guard

Shoulder-only posture monitoring for Ubuntu using Python, PySide6 and OpenCV 5. The head and face are not used for posture decisions.

## Run in PyCharm

Use this interpreter:

```text
/home/donkarlo/phd-venv/bin/python
```

Run the root file:

```text
nd_posture_guard.py
```

## 18-point calibration

Press **Calibrate shoulders** while sitting correctly.

For each of the three poses, the image freezes and asks for six clicks in this exact order:

1. LEFT shoulder: point next to the neck.
2. LEFT shoulder: middle of the shoulder line.
3. LEFT shoulder: outer shoulder tip.
4. RIGHT shoulder: point next to the neck.
5. RIGHT shoulder: middle of the shoulder line.
6. RIGHT shoulder: outer shoulder tip.

Repeat the same six points for:

- CENTER: sitting straight and facing forward.
- LEFT: keep the back straight and rotate the upper body only a little to the left.
- RIGHT: keep the back straight and rotate the upper body only a little to the right.

Total: **18 calibration points**.

After calibration, the six clicked shoulder anchors are tracked **directly**. The tracker does not discover arbitrary feature points on the face, furniture, or background. Normal tracking stays local. If a fast movement makes the anchors leave that local window, the tracker automatically tries to reacquire the same six shoulder anchors from the calibrated CENTER/LEFT/RIGHT poses with a wider search, so a temporary 0/6 state is no longer permanent.

One missing anchor is tolerated. Monitoring requires at least **4 of 6 anchors**, with at least **2 valid anchors on each shoulder**. If that condition is not met, the program reports tracking loss and pauses posture warnings instead of inventing a shoulder position.

The shoulder-drop threshold can still be changed after calibration without recalibrating.

## Alert sound

The spoken warning was removed. A posture alert now plays **one long beep** from the bundled `assets/one_long_beep.wav`. No network, TTS service, or extra Python package is required. Use **Test one long beep** in the GUI to test it immediately. Ubuntu desktop notifications are not used.

## Important settings

```yaml
monitoring:
  shoulder_drop_trigger_percent: 7.0
  shoulder_width_trigger_percent: 12.0
  shoulder_tilt_trigger_degrees: 12.0
  required_bad_frames: 5
  alert_cooldown_seconds: 3.0
```

`shoulder_drop_trigger_percent` can be changed from the GUI after calibration.


## Direct-anchor tracker settings

```yaml
vision:
  shoulder_anchor_template_size_px: 25
  shoulder_anchor_search_radius_px: 42
  shoulder_anchor_minimum_match_confidence: 0.45
  shoulder_anchor_minimum_valid_anchors: 4
  shoulder_anchor_maximum_group_motion_residual_px: 16.0
```

The cyan overlay is drawn only along the left and right shoulder anchors separately; it is never drawn as a single line across the face.

## Calibration state safety

Calibration is a one-way three-stage flow: CENTER -> LEFT -> RIGHT -> COMPLETE. Duplicate/double-clicked calibration commands are ignored while a stage is collecting its six points. If a stage is invalid, calibration stops cleanly and must be restarted explicitly; rejected points cannot remain stuck and cause an endless calibration loop.
