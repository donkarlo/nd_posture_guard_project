# ND Posture Guard

Shoulder-only posture monitoring for Ubuntu using Python, PySide6, and OpenCV 5.
Head and face position are not used for posture decisions.

## Calibration

Calibration uses **18 clicks total**:

1. CENTER: 3 points on the left shoulder + 3 on the right shoulder.
2. Slight LEFT turn while keeping the back straight: 3 + 3 points.
3. Slight RIGHT turn while keeping the back straight: 3 + 3 points.

For each shoulder, mark the inner visible shoulder area, the middle, and the outer shoulder tip. When a small turn hides the neck/shoulder junction, the first inner point may be the visible chin/shoulder contact or the nearest visible inner shoulder area. A shirt collar is optional and is not required.

## Tracking

The six manually selected anchors are followed with pyramidal Lucas-Kanade optical flow. Forward/backward checking rejects drifting points. If tracking is lost, template matching tries to reacquire the anchors from the CENTER, LEFT, and RIGHT calibration views.

Monitoring requires at least four valid anchors, with at least two valid anchors on each shoulder.

## Pause / Resume

Use **Pause monitoring** before leaving the computer. Camera preview remains active, but posture evaluation and alert beeps stop. Press **Resume monitoring** when you return; the app discards stale optical-flow state and reacquires the calibrated shoulder anchors.

## Settings persistence

The `Shoulder-drop gate` value is written to `settings.yaml` immediately whenever it is changed in the GUI. The next run restores the last value automatically.

## Run from PyCharm

Use the interpreter:

```text
/home/donkarlo/phd-venv/bin/python
```

Run the root file:

```text
nd_posture_guard.py
```
