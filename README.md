# ND Posture Guard — v0.25.0

ND Posture Guard is a personal Ubuntu webcam posture monitor. Version 0.25 replaces the old shoulder-texture model with a **seven-point geometry model** learned directly from your own GOOD and BAD examples.
** Good **
<img src="good.png" alt="Description" width="800">
** Bad **
<img src="bad.png" alt="Description" width="800">

## What the model uses now

Every new training sample is defined by exactly seven points, in this order:

1. Center of the eye on the **left side of the displayed image**.
2. Center of the eye on the **right side of the displayed image**.
3. Lowest visible point of the chin.
4. Left side of image: shoulder/neck junction.
5. Left side of image: outer shoulder endpoint.
6. Right side of image: shoulder/neck junction.
7. Right side of image: outer shoulder endpoint.

The first three points form the **face triangle**. Points 4–5 form the **left shoulder line** and points 6–7 form the **right shoulder line**.

The wording “left/right side of the image” is deliberate because the camera preview is mirrored. It avoids confusing anatomical left/right with screen left/right.

## Monitoring

During normal monitoring the same seven landmarks are tracked automatically with OpenCV optical flow plus periodic template re-anchoring learned from your own clicked samples.

The camera view displays the geometry **the program is actually using for its current decision**:

- the current eye-eye-chin triangle;
- the current left-image shoulder line;
- the current right-image shoulder line;
- the seven tracked landmark points;
- tracking confidence;
- GOOD/BAD state, BAD score and classification confidence.

The geometry overlay is color-coded: green for a confidently GOOD state, red for BAD, and amber when tracking confidence is low. This is useful for checking whether an unexpected classification comes from the classifier or from an inaccurate landmark estimate.

If tracking confidence becomes too low, the state becomes **TRACKING UNCERTAIN** and no warning beep is emitted from uncertain geometry.

## How GOOD/BAD classification works

The classifier does not use shirt texture as its primary signal. The seven coordinates are converted to a compact **28-dimensional geometric descriptor** containing normalized landmark positions, face proportions, chin displacement, shoulder vectors, shoulder tilt, and face-to-shoulder distances.

Geometry is translated and scaled relative to the shoulders before classification, so ordinary whole-body movement in the image has less influence than actual changes in head/neck/shoulder posture.

The classifier is personal and example-based. It uses your own GOOD and BAD samples in two complementary ways:

1. **Nearest-example distance:** the current 28-D vector is robustly scaled and compared with the nearest few GOOD and BAD examples. If it is closer to BAD examples, the distance-based BAD score rises.
2. **Learned GOOD→BAD posture axis:** the median GOOD geometry and median BAD geometry define a supervised direction in feature space. Dimensions with large within-class variance are down-weighted. The current posture is projected onto this axis and converted to a probability-like BAD score with a logistic function.

The two scores are combined as:

```text
raw BAD score = 0.40 × distance score + 0.60 × projection score
```

when the supervised projection is available. If it cannot be estimated reliably, only the distance score is used.

To suppress single-frame jitter, the displayed BAD score is the median of the last five raw scores. A posture is displayed as BAD when this smoothed score is above `0.5`. The **Beep threshold** is separate: it controls when warning audio is allowed to start, and the threshold must remain exceeded for the configured number of consecutive frames.

Classification confidence is based on distance from the neutral boundary:

```text
confidence = 2 × |BAD score − 0.5|
```

clipped to the interval `[0, 1]`.

For the exact equations and runtime pipeline, see [`docs/posture_guard_math.tex`](docs/posture_guard_math.tex).

## Existing six-point recordings

Existing recordings are **not deleted or overwritten**.

Older six-shoulder-point samples remain visible in **Review / delete training videos** and can still be watched or manually deleted. They are marked as legacy and are not mixed into the new seven-point model because they do not contain eye/chin landmarks.

Therefore v0.25 needs at least one new seven-point GOOD sample and one new seven-point BAD sample before the new model becomes ready.

## Review/delete fix

Deleting a training sample now removes it from the review list immediately. The worker then deletes the persistent sample and refreshes the list in place.

A background refresh never opens the review window. Closing the review window therefore cannot be followed by the old bug where a late delete/list callback unexpectedly opens it again.

## Long-running performance changes

The previous runtime repeatedly built large HOG/edge descriptors from shoulder image regions. Rebuilding the profile could also reopen and process every stored training frame.

v0.25 removes those expensive paths from active monitoring:

- runtime posture feature: 28 geometry values instead of a large image descriptor;
- optical flow tracks only seven points;
- template matching is local and periodic rather than full-frame every cycle;
- stored videos are no longer replayed to rebuild the classifier;
- the worker no longer makes an extra full camera-frame copy for each UI signal;
- UI camera-frame delivery is capped at **8 FPS during both monitoring and training**, preventing the Qt event queue from being flooded while point-selection/progress signals remain immediate.

Raw clips are still saved for review and future migration, but they are not continuously reprocessed.

## Persistent data

All user data stays outside the application source tree:

```text
/home/donkarlo/Dropbox/repo/data/nd_posture_guard_project
```

New samples are appended; old samples are never silently replaced.

## Pause / Resume

Use **Pause monitoring** when leaving the computer. Tracking geometry may remain visible, but no warning beep is generated while paused.

## Beep threshold persistence

The threshold remains saved atomically under:

```text
/home/donkarlo/Dropbox/repo/data/nd_posture_guard_project/runtime/runtime_settings.yaml
```

The project `settings.yaml` is not rewritten by the threshold control.

## Ubuntu dock/taskbar icon

The application still installs its icon and `.desktop` identity in the user-local Ubuntu locations at startup. v0.25 replaces the previous ambiguous icon with an upright human/posture symbol containing the face triangle, spine and shoulder guides.

## Run

```bash
cd /home/donkarlo/Dropbox/repo/nd_posture_guard_project
/home/donkarlo/phd-venv/bin/python nd_posture_guard.py
```

## Main implementation components

- `PostureGeometry`: semantic seven-landmark value object.
- `PostureGeometryFeatureExtractor`: compact normalized posture descriptor.
- `PostureGeometryTracker`: optical-flow + learned-template tracking.
- `ExamplePostureClassifier`: personal GOOD/BAD geometry classifier.
- `TrainingDatasetRepository`: append-only storage, legacy preservation and fast model rebuild.
- `ResponsiveMonitoringWorker`: keeps camera-frame delivery bounded during training so Qt does not accumulate an unbounded frame queue.
