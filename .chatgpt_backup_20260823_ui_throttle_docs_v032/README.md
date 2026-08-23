# ND Posture Guard — v0.25.0

ND Posture Guard is a personal Ubuntu webcam posture monitor. Version 0.25 replaces the old shoulder-texture model with a **seven-point geometry model** learned directly from your own GOOD and BAD examples.

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

The camera view displays:

- the best current face triangle;
- the best current left shoulder line;
- the best current right shoulder line;
- tracking confidence;
- GOOD/BAD score.

If tracking confidence becomes too low, the state becomes **TRACKING UNCERTAIN** and no warning beep is emitted from uncertain geometry.

## GOOD/BAD classification

The classifier no longer uses shirt texture as its primary signal. The seven coordinates are converted to a compact 28-dimensional geometric descriptor containing normalized landmark positions, face proportions, chin displacement, shoulder vectors, shoulder tilt, and face-to-shoulder distances.

Geometry is translated and scaled relative to the shoulders before classification, so ordinary whole-body movement in the image has less influence than actual changes in head/neck/shoulder posture.

The personal classifier combines nearest GOOD/BAD examples with a supervised GOOD→BAD direction learned from the examples you provide. The natural classification boundary remains 50%; **Beep threshold** controls only when warning audio starts.

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
- UI frame delivery is capped at 8 FPS to prevent Qt event-queue buildup during long sessions.

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
