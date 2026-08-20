# ND Posture Guard — v0.24.0

ND Posture Guard is a personal Ubuntu webcam posture monitor. It learns from your own **GOOD** and **BAD** shoulder-posture recordings, stores the training dataset outside the application source tree, and produces one long warning beep when BAD posture remains above the configured beep threshold.


## Ubuntu dock/taskbar icon

v0.24 adds a dedicated **ND Posture Guard** icon. This also works when the app is started directly from a terminal with:

```bash
python nd_posture_guard.py
```

At startup the application sets its Qt window icon and stable Linux desktop identity (`nd-posture-guard`). It also idempotently installs/updates these user-local desktop integration files:

```text
~/.local/share/icons/hicolor/scalable/apps/nd-posture-guard.svg
~/.local/share/applications/nd-posture-guard.desktop
```

No root/sudo access is required. Ubuntu/GNOME can therefore distinguish the running posture monitor from generic Python/PySide applications in the dock. The desktop integration is cosmetic and never prevents the posture monitor from starting if the desktop files cannot be written.

## Important changes in v0.23

### 1. The arbitrary 62% default is removed

The previous `62%` default did not have a principled meaning. The classifier's natural GOOD/BAD decision boundary is now:

```text
50%
```

The main control is therefore named **Beep threshold**.

- bad-score `<= 50%` → GOOD
- bad-score `> 50%` → BAD
- the adjustable Beep threshold decides when BAD evidence is strong enough to produce the warning sound

The default Beep threshold is `50%`. You can raise it if the monitor is too sensitive.

Old v0.22 runtime files that contain exactly the old unschematized `62%` default are migrated once to `50%`. Other previously saved user values are preserved.

### 2. Threshold persistence is now direct and atomic

Changing the Beep threshold no longer depends on the camera worker thread. The GUI/controller writes the value directly to:

```text
/home/donkarlo/Dropbox/repo/data/nd_posture_guard_project/runtime/runtime_settings.yaml
```

The file is written atomically. The project `settings.yaml` is not rewritten, so installing a newer ZIP does not replace the user's runtime value.

When the window closes, the value currently visible in the spin box is committed again. This also covers the case where a number is typed and the application is closed immediately without pressing Enter or Tab.

### 3. BAD detection is more sensitive to actual shoulder movement

The earlier classifier relied too heavily on normalized texture/HOG-like features. That could recognize the same shirt very well while failing to treat a substantial downward shoulder movement as BAD.

v0.23 rebuilds runtime descriptors from the already stored raw frames and six shoulder points. It adds strong fixed-camera spatial features:

- a low-resolution vertical-edge map for each shoulder;
- a row-by-row shoulder edge profile;
- the existing HOG-like local appearance descriptor.

The spatial components receive substantially more weight than shirt texture. Since the shoulder ROIs stay fixed in camera coordinates, a shoulder moving downward or forward moves inside the feature map instead of being normalized away.

The classifier also learns a supervised GOOD→BAD posture direction from the user's examples. Dimensions that consistently separate GOOD from BAD receive more importance than irrelevant changing texture.

No shoulder point is tracked during normal monitoring, so long-term anchor drift is not used.

## Persistent data location

All user training data is stored here:

```text
/home/donkarlo/Dropbox/repo/data/nd_posture_guard_project
```

The application creates the directory automatically if it does not exist.

Typical layout:

```text
/home/donkarlo/Dropbox/repo/data/nd_posture_guard_project/
├── dataset_manifest.json
├── runtime/
│   └── runtime_settings.yaml
└── samples/
    ├── good/
    │   └── <sample-id>/
    │       ├── metadata.json
    │       ├── anchor_frame.jpg
    │       ├── clip.avi
    │       ├── frames.npz
    │       └── features.npy
    └── bad/
        └── <sample-id>/
            └── ...
```

Replacing the application source or ZIP does not replace this dataset.

## Existing v0.20–v0.22 training videos

You do **not** need to retrain just because v0.23 uses a new runtime feature schema. The durable data is the saved raw video frames plus the six normalized shoulder points. On startup v0.23 rebuilds the new runtime feature representation from those files.

Compatible examples are merged. Raw samples are never silently replaced.

## Adding training data

Use either:

```text
Add good training data
Add bad training data
```

Each sample asks for exactly six shoulder points:

1. Left shoulder — inner/neck-side visible point.
2. Left shoulder — middle.
3. Left shoulder — outer shoulder tip.
4. Right shoulder — inner/neck-side visible point.
5. Right shoulder — middle.
6. Right shoulder — outer shoulder tip.

Then hold the selected posture naturally for about three seconds while the application records frames.

There are no separate CENTER/LEFT/RIGHT categories. If the real neck/shoulder junction is hidden during rotation, select the innermost visible part of the shoulder. A shirt collar is not required.

## How to improve the personal model

Use GOOD examples for postures that must not beep. Use BAD examples for postures that should beep.

If a bad posture is missed, add another BAD example of that posture. If a normal posture is incorrectly rejected, add another GOOD example of that normal posture.

The dataset is incremental: new compatible samples are appended and the runtime classifier is rebuilt from the complete dataset.

## Review and delete training videos

Press:

```text
Review / delete training videos
```

Saved samples are displayed newest first with their GOOD/BAD label. The selected clip loops in the review window. Use:

```text
Delete selected training sample
```

to remove one incorrect or unwanted recording. The model is rebuilt immediately from the remaining data.

## Pause / Resume

Use:

```text
Pause monitoring
Resume monitoring
```

when leaving the computer. No warning beep is generated while monitoring is paused.

## Warning sound

There are no Ubuntu desktop notifications. The warning is one long beep. Playback tries available Ubuntu audio backends in order:

1. `paplay`
2. `pw-play`
3. `aplay`
4. `ffplay`

## Run

```bash
cd /home/donkarlo/Dropbox/repo/nd_posture_guard_project
python nd_posture_guard.py
```

The application supports the standard `opencv-python` 5 package and does not require `cv2.HOGDescriptor` or `opencv-contrib-python`.

## Test status

The v0.23 package is tested for:

- persistent threshold save/load and migration from the old 62% default;
- preservation of project `settings.yaml` while runtime settings change;
- fixed-camera spatial shoulder descriptors;
- GOOD/BAD classifier separation;
- reuse of old raw training recordings;
- incremental dataset save/list/delete;
- alert playback fallback behavior.
