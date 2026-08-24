from __future__ import annotations

from PySide6.QtWidgets import QLabel

from nd_posture_guard.ui.isolated_main_window import IsolatedMainWindow


class AutoLandmarkMainWindow(IsolatedMainWindow):
    """UI wording for automatic per-frame landmark detection."""

    def __init__(self, title: str, bad_score_threshold: float, data_root: str) -> None:
        super().__init__(title, bad_score_threshold, data_root)
        self._progress.setFormat("Ready for automatic landmark capture")
        for label in self.findChildren(QLabel):
            if label.text().startswith(
                "For every GOOD or BAD example draw exactly three things"
            ):
                label.setText(
                    "Training is automatic: click Add GOOD or Add BAD and hold the intended posture. "
                    "The program freshly detects both eye centers, the chin and both shoulder lines "
                    "on every valid frame. No landmark is propagated from the previous frame; "
                    "a missed detection is ignored instead of reused."
                )
                break

    def set_training_started(self, label_name: str) -> None:
        self._training_active_ui = True
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setFormat(f"{label_name}: starting automatic landmark capture")
        self._state.setText(f"ADDING {label_name} — AUTO LANDMARK CAPTURE")
        self._state.setStyleSheet("font-weight: bold; color: #c07b00;")
        self._pause_button.setEnabled(False)
        self._good_button.setEnabled(False)
        self._bad_button.setEnabled(False)
        self._cancel_training_button.setEnabled(True)
        self._review_button.setEnabled(False)
        self._video.set_selection_enabled(False)
        self._video.clear_selection_points()
        self._video.set_overlay(
            f"{label_name} AUTO TRAINING\nHold the posture while landmarks are sampled."
        )

    def set_training_recording_progress(
        self,
        current: int,
        target: int,
        message: str,
    ) -> None:
        self._progress.setRange(0, target)
        self._progress.setValue(current)
        self._progress.setFormat(
            f"Automatic landmark capture: {current}/{target} valid frames"
        )
        self._video.set_selection_enabled(False)
        self._video.set_overlay(message)

    def set_training_points_required(
        self,
        message: str,
        selected: int,
        target: int,
    ) -> None:
        self._video.set_selection_enabled(False)
        self._video.set_overlay(message)
