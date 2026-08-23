from __future__ import annotations

from nd_posture_guard.ui.main_window import MainWindow


class IsolatedMainWindow(MainWindow):
    """Describe accepted training samples accurately while persistence is asynchronous."""

    def set_training_sample_saved(
        self, label_name: str, good: int, bad: int, model_ready: bool
    ) -> None:
        super().set_training_sample_saved(label_name, good, bad, model_ready)
        self._progress.setFormat(f"{label_name} sample accepted — saving in background")
        self._video.set_overlay(
            f"{label_name} SAMPLE ACCEPTED\n"
            f"Saving in background. Pending dataset: {good} GOOD / {bad} BAD."
        )
