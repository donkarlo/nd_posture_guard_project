from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread

from nd_posture_guard.application.stable_monitoring_worker import StableMonitoringWorker


class IsolatedMonitoringWorker(StableMonitoringWorker):
    """Keep persistence completely outside the camera/monitoring thread."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._save_jobs: Queue[tuple] = Queue()
        self._save_results: Queue[tuple[int, str, str | None]] = Queue()
        self._persistence_stop = Event()
        self._persistence_thread: Thread | None = None
        self._pending_good = 0
        self._pending_bad = 0

    def run(self) -> None:
        self._persistence_stop.clear()
        self._persistence_thread = Thread(
            target=self._persistence_loop,
            name="posture-persistence",
            daemon=True,
        )
        self._persistence_thread.start()
        try:
            super().run()
        finally:
            self._persistence_stop.set()

    def _process_commands(self) -> None:
        self._apply_save_results()
        super()._process_commands()

    def _finish_training_sample(self) -> None:
        """Release camera/UI immediately, then persist the sample on a daemon worker."""
        anchor = self._training_session.anchor_frame
        geometry = self._training_session.geometry
        label = self._training_session.label
        label_name = self._training_session.label_name
        points = self._training_session.current_points
        frames = self._training_session.frames
        try:
            feature = self._feature_extractor.extract(
                geometry, anchor.shape[1], anchor.shape[0]
            )
        except Exception as exc:
            self._abort_training(f"Could not prepare training sample: {exc}")
            return

        self._training_session.clear()
        self._training_message = ""
        self._monitoring_paused = self._paused_before_training
        self._geometry_tracker.reset()
        if label == 0:
            self._pending_good += 1
        else:
            self._pending_bad += 1

        predicted_good = self._known_good + self._pending_good
        predicted_bad = self._known_bad + self._pending_bad
        self.monitoring_paused_changed.emit(self._monitoring_paused)
        self.training_sample_saved.emit(
            label_name,
            predicted_good,
            predicted_bad,
            self._classifier.ready,
        )
        self.status_changed.emit(
            f"{label_name} sample accepted. Saving in background; camera monitoring remains live."
        )
        self._save_jobs.put((label, label_name, points, anchor, frames, feature))

    def _persistence_loop(self) -> None:
        while not self._persistence_stop.is_set() or not self._save_jobs.empty():
            try:
                label, label_name, points, anchor, frames, feature = self._save_jobs.get(
                    timeout=0.20
                )
            except Empty:
                continue
            error: str | None = None
            try:
                self._dataset_repository.save_sample(
                    label=label,
                    points=points,
                    anchor_frame=anchor,
                    frames=frames,
                    feature=feature,
                )
            except Exception as exc:
                error = str(exc)
            self._save_results.put((label, label_name, error))

    def _apply_save_results(self) -> None:
        changed = False
        while True:
            try:
                label, label_name, error = self._save_results.get_nowait()
            except Empty:
                break

            if label == 0:
                self._pending_good = max(0, self._pending_good - 1)
            else:
                self._pending_bad = max(0, self._pending_bad - 1)

            if error is not None:
                self.status_changed.emit(
                    f"{label_name} background save failed: {error}. Camera was not blocked."
                )
                changed = True
                continue

            if label == 0:
                self._known_good += 1
            else:
                self._known_bad += 1
            changed = True
            self.status_changed.emit(
                f"{label_name} sample persisted. Dataset: "
                f"{self._known_good} GOOD / {self._known_bad} BAD."
            )

        if not changed:
            return

        self.dataset_stats_changed.emit(
            self._known_good,
            self._known_bad,
            self._classifier.ready,
        )

        if self._pending_good == 0 and self._pending_bad == 0:
            self._reload_profile()
