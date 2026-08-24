from __future__ import annotations

from threading import Condition, Thread

from nd_posture_guard.alerts.alert_sound_player import AlertSoundPlayer


class AlertService:
    """Repeat the warning sound while a persistent BAD state remains active."""

    def __init__(
        self,
        cooldown_seconds: float,
        sound_player: AlertSoundPlayer,
    ) -> None:
        self._cooldown_seconds = max(0.25, float(cooldown_seconds))
        self._sound_player = sound_player
        self._condition = Condition()
        self._active = False
        self._stopping = False
        self._thread = Thread(
            target=self._alert_loop,
            name="posture-alert-repeat",
            daemon=True,
        )
        self._thread.start()

    @property
    def sound_path(self) -> str:
        return str(self._sound_player.sound_path)

    def set_active(self, active: bool) -> None:
        """Start or stop repeated alerts without depending on worker signal timing."""
        requested = bool(active)
        with self._condition:
            if self._stopping or requested == self._active:
                return
            self._active = requested
            self._condition.notify_all()

    def trigger(self) -> bool:
        """Play one immediate alert; retained for explicit/manual callers."""
        return self._sound_player.play()

    def test_alert(self) -> bool:
        return self._sound_player.play()

    def stop(self) -> None:
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            self._active = False
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _alert_loop(self) -> None:
        while True:
            with self._condition:
                while not self._active and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return

            self._sound_player.play()

            with self._condition:
                if self._stopping:
                    return
                if not self._active:
                    continue
                self._condition.wait(timeout=self._cooldown_seconds)
