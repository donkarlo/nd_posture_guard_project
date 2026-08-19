from __future__ import annotations

import time

from nd_posture_guard.alerts.alert_sound_player import AlertSoundPlayer


class AlertService:
    def __init__(
        self,
        cooldown_seconds: float,
        sound_player: AlertSoundPlayer,
    ) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._sound_player = sound_player
        self._last_alert_time = 0.0

    @property
    def sound_path(self) -> str:
        return str(self._sound_player.sound_path)

    def trigger(self) -> bool:
        now = time.monotonic()
        if now - self._last_alert_time < self._cooldown_seconds:
            return False

        self._last_alert_time = now
        return self._sound_player.play()

    def test_alert(self) -> bool:
        return self._sound_player.play()
