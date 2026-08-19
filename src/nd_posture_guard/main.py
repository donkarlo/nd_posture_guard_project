from __future__ import annotations

import argparse
from pathlib import Path

from nd_posture_guard.application.posture_guard_application import PostureGuardApplication


class Main:
    @staticmethod
    def run() -> int:
        parser = argparse.ArgumentParser(description="Webcam posture monitor")
        parser.add_argument(
            "--settings",
            type=Path,
            default=Path("settings.yaml"),
            help="Path to settings.yaml",
        )
        args = parser.parse_args()
        application = PostureGuardApplication(args.settings.resolve())
        return application.run()
