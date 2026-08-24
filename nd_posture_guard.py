from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


class NdPostureGuard:
    def __init__(self) -> None:
        self._project_root = Path(__file__).resolve().parent
        self._src_path = self._project_root / "src"
        self._settings_path = self._project_root / "settings.yaml"

    def run(self) -> int:
        self._validate_required_files()
        self._prepare_package_import()
        self._ensure_mediapipe()

        from nd_posture_guard.application.mediapipe_posture_guard_application import (
            MediaPipePostureGuardApplication,
        )

        application = MediaPipePostureGuardApplication(self._settings_path)
        return application.run()

    def _ensure_mediapipe(self) -> None:
        try:
            importlib.import_module("mediapipe")
            return
        except ImportError:
            pass

        print("MediaPipe is required for posture landmark detection; installing it once...")
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "mediapipe>=0.10.35,<1",
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "Could not install MediaPipe automatically. Run: "
                f"{sys.executable} -m pip install 'mediapipe>=0.10.35,<1'"
            )
        importlib.invalidate_caches()
        importlib.import_module("mediapipe")

    def _prepare_package_import(self) -> None:
        src_path = str(self._src_path)
        loaded_module = sys.modules.get("nd_posture_guard")
        if loaded_module is not None and not hasattr(loaded_module, "__path__"):
            sys.modules.pop("nd_posture_guard", None)
        sys.path[:] = [entry for entry in sys.path if entry != src_path]
        sys.path.insert(0, src_path)

    def _validate_required_files(self) -> None:
        package_path = self._src_path / "nd_posture_guard"
        if not package_path.is_dir():
            raise FileNotFoundError(
                f"Project package directory was not found: {package_path}"
            )
        if not (package_path / "__init__.py").is_file():
            raise FileNotFoundError(
                f"Package marker was not found: {package_path / '__init__.py'}"
            )
        if not self._settings_path.is_file():
            raise FileNotFoundError(f"Settings file was not found: {self._settings_path}")


if __name__ == "__main__":
    raise SystemExit(NdPostureGuard().run())
