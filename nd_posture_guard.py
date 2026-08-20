from __future__ import annotations

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

        from nd_posture_guard.application.posture_guard_application import (
            PostureGuardApplication,
        )

        application = PostureGuardApplication(self._settings_path)
        return application.run()

    def _prepare_package_import(self) -> None:
        src_path = str(self._src_path)

        # PyCharm can load this launcher under the name ``nd_posture_guard``.
        # If that happens, it shadows the real package in src/nd_posture_guard.
        loaded_module = sys.modules.get("nd_posture_guard")
        if loaded_module is not None and not hasattr(loaded_module, "__path__"):
            sys.modules.pop("nd_posture_guard", None)

        # Always give the src layout precedence over the project root.
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
            raise FileNotFoundError(
                f"Settings file was not found: {self._settings_path}"
            )


if __name__ == "__main__":
    raise SystemExit(NdPostureGuard().run())
