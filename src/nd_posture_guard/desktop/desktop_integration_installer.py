from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


class DesktopIntegrationInstaller:
    """Install/update the user's Linux desktop metadata for this application."""

    DESKTOP_ID = "nd-posture-guard"
    DISPLAY_NAME = "ND Posture Guard"
    ICON_NAME = "nd-posture-guard"
    STARTUP_WM_CLASS = "nd-posture-guard"

    def __init__(self, project_root: Path, home: Path | None = None) -> None:
        self._project_root = project_root.resolve()
        self._home = (home or Path.home()).resolve()

    @property
    def project_icon_path(self) -> Path:
        return self._project_root / "assets" / "nd_posture_guard.svg"

    @property
    def installed_icon_path(self) -> Path:
        return (
            self._home
            / ".local"
            / "share"
            / "icons"
            / "hicolor"
            / "scalable"
            / "apps"
            / f"{self.ICON_NAME}.svg"
        )

    @property
    def desktop_file_path(self) -> Path:
        return (
            self._home
            / ".local"
            / "share"
            / "applications"
            / f"{self.DESKTOP_ID}.desktop"
        )

    def install(self) -> None:
        """Install icon and desktop entry idempotently; never block app startup on failure."""
        if not sys.platform.startswith("linux"):
            return
        if not self.project_icon_path.is_file():
            return

        try:
            self.installed_icon_path.parent.mkdir(parents=True, exist_ok=True)
            self.desktop_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.project_icon_path, self.installed_icon_path)
            self.desktop_file_path.write_text(self._desktop_file_text(), encoding="utf-8")
            self.desktop_file_path.chmod(0o755)
        except OSError:
            # Desktop integration is cosmetic. Camera monitoring must still start
            # if the home directory is read-only or desktop files are unavailable.
            return

    def _desktop_file_text(self) -> str:
        launcher = self._project_root / "nd_posture_guard.py"
        working_directory = self._project_root
        python_executable = Path(sys.executable).resolve()
        return "\n".join(
            (
                "[Desktop Entry]",
                "Type=Application",
                f"Name={self.DISPLAY_NAME}",
                "Comment=Personal shoulder-posture monitor",
                f"Exec={self._quote_exec(python_executable)} {self._quote_exec(launcher)}",
                f"Path={working_directory}",
                f"Icon={self.ICON_NAME}",
                "Terminal=false",
                "Categories=Utility;Health;",
                f"StartupWMClass={self.STARTUP_WM_CLASS}",
                "StartupNotify=true",
                "",
            )
        )

    @staticmethod
    def _quote_exec(path: Path) -> str:
        value = str(path).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{value}"'
