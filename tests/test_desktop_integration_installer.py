from __future__ import annotations

from pathlib import Path

from nd_posture_guard.desktop.desktop_integration_installer import DesktopIntegrationInstaller


def test_installs_desktop_entry_and_icon(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    assets = project_root / "assets"
    assets.mkdir(parents=True)
    (assets / "nd_posture_guard.svg").write_text("<svg></svg>", encoding="utf-8")
    (project_root / "nd_posture_guard.py").write_text("print('ok')\n", encoding="utf-8")

    home = tmp_path / "home"
    installer = DesktopIntegrationInstaller(project_root=project_root, home=home)
    installer.install()

    assert installer.installed_icon_path.read_text(encoding="utf-8") == "<svg></svg>"
    desktop_text = installer.desktop_file_path.read_text(encoding="utf-8")
    assert "Name=ND Posture Guard" in desktop_text
    assert "Icon=nd-posture-guard" in desktop_text
    assert "StartupWMClass=nd-posture-guard" in desktop_text
    assert str(project_root / "nd_posture_guard.py") in desktop_text
