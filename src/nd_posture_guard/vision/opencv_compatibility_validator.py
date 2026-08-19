from __future__ import annotations

import cv2


class OpenCvCompatibilityValidator:
    def validate(self) -> None:
        version_text = str(getattr(cv2, "__version__", "0"))
        try:
            major_version = int(version_text.split(".", maxsplit=1)[0])
        except ValueError as exc:
            raise RuntimeError(f"Cannot parse OpenCV version: {version_text}") from exc
        if major_version < 5:
            raise RuntimeError(f"OpenCV 5.x is required, but OpenCV {version_text} is loaded.")
        if not hasattr(cv2, "matchTemplate") or not hasattr(cv2, "VideoCapture"):
            raise RuntimeError("This OpenCV build does not provide the required camera/template-matching APIs.")
