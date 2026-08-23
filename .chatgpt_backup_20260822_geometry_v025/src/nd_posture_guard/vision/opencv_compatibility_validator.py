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

        required_api_names = ("VideoCapture", "resize")
        missing = [name for name in required_api_names if not hasattr(cv2, name)]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"This OpenCV build is missing required APIs: {joined}."
            )
