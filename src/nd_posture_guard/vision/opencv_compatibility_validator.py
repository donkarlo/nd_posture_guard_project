from __future__ import annotations

import cv2


class OpenCvCompatibilityValidator:
    """Fail early when the installed OpenCV 5 build lacks APIs used by geometry tracking."""

    def validate(self) -> None:
        """Validate OpenCV major version and the exact runtime APIs the monitor needs."""
        version_text = str(getattr(cv2, "__version__", "0"))
        try:
            major_version = int(version_text.split(".", maxsplit=1)[0])
        except ValueError as exc:
            raise RuntimeError(f"Cannot parse OpenCV version: {version_text}") from exc
        if major_version < 5:
            raise RuntimeError(f"OpenCV 5.x is required, but OpenCV {version_text} is loaded.")

        required_api_names = (
            "VideoCapture",
            "VideoWriter",
            "resize",
            "cvtColor",
            "GaussianBlur",
            "calcOpticalFlowPyrLK",
            "matchTemplate",
            "minMaxLoc",
        )
        missing = [name for name in required_api_names if not hasattr(cv2, name)]
        if missing:
            raise RuntimeError(
                "This OpenCV build is missing required posture-tracking APIs: "
                + ", ".join(missing)
            )
