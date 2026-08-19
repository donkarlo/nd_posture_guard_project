from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.posture.shoulder_calibration_pose import ShoulderCalibrationPose
from nd_posture_guard.vision.tracked_shoulders import TrackedShoulders


class ShoulderAnchorTracker:
    def __init__(
        self,
        template_size_px: int,
        search_radius_px: int,
        minimum_match_confidence: float,
        minimum_valid_anchors: int,
        maximum_group_motion_residual_px: float,
    ) -> None:
        size = max(15, int(template_size_px))
        self._template_size = size if size % 2 == 1 else size + 1
        self._search_radius = max(12, int(search_radius_px))
        self._recovery_search_radius = max(96, self._search_radius * 3)
        self._minimum_match_confidence = max(0.05, min(float(minimum_match_confidence), 0.99))
        self._recovery_match_confidence = max(0.30, self._minimum_match_confidence * 0.78)
        self._minimum_valid_anchors = max(4, min(int(minimum_valid_anchors), 6))
        self._maximum_group_motion_residual = max(3.0, float(maximum_group_motion_residual_px))
        self._calibration_poses: list[ShoulderCalibrationPose] = []
        self._calibration_anchor_sets: list[NDArray[np.float32]] = []
        self._templates: list[list[NDArray[np.uint8]]] = [[] for _ in range(6)]
        self._anchors: NDArray[np.float32] | None = None
        self._confidence = 0.0
        self._valid_anchor_count = 0
        self._lost_frames = 0

    @property
    def ready(self) -> bool:
        return (
            len(self._calibration_poses) >= 3
            and self._anchors is not None
            and all(len(anchor_templates) >= 3 for anchor_templates in self._templates)
        )

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def valid_anchor_count(self) -> int:
        return self._valid_anchor_count

    def clear(self) -> None:
        self._calibration_poses.clear()
        self._calibration_anchor_sets.clear()
        self._templates = [[] for _ in range(6)]
        self._anchors = None
        self._confidence = 0.0
        self._valid_anchor_count = 0
        self._lost_frames = 0

    def add_calibration_pose(
        self,
        frame: NDArray[np.uint8],
        pose: ShoulderCalibrationPose,
    ) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pose_templates: list[NDArray[np.uint8]] = []
        for index, point in enumerate(pose.all_points):
            patch = self._extract_template(gray, point)
            if patch is None:
                raise ValueError(
                    f"Shoulder anchor {index + 1} is too close to the image edge. "
                    "Move slightly farther from the camera and calibrate again."
                )
            if float(np.std(patch)) < 3.0:
                raise ValueError(
                    f"Shoulder anchor {index + 1} has too little visible texture/contrast. "
                    "Place the point on the visible shoulder/clothing boundary and calibrate again."
                )
            pose_templates.append(patch)

        anchor_set = np.asarray(pose.all_points, dtype=np.float32)
        self._calibration_poses.append(pose)
        self._calibration_anchor_sets.append(anchor_set)
        for index, patch in enumerate(pose_templates):
            self._templates[index].append(patch)

        # While calibration is being collected, keep the most recent pose. Once all
        # three poses exist, seed normal monitoring from CENTER. If the user is still
        # turned LEFT/RIGHT, the recovery pass can immediately reacquire that pose.
        if len(self._calibration_anchor_sets) >= 3:
            self._anchors = self._calibration_anchor_sets[0].copy()
        else:
            self._anchors = anchor_set.copy()

        self._confidence = 1.0 if self.ready else 0.0
        self._valid_anchor_count = 6 if self.ready else 0
        self._lost_frames = 0

    def track(self, frame: NDArray[np.uint8]) -> TrackedShoulders | None:
        if not self.ready:
            self._confidence = 0.0
            self._valid_anchor_count = 0
            return None
        assert self._anchors is not None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        normal = self._track_from_seed(
            gray,
            self._anchors,
            self._search_radius,
            self._minimum_match_confidence,
        )
        if normal is not None:
            self._lost_frames = 0
            return self._accept(normal)

        # A fast head/body movement can move all six shoulder patches outside the
        # normal local search window. Previously that left the tracker permanently at
        # 0/6. Recovery searches from every calibrated CENTER/LEFT/RIGHT anchor set
        # using a wider window, then accepts only a geometrically plausible shoulder
        # configuration.
        recovered = self._recover(gray)
        if recovered is not None:
            self._lost_frames = 0
            return self._accept(recovered)

        self._lost_frames += 1
        self._confidence = 0.0
        self._valid_anchor_count = 0
        return None

    def _recover(
        self,
        gray: NDArray[np.uint8],
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]] | None:
        seeds: list[NDArray[np.float32]] = []
        if self._anchors is not None:
            seeds.append(self._anchors)
        seeds.extend(self._calibration_anchor_sets)

        best: tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]] | None = None
        best_quality = -1.0
        for seed in seeds:
            candidate = self._track_from_seed(
                gray,
                seed,
                self._recovery_search_radius,
                self._recovery_match_confidence,
            )
            if candidate is None:
                continue
            anchors, valid, scores = candidate
            accepted_scores = scores[valid]
            quality = float(np.mean(accepted_scores)) * (float(np.count_nonzero(valid)) / 6.0)
            if quality > best_quality:
                best_quality = quality
                best = (anchors, valid, scores)
        return best

    def _track_from_seed(
        self,
        gray: NDArray[np.uint8],
        seed_anchors: NDArray[np.float32],
        search_radius: int,
        confidence_threshold: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]] | None:
        candidates = seed_anchors.copy()
        scores = np.zeros(6, dtype=np.float32)
        raw_valid = np.zeros(6, dtype=bool)

        for index in range(6):
            matched = self._match_anchor(
                gray,
                seed_anchors[index],
                self._templates[index],
                search_radius,
            )
            if matched is None:
                continue
            point, score = matched
            candidates[index] = point
            scores[index] = score
            raw_valid[index] = score >= confidence_threshold

        valid = self._reject_group_motion_outliers(seed_anchors, candidates, raw_valid)
        valid_count = int(np.count_nonzero(valid))
        left_count = int(np.count_nonzero(valid[:3]))
        right_count = int(np.count_nonzero(valid[3:]))
        if valid_count < self._minimum_valid_anchors or left_count < 2 or right_count < 2:
            return None

        updated = self._fill_missing_with_group_motion(seed_anchors, candidates, valid)
        shoulders = self._shoulders_from_anchors(updated, valid)
        if not self._geometry_is_plausible(updated, shoulders):
            return None
        return updated, valid, scores

    def _accept(
        self,
        result: tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]],
    ) -> TrackedShoulders:
        anchors, valid, scores = result
        self._anchors = anchors
        valid_count = int(np.count_nonzero(valid))
        self._valid_anchor_count = valid_count
        accepted_scores = scores[valid]
        mean_score = float(np.mean(accepted_scores)) if len(accepted_scores) else 0.0
        self._confidence = min(1.0, mean_score * (valid_count / 6.0))
        return self._shoulders_from_anchors(anchors, valid)

    def _fill_missing_with_group_motion(
        self,
        seed_anchors: NDArray[np.float32],
        candidates: NDArray[np.float32],
        valid: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        updated = candidates.copy()
        for start, end in ((0, 3), (3, 6)):
            side_valid = valid[start:end]
            side_delta = np.median(
                candidates[start:end][side_valid] - seed_anchors[start:end][side_valid],
                axis=0,
            ).astype(np.float32)
            for local_index in range(3):
                absolute_index = start + local_index
                if not valid[absolute_index]:
                    updated[absolute_index] = seed_anchors[absolute_index] + side_delta
        return updated

    def _extract_template(
        self,
        gray: NDArray[np.uint8],
        point: tuple[float, float],
    ) -> NDArray[np.uint8] | None:
        half = self._template_size // 2
        x = int(round(point[0]))
        y = int(round(point[1]))
        x1, x2 = x - half, x + half + 1
        y1, y2 = y - half, y + half + 1
        if x1 < 0 or y1 < 0 or x2 > gray.shape[1] or y2 > gray.shape[0]:
            return None
        return gray[y1:y2, x1:x2].copy()

    def _match_anchor(
        self,
        gray: NDArray[np.uint8],
        previous_point: NDArray[np.float32],
        templates: list[NDArray[np.uint8]],
        search_radius: int,
    ) -> tuple[NDArray[np.float32], float] | None:
        half = self._template_size // 2
        center_x = int(round(float(previous_point[0])))
        center_y = int(round(float(previous_point[1])))
        x1 = max(0, center_x - search_radius - half)
        y1 = max(0, center_y - search_radius - half)
        x2 = min(gray.shape[1], center_x + search_radius + half + 1)
        y2 = min(gray.shape[0], center_y + search_radius + half + 1)
        search = gray[y1:y2, x1:x2]
        if search.shape[0] < self._template_size or search.shape[1] < self._template_size:
            return None

        best_score = -1.0
        best_point: NDArray[np.float32] | None = None
        for template in templates:
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, maximum, _, location = cv2.minMaxLoc(result)
            score = float(maximum)
            if not np.isfinite(score) or score <= best_score:
                continue
            best_score = score
            best_point = np.asarray(
                [x1 + location[0] + half, y1 + location[1] + half],
                dtype=np.float32,
            )
        if best_point is None:
            return None
        return best_point, best_score

    def _reject_group_motion_outliers(
        self,
        old_anchors: NDArray[np.float32],
        candidates: NDArray[np.float32],
        raw_valid: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        valid = raw_valid.copy()
        for start, end in ((0, 3), (3, 6)):
            indices = np.flatnonzero(valid[start:end]) + start
            if len(indices) < 2:
                continue
            deltas = candidates[indices] - old_anchors[indices]
            median = np.median(deltas, axis=0)
            residuals = np.linalg.norm(deltas - median, axis=1)
            for index, residual in zip(indices, residuals):
                if float(residual) > self._maximum_group_motion_residual:
                    valid[index] = False
        return valid

    def _geometry_is_plausible(
        self,
        anchors: NDArray[np.float32],
        shoulders: TrackedShoulders,
    ) -> bool:
        calibration_widths = np.asarray(
            [pose.shoulders.width for pose in self._calibration_poses], dtype=np.float32
        )
        median_width = float(np.median(calibration_widths))
        if shoulders.left[0] >= shoulders.right[0]:
            return False
        if shoulders.width < median_width * 0.55 or shoulders.width > median_width * 1.50:
            return False
        if abs(shoulders.left[1] - shoulders.right[1]) > shoulders.width * 0.50:
            return False

        left = anchors[:3]
        right = anchors[3:]
        if not (left[0, 0] > left[1, 0] > left[2, 0]):
            return False
        if not (right[0, 0] < right[1, 0] < right[2, 0]):
            return False
        return True

    @staticmethod
    def _shoulders_from_anchors(
        anchors: NDArray[np.float32],
        valid: NDArray[np.bool_],
    ) -> TrackedShoulders:
        left = np.mean(anchors[:3], axis=0)
        right = np.mean(anchors[3:6], axis=0)
        return TrackedShoulders(
            left=(float(left[0]), float(left[1])),
            right=(float(right[0]), float(right[1])),
            anchors=tuple((float(point[0]), float(point[1])) for point in anchors),
            valid_anchors=tuple(bool(value) for value in valid),
        )
