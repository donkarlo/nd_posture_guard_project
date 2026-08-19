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
        recovery_search_radius_px: int = 190,
        optical_flow_window_px: int = 41,
        optical_flow_pyramid_levels: int = 4,
        optical_flow_forward_backward_error_px: float = 4.0,
    ) -> None:
        size = max(15, int(template_size_px))
        self._template_size = size if size % 2 == 1 else size + 1
        self._search_radius = max(18, int(search_radius_px))
        self._recovery_search_radius = max(
            int(recovery_search_radius_px), self._search_radius * 4
        )
        flow_window = max(21, int(optical_flow_window_px))
        self._flow_window = flow_window if flow_window % 2 == 1 else flow_window + 1
        self._flow_levels = max(2, int(optical_flow_pyramid_levels))
        self._flow_fb_error = max(1.0, float(optical_flow_forward_backward_error_px))
        self._minimum_match_confidence = max(
            0.05, min(float(minimum_match_confidence), 0.99)
        )
        self._recovery_match_confidence = max(
            0.25, self._minimum_match_confidence * 0.72
        )
        self._minimum_valid_anchors = max(4, int(minimum_valid_anchors))
        self._maximum_group_motion_residual = max(
            4.0, float(maximum_group_motion_residual_px)
        )
        self._calibration_poses: list[ShoulderCalibrationPose] = []
        self._calibration_anchor_sets: list[NDArray[np.float32]] = []
        self._templates: list[list[NDArray[np.uint8]]] = []
        self._anchors: NDArray[np.float32] | None = None
        self._previous_gray: NDArray[np.uint8] | None = None
        self._anchors_per_shoulder = 0
        self._confidence = 0.0
        self._valid_anchor_count = 0
        self._lost_frames = 0

    @property
    def ready(self) -> bool:
        total = self._total_anchors
        return (
            len(self._calibration_poses) >= 3
            and total >= 4
            and self._anchors is not None
            and len(self._templates) == total
            and all(len(anchor_templates) >= 3 for anchor_templates in self._templates)
        )

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def valid_anchor_count(self) -> int:
        return self._valid_anchor_count

    @property
    def total_anchor_count(self) -> int:
        return self._total_anchors

    @property
    def _total_anchors(self) -> int:
        return self._anchors_per_shoulder * 2

    def clear(self) -> None:
        self._calibration_poses.clear()
        self._calibration_anchor_sets.clear()
        self._templates = []
        self._anchors = None
        self._previous_gray = None
        self._anchors_per_shoulder = 0
        self._confidence = 0.0
        self._valid_anchor_count = 0
        self._lost_frames = 0


    def prepare_for_reacquire(self) -> None:
        """Keep calibration but discard frame-to-frame tracking state."""
        self._previous_gray = None
        if self._calibration_anchor_sets:
            self._anchors = self._calibration_anchor_sets[0].copy()
        self._confidence = 0.0
        self._valid_anchor_count = 0
        self._lost_frames = 0

    def add_calibration_pose(
        self,
        frame: NDArray[np.uint8],
        pose: ShoulderCalibrationPose,
    ) -> None:
        if self._anchors_per_shoulder == 0:
            self._anchors_per_shoulder = pose.points_per_shoulder
            self._templates = [[] for _ in range(self._total_anchors)]
        elif pose.points_per_shoulder != self._anchors_per_shoulder:
            raise ValueError("Every calibration pose must use the same number of shoulder points.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pose_templates: list[NDArray[np.uint8]] = []
        for index, point in enumerate(pose.all_points):
            patch = self._extract_template(gray, point)
            if patch is None:
                raise ValueError(
                    f"Shoulder anchor {index + 1} is too close to the image edge. "
                    "Move slightly farther from the camera and calibrate again."
                )
            if float(np.std(patch)) < 2.0:
                raise ValueError(
                    f"Shoulder anchor {index + 1} has too little visible texture/contrast. "
                    "Place it on the visible shoulder edge, seam, or textured clothing and calibrate again."
                )
            pose_templates.append(patch)

        anchor_set = np.asarray(pose.all_points, dtype=np.float32)
        self._calibration_poses.append(pose)
        self._calibration_anchor_sets.append(anchor_set)
        for index, patch in enumerate(pose_templates):
            self._templates[index].append(patch)

        self._anchors = (
            self._calibration_anchor_sets[0].copy()
            if len(self._calibration_anchor_sets) >= 3
            else anchor_set.copy()
        )
        self._previous_gray = None
        self._confidence = 1.0 if self.ready else 0.0
        self._valid_anchor_count = self._total_anchors if self.ready else 0
        self._lost_frames = 0

    def track(self, frame: NDArray[np.uint8]) -> TrackedShoulders | None:
        if not self.ready:
            self._confidence = 0.0
            self._valid_anchor_count = 0
            return None
        assert self._anchors is not None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._previous_gray is not None:
            flowed = self._track_with_optical_flow(self._previous_gray, gray)
            if flowed is not None:
                self._lost_frames = 0
                return self._accept(flowed, gray)

        recovered = self._recover(gray)
        if recovered is not None:
            self._lost_frames = 0
            return self._accept(recovered, gray)

        self._lost_frames += 1
        self._confidence = 0.0
        self._valid_anchor_count = 0
        self._previous_gray = None
        return None

    def _track_with_optical_flow(
        self,
        previous_gray: NDArray[np.uint8],
        gray: NDArray[np.uint8],
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]] | None:
        assert self._anchors is not None
        old_points = self._anchors.reshape(-1, 1, 2).astype(np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        )
        new_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            old_points,
            None,
            winSize=(self._flow_window, self._flow_window),
            maxLevel=self._flow_levels,
            criteria=criteria,
        )
        if new_points is None or forward_status is None:
            return None
        back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            previous_gray,
            new_points,
            None,
            winSize=(self._flow_window, self._flow_window),
            maxLevel=self._flow_levels,
            criteria=criteria,
        )
        if back_points is None or backward_status is None:
            return None

        candidates = new_points.reshape(-1, 2).astype(np.float32)
        back = back_points.reshape(-1, 2).astype(np.float32)
        old = old_points.reshape(-1, 2)
        fb_error = np.linalg.norm(back - old, axis=1)
        raw_valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.isfinite(candidates).all(axis=1)
            & (fb_error <= self._flow_fb_error)
        )
        raw_valid &= self._points_inside_frame(candidates, gray.shape[1], gray.shape[0])
        valid = self._reject_group_motion_outliers(old, candidates, raw_valid)
        if not self._enough_valid(valid):
            return None

        updated = self._fill_missing_with_group_motion(old, candidates, valid)
        shoulders = self._shoulders_from_anchors(updated, valid)
        if not self._geometry_is_plausible(updated, shoulders):
            return None

        scores = np.zeros(self._total_anchors, dtype=np.float32)
        scores[valid] = np.clip(1.0 - fb_error[valid] / self._flow_fb_error, 0.0, 1.0)
        return updated, valid, scores

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
            quality = float(np.mean(accepted_scores)) * (
                float(np.count_nonzero(valid)) / max(self._total_anchors, 1)
            )
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
        total = self._total_anchors
        candidates = seed_anchors.copy()
        scores = np.zeros(total, dtype=np.float32)
        raw_valid = np.zeros(total, dtype=bool)

        for index in range(total):
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
        if not self._enough_valid(valid):
            return None

        updated = self._fill_missing_with_group_motion(seed_anchors, candidates, valid)
        shoulders = self._shoulders_from_anchors(updated, valid)
        if not self._geometry_is_plausible(updated, shoulders):
            return None
        return updated, valid, scores

    def _accept(
        self,
        result: tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]],
        gray: NDArray[np.uint8],
    ) -> TrackedShoulders:
        anchors, valid, scores = result
        self._anchors = anchors
        self._previous_gray = gray.copy()
        valid_count = int(np.count_nonzero(valid))
        self._valid_anchor_count = valid_count
        accepted_scores = scores[valid]
        mean_score = float(np.mean(accepted_scores)) if len(accepted_scores) else 0.0
        self._confidence = min(
            1.0,
            max(0.0, mean_score) * (valid_count / max(self._total_anchors, 1)),
        )
        return self._shoulders_from_anchors(anchors, valid)

    def _enough_valid(self, valid: NDArray[np.bool_]) -> bool:
        side = self._anchors_per_shoulder
        valid_count = int(np.count_nonzero(valid))
        left_count = int(np.count_nonzero(valid[:side]))
        right_count = int(np.count_nonzero(valid[side:]))
        required_total = min(max(self._minimum_valid_anchors, 4), self._total_anchors)
        required_side = 2 if side <= 3 else 2
        return (
            valid_count >= required_total
            and left_count >= required_side
            and right_count >= required_side
        )

    def _fill_missing_with_group_motion(
        self,
        seed_anchors: NDArray[np.float32],
        candidates: NDArray[np.float32],
        valid: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        updated = candidates.copy()
        side = self._anchors_per_shoulder
        for start, end in ((0, side), (side, side * 2)):
            side_valid = valid[start:end]
            if not np.any(side_valid):
                continue
            side_delta = np.median(
                candidates[start:end][side_valid] - seed_anchors[start:end][side_valid],
                axis=0,
            ).astype(np.float32)
            for local_index in range(side):
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
        side = self._anchors_per_shoulder
        for start, end in ((0, side), (side, side * 2)):
            indices = np.flatnonzero(valid[start:end]) + start
            if len(indices) < 3:
                continue
            deltas = candidates[indices] - old_anchors[indices]
            median = np.median(deltas, axis=0)
            residuals = np.linalg.norm(deltas - median, axis=1)
            dynamic_limit = self._maximum_group_motion_residual + max(
                0.0, float(np.linalg.norm(median)) * 0.18
            )
            for index, residual in zip(indices, residuals):
                if float(residual) > dynamic_limit:
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
        if shoulders.width < median_width * 0.42 or shoulders.width > median_width * 1.75:
            return False
        if abs(shoulders.left[1] - shoulders.right[1]) > shoulders.width * 0.72:
            return False

        side = self._anchors_per_shoulder
        left = anchors[:side]
        right = anchors[side:]
        left_progress = np.diff(left[:, 0])
        right_progress = np.diff(right[:, 0])
        if np.count_nonzero(left_progress < 0) < max(1, side - 2):
            return False
        if np.count_nonzero(right_progress > 0) < max(1, side - 2):
            return False
        return True

    def _shoulders_from_anchors(
        self,
        anchors: NDArray[np.float32],
        valid: NDArray[np.bool_],
    ) -> TrackedShoulders:
        side = self._anchors_per_shoulder
        left = np.mean(anchors[:side], axis=0)
        right = np.mean(anchors[side : side * 2], axis=0)
        return TrackedShoulders(
            left=(float(left[0]), float(left[1])),
            right=(float(right[0]), float(right[1])),
            anchors=tuple((float(point[0]), float(point[1])) for point in anchors),
            valid_anchors=tuple(bool(value) for value in valid),
        )

    @staticmethod
    def _points_inside_frame(
        points: NDArray[np.float32],
        width: int,
        height: int,
    ) -> NDArray[np.bool_]:
        return (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
