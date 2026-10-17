#!/usr/bin/env python3
"""CPU-only online localization backend for causal environment validation.

This is an engineering smoke backend, not a paper baseline.  For every
``localize`` action it recomputes ORB features for the current query and the
requested map references, intersects query tracks, triangulates map points
from map poses, and solves a query PnP pose.  It never reads query GT and does
not reuse a frozen pose candidate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
import time

import cv2
import numpy as np

from tools.agentic_localization_env import Candidate
from tools.eval_lilocbench_reloc3r import (
    base_to_front_rgb_optical,
    read_manifest,
    rotation_error,
    tum_camera_poses,
)


def state_trust(anchor_age: int, reject_streak: int) -> float:
    return float(np.exp(-float(anchor_age) / 20.0 - float(reject_streak) / 3.0))


def previous_state_risk(candidate_pose: np.ndarray, accepted_pose: Any) -> float:
    if accepted_pose is None:
        return 0.0
    previous = np.asarray(accepted_pose, dtype=np.float64)
    return max(
        float(np.linalg.norm(previous[:3, 3] - candidate_pose[:3, 3])) / 2.0,
        float(rotation_error(previous[:3, :3], candidate_pose[:3, :3])) / 30.0,
    )


class CpuOrbLocalizationBackend:
    """Reference-conditioned ORB/triangulation/PnP backend."""

    def __init__(self, data_root: Path, mapping_gt: Path, query_sequence: str,
                 retrieval_npz: Path, *, nfeatures: int = 1800,
                 ratio_test: float = 0.78, min_inliers: int = 20,
                 max_reprojection_px: float = 5.0,
                 min_pair_support_ratio: float = 0.5,
                 max_state_risk: float = 1.5):
        self.data_root = Path(data_root)
        self.query_sequence = str(query_sequence)
        self.nfeatures = int(nfeatures)
        self.ratio_test = float(ratio_test)
        self.min_inliers = int(min_inliers)
        self.max_reprojection_px = float(max_reprojection_px)
        self.min_pair_support_ratio = float(min_pair_support_ratio)
        self.max_state_risk = float(max_state_risk)
        _, self.mapping_frames, mapping_times, self.K, mapping_root = read_manifest(
            self.data_root, "mapping")
        _, self.query_frames, self.query_times, _, _ = read_manifest(
            self.data_root, self.query_sequence)
        self.mapping_pose, _ = tum_camera_poses(
            Path(mapping_gt), mapping_times,
            base_to_front_rgb_optical(mapping_root), 0.1)
        retrieval = np.load(Path(retrieval_npz))
        self.retrieval_indices = np.asarray(retrieval["mapping_indices"], dtype=np.int64)
        self.retrieval_scores = np.asarray(retrieval["mapping_scores"], dtype=np.float32)
        if len(self.retrieval_indices) != len(self.query_frames):
            raise ValueError("retrieval rows do not match query frames")
        self._action_counter = 0

    def num_frames(self, sequence: str) -> int:
        if str(sequence) != self.query_sequence:
            raise ValueError("backend was initialized for " + self.query_sequence)
        return len(self.query_frames)

    def frame_observation(self, sequence: str, frame_index: int,
                          state: Mapping[str, Any]) -> Mapping[str, Any]:
        del state
        refs = [int(x) for x in self.retrieval_indices[frame_index]]
        scores = [float(x) for x in self.retrieval_scores[frame_index]]
        groups = []
        for start in range(0, len(refs), 3):
            group_refs = refs[start:start + 3]
            group_scores = scores[start:start + 3]
            if not group_refs:
                continue
            translations = [self.mapping_pose[index][:3, 3] for index in group_refs]
            spread_m = 0.0
            if len(translations) > 1:
                spread_m = float(np.median([
                    np.linalg.norm(translations[i] - translations[j])
                    for i in range(len(translations)) for j in range(i + 1, len(translations))]))
            groups.append({
                "reference_indices": group_refs,
                "retrieval_scores": group_scores,
                "mean_retrieval_score": float(np.mean(group_scores)),
                "map_spread_m": spread_m,
            })
        return {
            "image_path": str(self.query_frames[frame_index]["path"]),
            "timestamp": float(self.query_times[frame_index]),
            "retrieval_references": refs,
            "retrieval_scores": scores,
            "reference_groups": groups,
            "mapping_size": len(self.mapping_frames),
        }

    @staticmethod
    def _audit() -> Dict[str, bool]:
        return {"query_gt_read": False, "uses_future": False, "selection_uses_gt": False}

    def _features(self, path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        orb = cv2.ORB_create(nfeatures=self.nfeatures)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is None:
            descriptors = np.empty((0, 32), dtype=np.uint8)
        return keypoints, descriptors

    @staticmethod
    def _sift_features(path: Path):
        """Higher-cost CPU geometry action used only when explicitly chosen.

        This is not a replacement for the GPU RoMa action.  It is a real
        online SIFT/triangulation/PnP computation so sequential-policy tests
        retain a distinct, non-cached strong-evidence branch on CPU-only
        machines.
        """
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        sift = cv2.SIFT_create(nfeatures=max(2 * 1800, 3000))
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)
        return keypoints, descriptors

    def _query_reference_matches_sift(self, query_kp, query_desc, ref_path: Path):
        ref_kp, ref_desc = self._sift_features(ref_path)
        if len(query_desc) == 0 or len(ref_desc) == 0:
            return {}, len(query_kp), len(ref_kp)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(query_desc, ref_desc, k=2)
        selected = {}
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                selected[int(pair[0].queryIdx)] = (
                    np.asarray(ref_kp[pair[0].trainIdx].pt, dtype=np.float64),
                    float(pair[0].distance))
        return selected, len(query_kp), len(ref_kp)

    def _query_reference_matches(self, query_kp, query_desc, ref_path: Path):
        ref_kp, ref_desc = self._features(ref_path)
        if len(query_desc) == 0 or len(ref_desc) == 0:
            return {}, len(query_kp), len(ref_kp)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(query_desc, ref_desc, k=2)
        selected = {}
        for pair in pairs:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.ratio_test * second.distance:
                selected[int(first.queryIdx)] = (
                    np.asarray(ref_kp[first.trainIdx].pt, dtype=np.float64),
                    float(first.distance),
                )
        return selected, len(query_kp), len(ref_kp)

    def _query_image_matches(self, query_desc, ref_desc):
        """Return current-query keypoint -> reference keypoint matches."""
        if len(query_desc) == 0 or len(ref_desc) == 0:
            return {}
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(query_desc, ref_desc, k=2)
        selected = {}
        for pair in pairs:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.ratio_test * second.distance:
                selected[int(first.queryIdx)] = (int(first.trainIdx), float(first.distance))
        return selected

    @staticmethod
    def _camera_matrix(c2w: np.ndarray, K: np.ndarray) -> np.ndarray:
        R = c2w[:3, :3].T
        t = -R @ c2w[:3, 3]
        return K @ np.column_stack([R, t])

    def _heldout_epipolar_evidence(self, frame_index: int,
                                   candidate_pose: np.ndarray,
                                   used_refs: Sequence[int]) -> Dict[str, Any]:
        """Acquire verification evidence from references not used to pose.

        This is deliberately a second matching route.  It does not reuse the
        candidate's inlier count: held-out query/reference matches are scored
        against the candidate pose through the predicted fundamental matrix.
        """
        all_refs = [int(x) for x in self.retrieval_indices[frame_index]]
        heldout = [x for x in all_refs if x not in set(int(r) for r in used_refs)][:3]
        if not heldout:
            return {"available": False, "reason": "no_heldout_reference"}
        query_kp, query_desc = self._features(Path(self.query_frames[frame_index]["path"]))
        R_query = candidate_pose[:3, :3].T
        total_matches = 0
        inliers = 0
        errors = []
        per_reference = []
        for ref in heldout:
            ref_kp, ref_desc = self._features(Path(self.mapping_frames[ref]["path"]))
            matches = self._query_image_matches(query_desc, ref_desc)
            if not matches:
                per_reference.append({"reference": ref, "matches": 0, "inliers": 0})
                continue
            map_pose = self.mapping_pose[ref]
            # Both stored poses are camera-to-world; convert map rotation to
            # world-to-camera before forming the ref->query relative motion.
            R_map = map_pose[:3, :3].T
            c_map = map_pose[:3, 3]
            R_rel = R_query @ R_map.T
            t_rel = R_query @ (c_map - candidate_pose[:3, 3])
            t_norm = float(np.linalg.norm(t_rel))
            if t_norm < 1e-8:
                per_reference.append({"reference": ref, "matches": len(matches), "inliers": 0})
                continue
            tx = np.array([[0.0, -t_rel[2], t_rel[1]],
                          [t_rel[2], 0.0, -t_rel[0]],
                          [-t_rel[1], t_rel[0], 0.0]])
            F = np.linalg.inv(self.K).T @ (tx @ R_rel) @ np.linalg.inv(self.K)
            ref_points, query_points = [], []
            for query_index, (ref_index, _) in matches.items():
                query_points.append(query_kp[query_index].pt)
                ref_points.append(ref_kp[ref_index].pt)
            ref_h = np.column_stack([np.asarray(ref_points, dtype=np.float64),
                                     np.ones(len(ref_points))])
            query_h = np.column_stack([np.asarray(query_points, dtype=np.float64),
                                       np.ones(len(query_points))])
            lines_q = (F @ ref_h.T).T
            lines_r = (F.T @ query_h.T).T
            d_q = np.abs(np.sum(query_h * lines_q, axis=1)) / np.maximum(
                np.linalg.norm(lines_q[:, :2], axis=1), 1e-8)
            d_r = np.abs(np.sum(ref_h * lines_r, axis=1)) / np.maximum(
                np.linalg.norm(lines_r[:, :2], axis=1), 1e-8)
            distances = 0.5 * (d_q + d_r)
            good = distances <= 4.0
            total_matches += len(distances)
            inliers += int(np.count_nonzero(good))
            errors.extend(float(x) for x in distances)
            per_reference.append({"reference": ref, "matches": len(distances),
                                  "inliers": int(np.count_nonzero(good)),
                                  "median_epipolar_px": float(np.median(distances))})
        ratio = inliers / total_matches if total_matches else 0.0
        return {
            "available": total_matches > 0,
            "heldout_references": heldout,
            "matches": int(total_matches),
            "inliers": int(inliers),
            "inlier_ratio": float(ratio),
            "median_epipolar_px": None if not errors else float(np.median(errors)),
            "per_reference": per_reference,
        }

    def _triangulate_pair(self, first, second, query_kp, refs):
        common = sorted(set(first) & set(second))
        if len(common) < 8:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64), 0
        qxy = np.asarray([query_kp[i].pt for i in common], dtype=np.float64)
        first_xy = np.asarray([first[i][0] for i in common], dtype=np.float64)
        second_xy = np.asarray([second[i][0] for i in common], dtype=np.float64)
        P1 = self._camera_matrix(self.mapping_pose[refs[0]], self.K)
        P2 = self._camera_matrix(self.mapping_pose[refs[1]], self.K)
        points = cv2.triangulatePoints(P1, P2, first_xy.T, second_xy.T).T
        valid = np.abs(points[:, 3]) > 1e-8
        xyz = np.zeros((len(points), 3), dtype=np.float64)
        xyz[valid] = points[valid, :3] / points[valid, 3:4]
        finite = valid & np.all(np.isfinite(xyz), axis=1) & (np.linalg.norm(xyz, axis=1) < 100.0)
        return xyz[finite], qxy[finite], int(np.count_nonzero(finite))

    def localize(self, sequence: str, frame_index: int,
                 reference_indices: Sequence[int],
                 state: Mapping[str, Any]) -> Mapping[str, Any]:
        del state
        started = time.perf_counter()
        if str(sequence) != self.query_sequence:
            raise ValueError("unexpected sequence")
        refs = tuple(int(x) for x in reference_indices)
        if len(refs) < 2 or any(x < 0 or x >= len(self.mapping_frames) for x in refs):
            raise ValueError("ORB localization requires at least two valid map references")
        query_path = Path(self.query_frames[frame_index]["path"])
        query_kp, query_desc = self._features(query_path)
        ref_matches = []
        feature_counts = []
        for ref in refs:
            matches, q_count, r_count = self._query_reference_matches(
                query_kp, query_desc, Path(self.mapping_frames[ref]["path"]))
            ref_matches.append(matches)
            feature_counts.append({"query": q_count, "reference": r_count, "matches": len(matches)})

        all_xyz, all_qxy = [], []
        pair_support = []
        for left in range(len(refs)):
            for right in range(left + 1, len(refs)):
                xyz, qxy, count = self._triangulate_pair(
                    ref_matches[left], ref_matches[right], query_kp,
                    (refs[left], refs[right]))
                if len(xyz):
                    all_xyz.append(xyz)
                    all_qxy.append(qxy)
                pair_support.append(count)
        if all_xyz:
            xyz = np.concatenate(all_xyz, axis=0)
            qxy = np.concatenate(all_qxy, axis=0)
            correspondences = int(len(xyz))
            pose, inliers, reprojection = self._solve_pnp(xyz, qxy)
        else:
            pose, inliers, reprojection, correspondences = None, 0, None, 0
        self._action_counter += 1
        support_ratio = (sum(x >= 8 for x in pair_support) / len(pair_support)
                         if pair_support else 0.0)
        return {
            "pose": None if pose is None else pose.tolist(),
            "source": "orb_online",
            "reference_indices": list(refs),
            "evidence": {
                "num_references": len(refs),
                "inliers": int(inliers),
                "correspondences": correspondences,
                "median_reprojection_px": reprojection,
                "pair_support": pair_support,
                "pair_support_ratio": support_ratio,
                "feature_counts": feature_counts,
            },
            "execution": {
                "executed": True, "cache_hit": False,
                "tool": "ORB-triangulation-PnP",
                "stages": ["orb_feature_matching", "triangulation", "pnp_pose_estimation"],
                "query_frame_index": int(frame_index),
                "reference_indices": list(refs),
                "action_counter": self._action_counter,
                "elapsed_s": time.perf_counter() - started,
            },
            "causal": self._audit(),
        }

    def strong_geometry(self, sequence: str, frame_index: int,
                        reference_indices: Sequence[int],
                        state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute an online, higher-cost SIFT+multi-view geometry branch.

        The experiment marks this as a CPU proxy.  The paper-grade GPU
        backend exposes RoMa under the same action, so policy and budget code
        never confuse a cached recovery with either implementation.
        """
        del state
        started = time.perf_counter()
        if str(sequence) != self.query_sequence:
            raise ValueError("unexpected sequence")
        refs = tuple(int(x) for x in reference_indices)
        if len(refs) < 2 or any(x < 0 or x >= len(self.mapping_frames) for x in refs):
            raise ValueError("strong geometry requires at least two valid map references")
        query_kp, query_desc = self._sift_features(Path(self.query_frames[frame_index]["path"]))
        ref_matches, feature_counts = [], []
        for ref in refs:
            matches, q_count, r_count = self._query_reference_matches_sift(
                query_kp, query_desc, Path(self.mapping_frames[ref]["path"]))
            ref_matches.append(matches)
            feature_counts.append({"query": q_count, "reference": r_count, "matches": len(matches)})
        all_xyz, all_qxy, pair_support = [], [], []
        for left in range(len(refs)):
            for right in range(left + 1, len(refs)):
                xyz, qxy, count = self._triangulate_pair(
                    ref_matches[left], ref_matches[right], query_kp,
                    (refs[left], refs[right]))
                pair_support.append(count)
                if len(xyz):
                    all_xyz.append(xyz)
                    all_qxy.append(qxy)
        if all_xyz:
            xyz = np.concatenate(all_xyz, axis=0)
            qxy = np.concatenate(all_qxy, axis=0)
            pose, inliers, reprojection = self._solve_pnp(xyz, qxy)
            correspondences = int(len(xyz))
        else:
            pose, inliers, reprojection, correspondences = None, 0, None, 0
        self._action_counter += 1
        support_ratio = (sum(x >= 8 for x in pair_support) / len(pair_support)
                         if pair_support else 0.0)
        return {
            "pose": None if pose is None else pose.tolist(),
            "source": "sift_online_geometry_cpu_proxy",
            "reference_indices": list(refs),
            "evidence": {"num_references": len(refs), "inliers": int(inliers),
                         "correspondences": correspondences,
                         "median_reprojection_px": reprojection,
                         "pair_support": pair_support,
                         "pair_support_ratio": support_ratio,
                         "feature_counts": feature_counts,
                         "geometry_backend": "SIFT+triangulation+PnP CPU proxy"},
            "execution": {"executed": True, "cache_hit": False,
                          "tool": "SIFT-triangulation-PnP (CPU proxy)",
                          "stages": ["sift_feature_matching", "multi_view_triangulation",
                                     "pnp_pose_estimation"],
                          "query_frame_index": int(frame_index),
                          "reference_indices": list(refs),
                          "action_counter": self._action_counter,
                          "elapsed_s": time.perf_counter() - started},
            "causal": self._audit(),
        }

    def _solve_pnp(self, xyz: np.ndarray, qxy: np.ndarray):
        if len(xyz) < 6:
            return None, 0, None
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            xyz.astype(np.float64), qxy.astype(np.float64), self.K, None,
            iterationsCount=500, reprojectionError=self.max_reprojection_px,
            confidence=0.995, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < 4:
            return None, 0, None
        R, _ = cv2.Rodrigues(rvec)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = (-R.T @ tvec).reshape(3)
        projected, _ = cv2.projectPoints(xyz[inliers[:, 0]], rvec, tvec, self.K, None)
        error = np.linalg.norm(projected[:, 0, :] - qxy[inliers[:, 0]], axis=1)
        return c2w, int(len(inliers)), float(np.median(error))

    def retrieve_memory(self, sequence: str, frame_index: int,
                        memory: Sequence[Mapping[str, Any]],
                        state: Mapping[str, Any]) -> Mapping[str, Any]:
        del state
        started = time.perf_counter()
        if str(sequence) != self.query_sequence:
            raise ValueError("unexpected sequence")
        usable = [item for item in memory if int(item["frame_index"]) < frame_index]
        if not usable:
            return {"pose": None, "source": "memory", "evidence": {"available": False},
                    "execution": {"executed": True, "cache_hit": False,
                                  "tool": "historical-keyframe-ORB-triangulation-PnP",
                                  "stages": ["memory_availability_check"],
                                  "query_frame_index": int(frame_index), "reference_indices": [],
                                  "elapsed_s": time.perf_counter() - started},
                    "causal": self._audit()}
        item = usable[-1]
        keyframe_index = int(item["frame_index"])
        metadata = dict(item.get("metadata", {}))
        refs = tuple(int(x) for x in metadata.get("references", ()))
        if len(refs) < 2:
            return {"pose": None, "source": "memory",
                    "evidence": {"available": False, "reason": "keyframe_has_no_map_support",
                                  "memory_id": int(item["memory_id"])},
                    "execution": {"executed": True, "cache_hit": False,
                                  "tool": "historical-keyframe-ORB-triangulation-PnP",
                                  "stages": ["memory_provenance_check"],
                                  "query_frame_index": int(frame_index), "reference_indices": [],
                                  "elapsed_s": time.perf_counter() - started},
                    "causal": self._audit()}

        # Build a small local map around the historical keyframe.  The
        # accepted historical global pose is used only as provenance; the
        # current pose is re-estimated from current-to-keyframe tracks and
        # metric 3D points triangulated from the historical keyframe's map
        # references.  This avoids the invalid "copy the old pose" shortcut.
        current_path = Path(self.query_frames[frame_index]["path"])
        keyframe_path = Path(self.query_frames[keyframe_index]["path"])
        current_kp, current_desc = self._features(current_path)
        keyframe_kp, keyframe_desc = self._features(keyframe_path)
        historical_matches = []
        for ref in refs:
            matches, _, _ = self._query_reference_matches(
                keyframe_kp, keyframe_desc, Path(self.mapping_frames[ref]["path"]))
            historical_matches.append(matches)
        all_xyz, pair_support = [], []
        for left in range(len(refs)):
            for right in range(left + 1, len(refs)):
                common = sorted(set(historical_matches[left]) & set(historical_matches[right]))
                pair_support.append(len(common))
                if len(common) < 8:
                    continue
                first_xy = np.asarray([historical_matches[left][i][0] for i in common], dtype=np.float64)
                second_xy = np.asarray([historical_matches[right][i][0] for i in common], dtype=np.float64)
                P1 = self._camera_matrix(self.mapping_pose[refs[left]], self.K)
                P2 = self._camera_matrix(self.mapping_pose[refs[right]], self.K)
                points = cv2.triangulatePoints(P1, P2, first_xy.T, second_xy.T).T
                valid = np.abs(points[:, 3]) > 1e-8
                xyz = np.zeros((len(points), 3), dtype=np.float64)
                xyz[valid] = points[valid, :3] / points[valid, 3:4]
                finite = valid & np.all(np.isfinite(xyz), axis=1) & (np.linalg.norm(xyz, axis=1) < 100.0)
                if np.any(finite):
                    all_xyz.append(xyz[finite])

        pose = None
        inliers = 0
        reprojection = None
        correspondences = 0
        if all_xyz:
            keyframe_to_current = self._query_image_matches(current_desc, keyframe_desc)
            current_xy, matched_xyz = [], []
            # Duplicate historical tracks are harmless for RANSAC; retain the
            # first metric point associated with each historical keypoint.
            # Reconstruct the association using each pair's common keypoint
            # IDs, rather than image coordinates, so descriptor matches remain
            # causal and auditable.
            point_by_historical = {}
            for left in range(len(refs)):
                for right in range(left + 1, len(refs)):
                    common = sorted(set(historical_matches[left]) & set(historical_matches[right]))
                    if len(common) < 8:
                        continue
                    first_xy = np.asarray([historical_matches[left][i][0] for i in common], dtype=np.float64)
                    second_xy = np.asarray([historical_matches[right][i][0] for i in common], dtype=np.float64)
                    points = cv2.triangulatePoints(
                        self._camera_matrix(self.mapping_pose[refs[left]], self.K),
                        self._camera_matrix(self.mapping_pose[refs[right]], self.K),
                        first_xy.T, second_xy.T).T
                    valid = np.abs(points[:, 3]) > 1e-8
                    for hist_index, point, is_valid in zip(common, points, valid):
                        if not is_valid:
                            continue
                        xyz_point = point[:3] / point[3]
                        if np.all(np.isfinite(xyz_point)) and np.linalg.norm(xyz_point) < 100.0:
                            point_by_historical.setdefault(int(hist_index), xyz_point)
            for current_index, (hist_index, _) in keyframe_to_current.items():
                if hist_index in point_by_historical:
                    current_xy.append(current_kp[current_index].pt)
                    matched_xyz.append(point_by_historical[hist_index])
            if matched_xyz:
                xyz_current = np.asarray(matched_xyz, dtype=np.float64)
                xy_current = np.asarray(current_xy, dtype=np.float64)
                correspondences = len(xyz_current)
                pose, inliers, reprojection = self._solve_pnp(xyz_current, xy_current)
        support_ratio = (sum(x >= 8 for x in pair_support) / len(pair_support)
                         if pair_support else 0.0)
        return {
            "pose": None if pose is None else pose.tolist(),
            "source": "historical_keyframe_orb",
            "reference_indices": list(refs),
            "evidence": {
                "available": pose is not None,
                "memory_id": int(item["memory_id"]),
                "keyframe_index": keyframe_index,
                "reference_indices": list(refs),
                "elapsed_s": time.perf_counter() - started,
                "inliers": int(inliers),
                "correspondences": int(correspondences),
                "median_reprojection_px": reprojection,
                "pair_support": pair_support,
                "pair_support_ratio": support_ratio,
            },
            "execution": {
                "executed": True, "cache_hit": False,
                "tool": "historical-keyframe-ORB-triangulation-PnP",
                "stages": ["keyframe_feature_matching", "local_map_triangulation",
                           "current_keyframe_matching", "pnp_pose_estimation"],
                "query_frame_index": int(frame_index),
                "historical_frame_index": keyframe_index,
                "reference_indices": list(refs),
            },
            "causal": self._audit(),
        }

    def verify(self, sequence: str, frame_index: int, candidate: Candidate,
               state: Mapping[str, Any]) -> Mapping[str, Any]:
        del sequence
        started = time.perf_counter()
        if candidate.pose is None:
            return {"accepted": False, "evidence": {"reason": "missing_pose", "elapsed_s": time.perf_counter() - started},
                    "causal": self._audit()}
        inliers = int(candidate.evidence.get("inliers", 0))
        reprojection = candidate.evidence.get("median_reprojection_px")
        support_ratio = float(candidate.evidence.get("pair_support_ratio", 0.0))
        raw_state_risk = previous_state_risk(candidate.pose, state.get("accepted_pose"))
        trust = state_trust(int(state.get("anchor_age", 0)), int(state.get("reject_streak", 0)))
        relaxed_state_risk = trust * raw_state_risk
        quality_risk = max(
            max(0.0, float(self.min_inliers - inliers)) / max(self.min_inliers, 1),
            0.0 if reprojection is None else float(reprojection) / self.max_reprojection_px,
            max(0.0, 1.0 - support_ratio),
        )
        independent = self._heldout_epipolar_evidence(
            frame_index, candidate.pose, candidate.reference_indices)
        heldout_matches = int(independent.get("matches", 0))
        heldout_inliers = int(independent.get("inliers", 0))
        heldout_ratio = float(independent.get("inlier_ratio", 0.0))
        heldout_median = independent.get("median_epipolar_px")
        independent_risk = max(
            max(0.0, 12.0 - heldout_inliers) / 12.0,
            max(0.0, 0.35 - heldout_ratio) / 0.35,
            0.0 if heldout_median is None else float(heldout_median) / 4.0,
        )
        total_risk = max(quality_risk, relaxed_state_risk)
        total_risk = max(total_risk, independent_risk)
        accepted = (inliers >= self.min_inliers and reprojection is not None and
                    float(reprojection) <= self.max_reprojection_px and
                    support_ratio >= self.min_pair_support_ratio and
                    independent.get("available", False) and heldout_inliers >= 12 and
                    heldout_ratio >= 0.35 and heldout_median is not None and
                    float(heldout_median) <= 4.0 and
                    total_risk <= self.max_state_risk)
        return {"accepted": accepted, "evidence": {
            "inliers": inliers, "median_reprojection_px": reprojection,
            "pair_support_ratio": support_ratio,
            "quality_risk": quality_risk,
            "previous_state_risk": raw_state_risk,
            "state_trust": trust,
            "relaxed_state_risk": relaxed_state_risk,
            "independent_verification": independent,
            "independent_risk": independent_risk,
            "total_risk": total_risk,
            "confidence": float(np.exp(-total_risk)),
            "elapsed_s": time.perf_counter() - started,
        }, "causal": self._audit()}


__all__ = ["CpuOrbLocalizationBackend", "previous_state_risk", "state_trust"]
