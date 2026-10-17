#!/usr/bin/env python3
"""Online LILocBench backend for :mod:`agentic_localization_env`.

Unlike the AVL replay scripts, ``localize`` executes Reloc3r on the current
query image and exactly the references supplied by the action.  Retrieval is
only a proposal mechanism; it is not a frozen pose-candidate pool.  Query GT
is intentionally not loaded by this module.
"""

from __future__ import annotations

import sys
import time
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reloc3r.reloc3r_relpose import inference_relpose, setup_reloc3r_relpose_model  # noqa: E402
from reloc3r.reloc3r_visloc import Reloc3rVisloc  # noqa: E402
from tools.agentic_localization_env import Candidate  # noqa: E402
from tools.eval_lilocbench_reloc3r import (  # noqa: E402
    base_to_front_rgb_optical,
    candidate_dispersion,
    make_batch,
    read_manifest,
    rotation_error,
    tum_camera_poses,
)


def state_trust(anchor_age: int, reject_streak: int) -> float:
    """Causal trust in the last accepted anchor before current evidence."""
    return float(np.exp(-float(anchor_age) / 20.0 - float(reject_streak) / 3.0))


def previous_state_risk(candidate_pose: np.ndarray,
                        accepted_pose: Any) -> float:
    """Dimensionless disagreement with the previous accepted state."""
    if accepted_pose is None:
        return 0.0
    previous = np.asarray(accepted_pose, dtype=np.float64)
    return max(
        float(np.linalg.norm(previous[:3, 3] - candidate_pose[:3, 3])) / 2.0,
        float(rotation_error(previous[:3, :3], candidate_pose[:3, :3])) / 30.0,
    )


class LILocBenchReloc3rBackend:
    """Reference-conditioned Reloc3r backend with an evidence-only verifier."""

    def __init__(self, data_root: Path, mapping_gt: Path, query_sequence: str,
                 retrieval_npz: Path, *, device: str = "cuda",
                 model_name: str = "Reloc3rRelpose(img_size=512)",
                 min_references: int = 2, max_dispersion_m: float = 1.5,
                 max_dispersion_deg: float = 15.0,
                 min_memory_confidence: float = 0.5,
                 max_state_risk: float = 1.5,
                 verifier_config: Path | None = None,
                 preaction_features: bool = False):
        self.data_root = Path(data_root)
        self.query_sequence = str(query_sequence)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "online Reloc3r backend requested CUDA, but torch.cuda.is_available() is false; "
                "refusing to fall back to frozen candidates")
        self.min_references = int(min_references)
        self.max_dispersion_m = float(max_dispersion_m)
        self.max_dispersion_deg = float(max_dispersion_deg)
        self.min_memory_confidence = float(min_memory_confidence)
        self.max_state_risk = float(max_state_risk)
        # Optional Stage 3D evidence.  It is computed after cheap Reloc3r and
        # before any subsequent Geometry decision; its elapsed time is included
        # in the Reloc3r execution receipt.  It is disabled for legacy runs.
        self.preaction_features = bool(preaction_features)
        # This configuration is deliberately source-specific.  It is loaded
        # before evaluation and never fitted from the evaluated sequence.  A
        # calibration utility writes the same schema after selecting values on
        # a disjoint calibration split.
        self.verifier_config = {
            "reloc3r_online": {
                "min_references": self.min_references,
                "max_dispersion_m": self.max_dispersion_m,
                "max_dispersion_deg": self.max_dispersion_deg,
                "max_state_risk": self.max_state_risk,
                "heldout_max_translation_m": 1.5,
                "heldout_max_rotation_deg": 20.0,
            },
            "roma_online_geometry": {
                "min_pnp_inliers": 24,
                "max_reprojection_px": 3.0,
                "max_state_risk": self.max_state_risk,
                "heldout_min_inliers": 24,
                "heldout_min_inlier_ratio": 0.10,
                "heldout_max_epipolar_px": 4.0,
            },
            "memory": {"min_confidence": self.min_memory_confidence},
        }
        if verifier_config is not None:
            loaded = json.loads(Path(verifier_config).read_text(encoding="utf-8"))
            sources = loaded.get("sources", loaded)
            if not isinstance(sources, Mapping):
                raise ValueError("verifier configuration must contain a source mapping")
            for source, values in sources.items():
                if source in self.verifier_config and isinstance(values, Mapping):
                    self.verifier_config[source].update(values)
        _, self.mapping_frames, mapping_times, self.mapping_intrinsics, mapping_root = read_manifest(
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
        self.model = setup_reloc3r_relpose_model(model_name, self.device)
        # Deliberately lazy: a normal Reloc3r-only run does not allocate the
        # stronger RoMa model.  When selected, it is run on the current query
        # and references, never read from AVL-004 cache files.
        self._roma_model = None

    def _roma(self):
        """Return the lazy RoMa model shared by geometry and verification."""
        try:
            from romatch import roma_indoor
        except ImportError as error:  # pragma: no cover - deployment dependency
            raise RuntimeError("RoMa strong-geometry action requires romatch==0.1.2") from error
        if self._roma_model is None:
            self._roma_model = roma_indoor(
                device=self.device, coarse_res=560, upsample_res=864,
                use_custom_corr=False)
            self._roma_model.eval()
        return self._roma_model

    def num_frames(self, sequence: str) -> int:
        if str(sequence) != self.query_sequence:
            raise ValueError("backend was initialized for " + self.query_sequence)
        return len(self.query_frames)

    def frame_observation(self, sequence: str, frame_index: int,
                          state: Mapping[str, Any]) -> Mapping[str, Any]:
        del state
        refs = self.retrieval_indices[frame_index]
        scores = self.retrieval_scores[frame_index]
        groups = []
        for start in range(0, len(refs), 3):
            group_refs = [int(x) for x in refs[start:start + 3]]
            group_scores = [float(x) for x in scores[start:start + 3]]
            if group_refs:
                groups.append({"reference_indices": group_refs,
                               "retrieval_scores": group_scores,
                               "mean_retrieval_score": float(np.mean(group_scores))})
        return {
            "image_path": str(self.query_frames[frame_index]["path"]),
            "timestamp": float(self.query_times[frame_index]),
            "retrieval_references": [int(x) for x in refs],
            "retrieval_scores": [float(x) for x in scores],
            "reference_groups": groups,
            "mapping_size": len(self.mapping_frames),
        }

    @staticmethod
    def _audit() -> Dict[str, bool]:
        return {"query_gt_read": False, "uses_future": False, "selection_uses_gt": False}

    @torch.inference_mode()
    def localize(self, sequence: str, frame_index: int,
                 reference_indices: Sequence[int],
                 state: Mapping[str, Any]) -> Mapping[str, Any]:
        started = time.perf_counter()
        if str(sequence) != self.query_sequence:
            raise ValueError("unexpected sequence")
        refs = tuple(int(x) for x in reference_indices)
        if any(x >= len(self.mapping_frames) for x in refs):
            raise ValueError("reference index outside mapping database")
        if not refs:
            raise ValueError("at least one reference is required")
        query_path = self.query_frames[frame_index]["path"]
        pairs = [(self.mapping_frames[index]["path"], query_path) for index in refs]
        batch = make_batch(pairs, self.mapping_intrinsics, self.device)
        relative = inference_relpose(
            batch, self.model, self.device, use_amp=self.device.type == "cuda")
        relative = relative.detach().cpu().numpy()
        absolute = [self.mapping_pose[index] @ relative[pos]
                    for pos, index in enumerate(refs)]
        pose = absolute[0] if len(absolute) == 1 else Reloc3rVisloc().motion_averaging(
            [self.mapping_pose[index] for index in refs], list(relative))
        disp_m, disp_deg = candidate_dispersion(absolute)
        evidence = {
            "num_references": len(refs),
            "translation_dispersion_m": disp_m,
            "rotation_dispersion_deg": disp_deg,
        }
        if self.preaction_features:
            evidence.update(self._preaction_evidence(query_path, refs, pose, state))
        return {
            "pose": pose.tolist(),
            "source": "reloc3r_online",
            "reference_indices": list(refs),
            "evidence": evidence,
            "execution": {
                "executed": True,
                "cache_hit": False,
                "tool": "Reloc3rRelpose",
                "pair_calls": len(refs),
                "stages": ["pair_preprocessing", "relative_pose_estimation", "pose_aggregation"],
                "query_frame_index": int(frame_index),
                "reference_indices": list(refs),
                "elapsed_s": time.perf_counter() - started,
            },
            "causal": self._audit(),
        }

    def _preaction_evidence(self, query_path: Path, references: Sequence[int],
                             candidate_pose: np.ndarray,
                             state: Mapping[str, Any]) -> Dict[str, Any]:
        """Compute cheap, pre-RoMa correspondence and appearance cues.

        This helper deliberately receives paths rather than poses or Geometry
        outputs.  The metrics are derived from ORB/Hamming matches only and
        are charged to the current cheap localization action by the caller.
        """
        try:
            import cv2
        except ImportError:
            return {"cheap_correspondence": {"available": False, "reason": "opencv_unavailable"},
                    "appearance_change": {"available": False, "reason": "opencv_unavailable"}}
        query = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
        if query is None:
            return {"cheap_correspondence": {"available": False, "reason": "query_unreadable"},
                    "appearance_change": {"available": False, "reason": "query_unreadable"}}
        orb = cv2.ORB_create(nfeatures=500)
        query_keypoints, query_desc = orb.detectAndCompute(query, None)
        if query_desc is None or not query_keypoints:
            return {"cheap_correspondence": {"available": False, "reason": "query_no_descriptors"},
                    "appearance_change": {"available": False, "reason": "query_no_descriptors"}}
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        coordinates, scores, appearance, hist_correlations = [], [], [], []
        height, width = query.shape[:2]
        for reference in references:
            image = cv2.imread(str(self.mapping_frames[int(reference)]["path"]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            keypoints, descriptors = orb.detectAndCompute(image, None)
            if descriptors is None or not keypoints:
                continue
            pairs = matcher.knnMatch(query_desc, descriptors, k=2)
            good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
            coordinates.extend([query_keypoints[m.queryIdx].pt for m in good])
            scores.extend([float(m.distance) for m in good])
            query_mean, image_mean = float(np.mean(query)), float(np.mean(image))
            query_std, image_std = float(np.std(query)), float(np.std(image))
            appearance.append(abs(query_mean - image_mean) / 255.0 + abs(query_std - image_std) / 255.0)
            query_hist = cv2.normalize(cv2.calcHist([query], [0], None, [32], [0, 256]), None).ravel()
            image_hist = cv2.normalize(cv2.calcHist([image], [0], None, [32], [0, 256]), None).ravel()
            hist_correlations.append(float(cv2.compareHist(query_hist, image_hist, cv2.HISTCMP_CORREL)))
        cheap = {"available": True, "valid_matches": int(len(coordinates))}
        if coordinates:
            xy = np.asarray(coordinates, dtype=np.float64)
            # Four-by-four occupancy is a low-cost spatial spread statistic.
            cells = np.clip((xy / np.array([width, height]) * 4).astype(int), 0, 3)
            cell_ids = cells[:, 1] * 4 + cells[:, 0]
            counts = np.bincount(cell_ids, minlength=16).astype(np.float64)
            probabilities = counts / counts.sum()
            nonzero = probabilities[probabilities > 0]
            cheap.update({
                "coverage_ratio": float(np.count_nonzero(counts) / 16.0),
                "spatial_entropy": float(-np.sum(nonzero * np.log(nonzero)) / np.log(16.0)),
                "concentration": float(np.max(probabilities)),
                "median_pixel_radius": float(np.median(np.linalg.norm(xy - np.mean(xy, axis=0), axis=1))),
                "match_distance_median": float(np.median(scores)) if scores else 0.0,
            })
        else:
            cheap.update({"coverage_ratio": 0.0, "spatial_entropy": 0.0,
                          "concentration": 0.0, "median_pixel_radius": 0.0,
                          "match_distance_median": 0.0})
        previous = state.get("accepted_pose")
        temporal = {"available": previous is not None}
        if previous is not None:
            previous = np.asarray(previous, dtype=np.float64)
            temporal.update({
                "translation_disagreement_m": float(np.linalg.norm(candidate_pose[:3, 3] - previous[:3, 3])),
                "rotation_disagreement_deg": float(rotation_error(previous[:3, :3], candidate_pose[:3, :3])),
                "anchor_retrieval_consistency": float(np.exp(-np.linalg.norm(candidate_pose[:3, 3] - previous[:3, 3]))),
            })
        else:
            temporal.update({"translation_disagreement_m": 0.0, "rotation_disagreement_deg": 0.0,
                             "anchor_retrieval_consistency": 0.0})
        return {"cheap_correspondence": cheap,
                "temporal_consistency": temporal,
                "appearance_change": {"available": bool(appearance),
                                       "query_reference_cosine": float(np.mean(hist_correlations)) if hist_correlations else 0.0,
                                       "change_score": float(np.mean(appearance)) if appearance else 0.0,
                                       "brightness_change": float(np.mean(appearance)) if appearance else 0.0}}

    def strong_geometry(self, sequence: str, frame_index: int,
                        reference_indices: Sequence[int],
                        state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run RoMa matching, map-view triangulation, and query PnP online.

        This is the paper-facing strong action.  It intentionally contains no
        access to the historical AVL-004 JSONL outputs: selecting the action
        causes fresh RoMa correspondence inference for the current query and
        the exact requested map images.
        """
        del state
        started = time.perf_counter()
        if str(sequence) != self.query_sequence:
            raise ValueError("unexpected sequence")
        refs = tuple(int(x) for x in reference_indices)
        if len(refs) < 2 or any(x < 0 or x >= len(self.mapping_frames) for x in refs):
            raise ValueError("RoMa geometry requires at least two valid map references")
        from tools.run_avl004_roma_recovery import (
            camera_matrix, intersect_query, roma_pair, solve_pose, triangulate,
        )
        model = self._roma()
        query_path = self.query_frames[frame_index]["path"]
        pair_matches = [roma_pair(model, query_path,
                                  self.mapping_frames[ref]["path"], self.device,
                                  6000) for ref in refs]
        matrices = [camera_matrix(self.mapping_pose[ref], self.mapping_intrinsics)
                    for ref in refs]
        all_xyz, all_qxy, all_score, pair_stats = [], [], [], []
        for left, right in combinations(range(len(refs)), 2):
            qxy, first_xy, second_xy, score = intersect_query(
                pair_matches[left], pair_matches[right], radius=2.0, max_points=1800)
            xyz, valid = triangulate(matrices[left], matrices[right], first_xy, second_xy)
            if len(xyz):
                all_xyz.append(xyz)
                all_qxy.append(qxy[valid])
                all_score.append(score[valid])
            pair_stats.append({"reference_indices": [refs[left], refs[right]],
                               "intersections": int(len(qxy)),
                               "triangulated": int(len(xyz))})
        if all_xyz:
            xyz = np.concatenate(all_xyz)
            qxy = np.concatenate(all_qxy)
            score = np.concatenate(all_score)
            keep = np.argsort(-score, kind="stable")[:3000]
            pose, inliers, reproj = solve_pose(xyz[keep], qxy[keep], self.mapping_intrinsics, None)
            correspondences = int(len(keep))
        else:
            pose, inliers, reproj, correspondences = None, 0, None, 0
        return {
            "pose": None if pose is None else pose.tolist(),
            "source": "roma_online_geometry",
            "reference_indices": list(refs),
            "evidence": {"num_references": len(refs), "inliers": int(inliers),
                         "correspondences": correspondences,
                         "median_reprojection_px": reproj,
                         "pair_stats": pair_stats,
                         "geometry_backend": "RoMa+triangulation+PnP"},
            "execution": {"executed": True, "cache_hit": False,
                          "tool": "RoMa indoor + triangulation + PnP",
                          "pair_calls": len(refs),
                          "stages": ["roma_dense_matching", "map_view_triangulation",
                                     "pnp_pose_estimation"],
                          "query_frame_index": int(frame_index),
                          "reference_indices": list(refs),
                          "elapsed_s": time.perf_counter() - started},
            "causal": self._audit(),
        }

    @staticmethod
    def _skew(vector: np.ndarray) -> np.ndarray:
        return np.array([[0.0, -vector[2], vector[1]],
                         [vector[2], 0.0, -vector[0]],
                         [-vector[1], vector[0], 0.0]], dtype=np.float64)

    def _heldout_reference(self, frame_index: int, used_refs: Sequence[int]) -> int | None:
        used = {int(ref) for ref in used_refs}
        for ref in self.retrieval_indices[frame_index]:
            if int(ref) not in used:
                return int(ref)
        return None

    def _heldout_reloc3r_evidence(self, frame_index: int, candidate_pose: np.ndarray,
                                  used_refs: Sequence[int]) -> Dict[str, Any]:
        """Validate a Reloc3r pose against a fresh, unused map view.

        The held-out pair is executed here rather than borrowed from the pose
        action.  It therefore remains an independent reference observation,
        although it uses the Reloc3r source's natural relative-pose evidence.
        """
        started = time.perf_counter()
        heldout = self._heldout_reference(frame_index, used_refs)
        if heldout is None:
            return {"available": False, "reason": "no_heldout_reference",
                    "elapsed_s": time.perf_counter() - started}
        query_path = self.query_frames[frame_index]["path"]
        batch = make_batch([(self.mapping_frames[heldout]["path"], query_path)],
                           self.mapping_intrinsics, self.device)
        relative = inference_relpose(batch, self.model, self.device,
                                     use_amp=self.device.type == "cuda")
        heldout_pose = self.mapping_pose[heldout] @ relative[0].detach().cpu().numpy()
        translation = float(np.linalg.norm(heldout_pose[:3, 3] - candidate_pose[:3, 3]))
        rotation = float(rotation_error(heldout_pose[:3, :3], candidate_pose[:3, :3]))
        return {"available": True, "heldout_references": [heldout],
                "translation_disagreement_m": translation,
                "rotation_disagreement_deg": rotation,
                "elapsed_s": time.perf_counter() - started,
                "tool": "Reloc3r held-out relative-pose consistency"}

    def _heldout_roma_evidence(self, frame_index: int, candidate_pose: np.ndarray,
                               used_refs: Sequence[int]) -> Dict[str, Any]:
        """Score new RoMa matches on a held-out reference by candidate epipolar error."""
        started = time.perf_counter()
        heldout = self._heldout_reference(frame_index, used_refs)
        if heldout is None:
            return {"available": False, "reason": "no_heldout_reference",
                    "elapsed_s": time.perf_counter() - started}
        from tools.run_avl004_roma_recovery import roma_pair
        qxy, rxy, score = roma_pair(self._roma(), self.query_frames[frame_index]["path"],
                                    self.mapping_frames[heldout]["path"], self.device, 6000)
        if not len(qxy):
            return {"available": False, "heldout_references": [heldout], "matches": 0,
                    "elapsed_s": time.perf_counter() - started}
        # Map and candidate poses are camera-to-world.  Form the held-out-map
        # to query essential matrix and score a symmetric pixel-line error.
        map_pose = self.mapping_pose[heldout]
        R_query = candidate_pose[:3, :3].T
        R_map = map_pose[:3, :3].T
        R_rel = R_query @ R_map.T
        translation = R_query @ (map_pose[:3, 3] - candidate_pose[:3, 3])
        if float(np.linalg.norm(translation)) < 1e-8:
            return {"available": False, "heldout_references": [heldout], "matches": int(len(qxy)),
                    "reason": "degenerate_relative_translation", "elapsed_s": time.perf_counter() - started}
        inv_k = np.linalg.inv(self.mapping_intrinsics)
        fundamental = inv_k.T @ (self._skew(translation) @ R_rel) @ inv_k
        ref_h = np.column_stack([rxy.astype(np.float64), np.ones(len(rxy))])
        query_h = np.column_stack([qxy.astype(np.float64), np.ones(len(qxy))])
        query_lines = (fundamental @ ref_h.T).T
        ref_lines = (fundamental.T @ query_h.T).T
        d_query = np.abs(np.sum(query_h * query_lines, axis=1)) / np.maximum(
            np.linalg.norm(query_lines[:, :2], axis=1), 1e-8)
        d_ref = np.abs(np.sum(ref_h * ref_lines, axis=1)) / np.maximum(
            np.linalg.norm(ref_lines[:, :2], axis=1), 1e-8)
        errors = 0.5 * (d_query + d_ref)
        inlier = errors <= 4.0
        # RoMa's confidence is reported, but acceptance is based on geometry
        # alone so confidence cannot conceal a poor held-out pose fit.
        return {"available": True, "heldout_references": [heldout],
                "matches": int(len(errors)), "inliers": int(np.count_nonzero(inlier)),
                "inlier_ratio": float(np.mean(inlier)),
                "median_epipolar_px": float(np.median(errors)),
                "mean_match_score": float(np.mean(score)) if len(score) else None,
                "elapsed_s": time.perf_counter() - started,
                "tool": "RoMa held-out epipolar verification"}

    def retrieve_memory(self, sequence: str, frame_index: int,
                        memory: Sequence[Mapping[str, Any]],
                        state: Mapping[str, Any]) -> Mapping[str, Any]:
        del state
        started = time.perf_counter()
        if str(sequence) != self.query_sequence:
            raise ValueError("unexpected sequence")
        if not memory:
            return {"pose": None, "source": "memory", "evidence": {"available": False},
                    "execution": {"executed": True, "cache_hit": False,
                                  "tool": "memory_availability_check",
                                  "stages": ["memory_availability_check"],
                                  "query_frame_index": int(frame_index), "reference_indices": [],
                                  "elapsed_s": time.perf_counter() - started},
                    "causal": self._audit()}
        # Memory is sorted by insertion time by the environment.  The frame
        # guard is repeated here so a backend cannot accidentally use a future
        # record if called outside the environment.
        usable = [item for item in memory if int(item["frame_index"]) < frame_index]
        if not usable:
            return {"pose": None, "source": "memory", "evidence": {"available": False},
                    "execution": {"executed": True, "cache_hit": False,
                                  "tool": "memory_availability_check",
                                  "stages": ["memory_availability_check"],
                                  "query_frame_index": int(frame_index), "reference_indices": [],
                                  "elapsed_s": time.perf_counter() - started},
                    "causal": self._audit()}
        item = usable[-1]
        return {
            "pose": item["pose"], "source": "memory",
            "evidence": {"available": True, "memory_id": int(item["memory_id"]),
                         "confidence": float(item.get("confidence", 0.0))},
            "execution": {"executed": True, "cache_hit": False,
                          "tool": "memory_proposal", "stages": ["memory_lookup"],
                          "query_frame_index": int(frame_index), "reference_indices": [],
                          "elapsed_s": time.perf_counter() - started},
            "causal": self._audit(),
        }

    def verify(self, sequence: str, frame_index: int, candidate: Candidate,
               state: Mapping[str, Any]) -> Mapping[str, Any]:
        if str(sequence) != self.query_sequence:
            raise ValueError("unexpected sequence")
        started = time.perf_counter()
        if candidate.pose is None:
            return {"accepted": False, "evidence": {"reason": "missing_pose", "elapsed_s": time.perf_counter() - started},
                    "causal": self._audit()}
        if candidate.source == "memory":
            confidence = float(candidate.evidence.get("confidence", 0.0))
            thresholds = self.verifier_config["memory"]
            accepted = confidence >= float(thresholds["min_confidence"])
            return {"accepted": accepted,
                    "evidence": {"confidence": confidence, "memory_check": True,
                                 "elapsed_s": time.perf_counter() - started},
                    "causal": self._audit()}
        if candidate.source == "roma_online_geometry":
            thresholds = self.verifier_config["roma_online_geometry"]
            inliers = int(candidate.evidence.get("inliers", 0))
            reproj = candidate.evidence.get("median_reprojection_px")
            raw_state_risk = previous_state_risk(candidate.pose, state.get("accepted_pose"))
            trust = state_trust(int(state.get("anchor_age", 0)),
                                int(state.get("reject_streak", 0)))
            pnp_risk = max(
                max(0.0, float(thresholds["min_pnp_inliers"]) - inliers) /
                max(float(thresholds["min_pnp_inliers"]), 1.0),
                0.0 if reproj is None else float(reproj) / float(thresholds["max_reprojection_px"]))
            independent = self._heldout_roma_evidence(frame_index, candidate.pose,
                                                       candidate.reference_indices)
            heldout_inliers = int(independent.get("inliers", 0))
            heldout_ratio = float(independent.get("inlier_ratio", 0.0))
            heldout_median = independent.get("median_epipolar_px")
            independent_risk = max(
                max(0.0, float(thresholds["heldout_min_inliers"]) - heldout_inliers) /
                max(float(thresholds["heldout_min_inliers"]), 1.0),
                max(0.0, float(thresholds["heldout_min_inlier_ratio"]) - heldout_ratio) /
                max(float(thresholds["heldout_min_inlier_ratio"]), 1e-6),
                1.0 if heldout_median is None else float(heldout_median) /
                float(thresholds["heldout_max_epipolar_px"]),
            )
            total_risk = max(pnp_risk, independent_risk, trust * raw_state_risk)
            accepted = (
                inliers >= int(thresholds["min_pnp_inliers"]) and reproj is not None and
                float(reproj) <= float(thresholds["max_reprojection_px"]) and
                bool(independent.get("available", False)) and
                heldout_inliers >= int(thresholds["heldout_min_inliers"]) and
                heldout_ratio >= float(thresholds["heldout_min_inlier_ratio"]) and
                heldout_median is not None and
                float(heldout_median) <= float(thresholds["heldout_max_epipolar_px"]) and
                total_risk <= float(thresholds["max_state_risk"]))
            return {"accepted": accepted, "evidence": {
                "inliers": inliers, "median_reprojection_px": reproj,
                "pnp_risk": pnp_risk, "previous_state_risk": raw_state_risk,
                "state_trust": trust, "total_risk": total_risk,
                "independent_verification": independent,
                "independent_risk": independent_risk,
                "verifier_source": "roma_online_geometry",
                "confidence": float(np.exp(-total_risk)),
                "elapsed_s": time.perf_counter() - started,
            }, "causal": self._audit()}
        if candidate.source != "reloc3r_online":
            return {"accepted": False, "evidence": {"reason": "unsupported_candidate_source",
                    "source": candidate.source, "elapsed_s": time.perf_counter() - started},
                    "causal": self._audit()}
        thresholds = self.verifier_config["reloc3r_online"]
        num_refs = int(candidate.evidence.get("num_references", 0))
        disp_m = float(candidate.evidence.get("translation_dispersion_m", float("inf")))
        disp_deg = float(candidate.evidence.get("rotation_dispersion_deg", float("inf")))
        previous = state.get("accepted_pose")
        anchor_age = int(state.get("anchor_age", 0))
        reject_streak = int(state.get("reject_streak", 0))
        trust = state_trust(anchor_age, reject_streak)
        raw_state_risk = previous_state_risk(candidate.pose, previous)
        relaxed_state_risk = trust * raw_state_risk
        dispersion_risk = max(disp_m / max(float(thresholds["max_dispersion_m"]), 1e-6),
                              disp_deg / max(float(thresholds["max_dispersion_deg"]), 1e-6))
        independent = self._heldout_reloc3r_evidence(frame_index, candidate.pose,
                                                      candidate.reference_indices)
        heldout_translation = float(independent.get("translation_disagreement_m", float("inf")))
        heldout_rotation = float(independent.get("rotation_disagreement_deg", float("inf")))
        independent_risk = max(
            heldout_translation / max(float(thresholds["heldout_max_translation_m"]), 1e-6),
            heldout_rotation / max(float(thresholds["heldout_max_rotation_deg"]), 1e-6))
        total_risk = max(dispersion_risk, relaxed_state_risk, independent_risk)
        accepted = (num_refs >= int(thresholds["min_references"]) and
                    disp_m <= float(thresholds["max_dispersion_m"]) and
                    disp_deg <= float(thresholds["max_dispersion_deg"]) and
                    bool(independent.get("available", False)) and
                    heldout_translation <= float(thresholds["heldout_max_translation_m"]) and
                    heldout_rotation <= float(thresholds["heldout_max_rotation_deg"]) and
                    total_risk <= float(thresholds["max_state_risk"]))
        return {"accepted": accepted, "evidence": {
            "num_references": num_refs,
            "translation_dispersion_m": disp_m,
            "rotation_dispersion_deg": disp_deg,
            "dispersion_risk": dispersion_risk,
            "previous_state_risk": raw_state_risk,
            "state_trust": trust,
            "relaxed_state_risk": relaxed_state_risk,
            "independent_verification": independent,
            "independent_risk": independent_risk,
            "total_risk": total_risk,
            "anchor_age": anchor_age,
            "reject_streak": reject_streak,
            "verifier_source": "reloc3r_online",
            "confidence": float(np.exp(-total_risk)),
            "elapsed_s": time.perf_counter() - started,
        }, "causal": self._audit()}


__all__ = ["LILocBenchReloc3rBackend", "previous_state_risk", "state_trust"]
