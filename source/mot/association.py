from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import BBox, bbox_iou


@dataclass
class MatchResult:
    matches: list[tuple[int, int]]
    unmatched_tracks: list[int]
    unmatched_detections: list[int]


def _bbox_center(box: BBox) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _normalized_center_distance(track_box: BBox, det_box: BBox) -> float:
    tx, ty = _bbox_center(track_box)
    dx, dy = _bbox_center(det_box)
    dist = float(np.hypot(tx - dx, ty - dy))

    tdiag = float(np.hypot(track_box[2], track_box[3]))
    ddiag = float(np.hypot(det_box[2], det_box[3]))
    scale = max(1.0, 0.5 * (tdiag + ddiag))
    return dist / scale


def build_combined_cost_matrix(
    track_boxes: Sequence[BBox],
    det_boxes: Sequence[BBox],
    weight_iou: float,
    weight_center_distance: float,
) -> np.ndarray:
    if not track_boxes or not det_boxes:
        return np.empty((len(track_boxes), len(det_boxes)))
    matrix = np.zeros((len(track_boxes), len(det_boxes)), dtype=float)
    for i, tbox in enumerate(track_boxes):
        for j, dbox in enumerate(det_boxes):
            iou_cost = 1.0 - bbox_iou(tbox, dbox)
            center_cost = min(1.0, _normalized_center_distance(tbox, dbox))
            matrix[i, j] = weight_iou * iou_cost + weight_center_distance * center_cost
    return matrix


def hungarian_match(
    track_boxes: Sequence[BBox],
    det_boxes: Sequence[BBox],
    iou_threshold: float = 0.3,
    max_center_distance: float = 1.5,
    max_cost: float = 0.9,
    weight_iou: float = 0.7,
    weight_center_distance: float = 0.3,
) -> MatchResult:
    if not track_boxes:
        return MatchResult(matches=[], unmatched_tracks=[], unmatched_detections=list(range(len(det_boxes))))
    if not det_boxes:
        return MatchResult(matches=[], unmatched_tracks=list(range(len(track_boxes))), unmatched_detections=[])

    cost = build_combined_cost_matrix(track_boxes, det_boxes, weight_iou, weight_center_distance)
    row_ind, col_ind = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    unmatched_tracks = set(range(len(track_boxes)))
    unmatched_dets = set(range(len(det_boxes)))

    for ti, di in zip(row_ind, col_ind):
        iou = bbox_iou(track_boxes[ti], det_boxes[di])
        center_distance = _normalized_center_distance(track_boxes[ti], det_boxes[di])
        pair_cost = cost[ti, di]

        # Gating: reject implausible associations before they affect track identities.
        if iou < iou_threshold:
            continue
        if center_distance > max_center_distance:
            continue
        if pair_cost > max_cost:
            continue
        matches.append((ti, di))
        unmatched_tracks.discard(ti)
        unmatched_dets.discard(di)

    return MatchResult(
        matches=matches,
        unmatched_tracks=sorted(unmatched_tracks),
        unmatched_detections=sorted(unmatched_dets),
    )
