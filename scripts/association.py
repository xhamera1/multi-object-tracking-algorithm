from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .iou import IOU, BBox


@dataclass
class MatchResult:
    matches: list[tuple[int, int]]
    unmatched_tracks: list[int]
    unmatched_detections: list[int]


def build_iou_cost_matrix(
    track_boxes: Sequence[BBox],
    det_boxes: Sequence[BBox],
) -> np.ndarray:
    if not track_boxes or not det_boxes:
        return np.empty((len(track_boxes), len(det_boxes)))
    matrix = np.zeros((len(track_boxes), len(det_boxes)), dtype=float)
    for i, tbox in enumerate(track_boxes):
        for j, dbox in enumerate(det_boxes):
            matrix[i, j] = 1.0 - IOU(tbox, dbox)
    return matrix


def hungarian_match(
    track_boxes: Sequence[BBox],
    det_boxes: Sequence[BBox],
    iou_threshold: float = 0.3,
    max_cost: float = 0.9,
) -> MatchResult:
    if not track_boxes:
        return MatchResult(
            matches=[],
            unmatched_tracks=[],
            unmatched_detections=list(range(len(det_boxes))),
        )
    if not det_boxes:
        return MatchResult(
            matches=[],
            unmatched_tracks=list(range(len(track_boxes))),
            unmatched_detections=[],
        )

    cost = build_iou_cost_matrix(track_boxes, det_boxes)
    row_ind, col_ind = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    unmatched_tracks = set(range(len(track_boxes)))
    unmatched_dets = set(range(len(det_boxes)))

    for ti, di in zip(row_ind, col_ind):
        iou = IOU(track_boxes[ti], det_boxes[di])
        if iou < iou_threshold:
            continue
        if cost[ti, di] > max_cost:
            continue
        matches.append((ti, di))
        unmatched_tracks.discard(ti)
        unmatched_dets.discard(di)

    return MatchResult(
        matches=matches,
        unmatched_tracks=sorted(unmatched_tracks),
        unmatched_detections=sorted(unmatched_dets),
    )
