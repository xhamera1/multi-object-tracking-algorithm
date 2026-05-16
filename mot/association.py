from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .iou import box_iou_matrix
from .types import BBox


@dataclass
class AssociationResult:
    matched_track_to_detection: list[tuple[int, int]]
    unmatched_track_indices: list[int]
    unmatched_detection_indices: list[int]


def iou_cost_matrix(
    track_boxes: list[BBox],
    detection_boxes: list[BBox],
) -> np.ndarray:
    """Returns `1 - IoU` with shape `(len(track_boxes), len(detection_boxes))`."""
    t_list = list(track_boxes)
    d_list = list(detection_boxes)
    if not t_list or not d_list:
        return np.empty((len(t_list), len(d_list)))
    return 1.0 - box_iou_matrix(t_list, d_list)


def linear_assignment_iou_match(
    track_boxes: list[BBox],
    detection_boxes: list[BBox],
    iou_threshold: float = 0.3,
    max_cost: float = 0.9,
) -> AssociationResult:
    if not track_boxes:
        return AssociationResult(
            matched_track_to_detection=[],
            unmatched_track_indices=[],
            unmatched_detection_indices=list(range(len(detection_boxes))),
        )
    if not detection_boxes:
        return AssociationResult(
            matched_track_to_detection=[],
            unmatched_track_indices=list(range(len(track_boxes))),
            unmatched_detection_indices=[],
        )

    cost = iou_cost_matrix(track_boxes, detection_boxes)
    row_ind, col_ind = linear_sum_assignment(cost)

    matched: list[tuple[int, int]] = []
    unmatched_tracks = set(range(len(track_boxes)))
    unmatched_dets = set(range(len(detection_boxes)))

    for ti, di in zip(row_ind, col_ind, strict=True):
        c = float(cost[ti, di])
        if c > 1.0 - iou_threshold:
            continue
        if c > max_cost:
            continue
        matched.append((ti, di))
        unmatched_tracks.discard(ti)
        unmatched_dets.discard(di)

    return AssociationResult(
        matched_track_to_detection=matched,
        unmatched_track_indices=sorted(unmatched_tracks),
        unmatched_detection_indices=sorted(unmatched_dets),
    )
