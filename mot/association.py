from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from .iou import box_iou_matrix
from .types import BBox

if TYPE_CHECKING:
    from .tracker import Detection, Track


def iou_cost_matrix(
    track_boxes: list[BBox],
    detection_boxes: list[BBox],
) -> np.ndarray:
    if not track_boxes or not detection_boxes:
        return np.empty((len(track_boxes), len(detection_boxes)))

    return 1.0 - box_iou_matrix(track_boxes, detection_boxes)


def match(
    tracks: list[Track],
    detections: list[Detection],
    max_cost: float,
) -> tuple[
    list[tuple[Track, Detection]],
    list[Track],
    list[Detection],
]:
    if not tracks:
        return [], [], detections.copy()
    if not detections:
        return [], tracks.copy(), []

    C = iou_cost_matrix(
        [t.bbox for t in tracks],
        [d.bbox for d in detections],
    )

    row_ind, col_ind = linear_sum_assignment(C)

    matched = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(detections)))

    for track_id, det_id in zip(row_ind, col_ind, strict=True):

        # reject matches with high cost
        if C[track_id, det_id] > max_cost:
            continue

        matched.append((tracks[track_id], detections[det_id]))
        unmatched_tracks.remove(track_id)
        unmatched_dets.remove(det_id)

    return (
        matched,
        [tracks[id] for id in sorted(unmatched_tracks)],
        [detections[id] for id in sorted(unmatched_dets)],
    )
