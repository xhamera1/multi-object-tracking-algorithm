from __future__ import annotations

import numpy as np

from .types import BBox


def box_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def box_iou_matrix(
    track_boxes: list[BBox],
    detection_boxes: list[BBox],
) -> np.ndarray:
    """Pairwise IoU, shape `(n_tracks, n_dets)`; empty side yields zero rows/cols."""
    n_t, n_d = len(track_boxes), len(detection_boxes)
    if n_t == 0 or n_d == 0:
        return np.zeros((n_t, n_d), dtype=float)

    t = np.asarray(track_boxes, dtype=float)
    d = np.asarray(detection_boxes, dtype=float)

    tx1, ty1, tw, th = t[:, 0:1], t[:, 1:2], t[:, 2:3], t[:, 3:4]
    dx1, dy1, dw, dh = d[:, 0], d[:, 1], d[:, 2], d[:, 3]

    tx2 = tx1 + tw
    ty2 = ty1 + th
    dx2 = dx1 + dw
    dy2 = dy1 + dh

    inter_x1 = np.maximum(tx1, dx1)
    inter_y1 = np.maximum(ty1, dy1)
    inter_x2 = np.minimum(tx2, dx2)
    inter_y2 = np.minimum(ty2, dy2)

    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter_area = inter_w * inter_h

    area_t = tw * th
    area_d = dw * dh
    union = area_t + area_d - inter_area
    return np.where(union > 0, inter_area / union, 0.0)
