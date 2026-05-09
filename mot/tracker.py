from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .association import hungarian_match
from .io import Detection, TrackResult
from .kalman import SimpleMotionModel


BBox = Tuple[float, float, float, float]


@dataclass
class Track:
    track_id: int
    bbox: BBox
    age: int = 1
    time_since_update: int = 0
    hits: int = 1

    def __post_init__(self) -> None:
        self.motion = SimpleMotionModel(self.bbox)

    def predict(self) -> None:
        self.bbox = self.motion.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: BBox) -> None:
        self.bbox = bbox
        self.motion.update(bbox)
        self.time_since_update = 0
        self.hits += 1

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3


class MultiObjectTracker:
    def __init__(
        self,
        det_conf_threshold: float = 0.3,
        iou_match_threshold: float = 0.3,
        max_center_distance: float = 1.5,
        max_match_cost: float = 0.9,
        weight_iou: float = 0.7,
        weight_center_distance: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        next_track_id_start: int = 1,
    ) -> None:
        self.det_conf_threshold = det_conf_threshold
        self.iou_match_threshold = iou_match_threshold
        self.max_center_distance = max_center_distance
        self.max_match_cost = max_match_cost
        self.weight_iou = weight_iou
        self.weight_center_distance = weight_center_distance
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_track_id = next_track_id_start
        self.tracks: list[Track] = []

    def _new_track(self, bbox: BBox) -> Track:
        track = Track(track_id=self.next_track_id, bbox=bbox)
        self.next_track_id += 1
        return track

    def step(self, frame_detections: Sequence[Detection]) -> None:
        dets = [d for d in frame_detections if d.confidence >= self.det_conf_threshold]
        det_boxes: list[BBox] = [(d.x, d.y, d.w, d.h) for d in dets]

        for track in self.tracks:
            track.predict()

        track_boxes = [t.bbox for t in self.tracks]
        match_result = hungarian_match(
            track_boxes,
            det_boxes,
            iou_threshold=self.iou_match_threshold,
            max_center_distance=self.max_center_distance,
            max_cost=self.max_match_cost,
            weight_iou=self.weight_iou,
            weight_center_distance=self.weight_center_distance,
        )

        for track_idx, det_idx in match_result.matches:
            self.tracks[track_idx].update(det_boxes[det_idx])

        for det_idx in match_result.unmatched_detections:
            self.tracks.append(self._new_track(det_boxes[det_idx]))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

    def collect_frame_results(self, frame: int, confirmed_only: bool = True) -> list[TrackResult]:
        rows: list[TrackResult] = []
        for track in self.tracks:
            if track.time_since_update > 0:
                continue
            if confirmed_only and track.hits < self.min_hits:
                continue
            x, y, w, h = track.bbox
            rows.append(
                TrackResult(frame=frame, track_id=track.track_id, x=x, y=y, w=w, h=h)
            )
        return rows
