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
        det_low_conf_threshold: float = 0.1,
        iou_match_threshold: float = 0.3,
        iou_match_threshold_low: float = 0.5,
        max_match_cost: float = 0.9,
        max_age: int = 30,
        min_hits: int = 3,
        next_track_id_start: int = 1,
    ) -> None:
        self.det_conf_threshold = det_conf_threshold
        self.det_low_conf_threshold = det_low_conf_threshold
        self.iou_match_threshold = iou_match_threshold
        self.iou_match_threshold_low = iou_match_threshold_low
        self.max_match_cost = max_match_cost
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_track_id = next_track_id_start
        self.tracks: list[Track] = []

    def _new_track(self, bbox: BBox) -> Track:
        track = Track(track_id=self.next_track_id, bbox=bbox)
        self.next_track_id += 1
        return track

    def step(self, frame_detections: Sequence[Detection]) -> None:
        high_score_dets = [d for d in frame_detections if d.confidence >= self.det_conf_threshold]
        low_score_dets = [d for d in frame_detections if self.det_low_conf_threshold <= d.confidence < self.det_conf_threshold]

        high_score_boxes: list[BBox] = [(d.x, d.y, d.w, d.h) for d in high_score_dets]
        low_score_boxes: list[BBox] = [(d.x, d.y, d.w, d.h) for d in low_score_dets]

        for track in self.tracks:
            track.predict()

        track_boxes = [t.bbox for t in self.tracks]
        
        # First Stage
        match_result = hungarian_match(
            track_boxes,
            high_score_boxes,
            iou_threshold=self.iou_match_threshold,
            max_cost=self.max_match_cost,
        )

        for track_idx, det_idx in match_result.matches:
            self.tracks[track_idx].update(high_score_boxes[det_idx])

        # Second Stage
        # For unmatched tracks, try to match with low score detections using only IoU
        unmatched_track_indices = [
            i for i in match_result.unmatched_tracks 
            if self.tracks[i].is_confirmed
        ]
        unmatched_track_boxes = [self.tracks[i].bbox for i in unmatched_track_indices]
        
        match_result_low = hungarian_match(
            unmatched_track_boxes,
            low_score_boxes,
            iou_threshold=self.iou_match_threshold_low,
            max_cost=1.0,
        )
        
        for local_track_idx, det_idx in match_result_low.matches:
            track_idx = unmatched_track_indices[local_track_idx]
            self.tracks[track_idx].update(low_score_boxes[det_idx])

        # New tracks from unmatched high score dets
        for det_idx in match_result.unmatched_detections:
            self.tracks.append(self._new_track(high_score_boxes[det_idx]))

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
