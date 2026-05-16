from __future__ import annotations

from dataclasses import dataclass

from .association import AssociationResult, linear_assignment_iou_match
from .kalman import KalmanBBoxFilter
from .types import BBox, Detection, TrackResult


@dataclass(frozen=True)
class TrackerConfig:
    det_conf_threshold: float = 0.3
    det_low_conf_threshold: float = 0.1
    iou_match_threshold: float = 0.3
    iou_match_threshold_low: float = 0.5
    max_match_cost: float = 0.9
    max_age: int = 30
    next_track_id_start: int = 1


@dataclass
class Track:
    track_id: int
    bbox: BBox
    age: int = 1
    time_since_update: int = 0

    def __post_init__(self) -> None:
        self.motion = KalmanBBoxFilter(self.bbox)

    def predict(self) -> None:
        self.bbox = self.motion.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: BBox) -> None:
        self.bbox = bbox
        self.motion.update(bbox)
        self.time_since_update = 0


class Tracker:
    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self.next_track_id = config.next_track_id_start
        self.tracks: list[Track] = []

    def _new_track(self, bbox: BBox) -> Track:
        track = Track(track_id=self.next_track_id, bbox=bbox)
        self.next_track_id += 1
        return track

    def _partition_detections_by_confidence(
        self, frame_detections: list[Detection]
    ) -> tuple[list[Detection], list[BBox], list[BBox]]:
        high = [
            d
            for d in frame_detections
            if d.confidence >= self._config.det_conf_threshold
        ]
        low = [
            d
            for d in frame_detections
            if self._config.det_low_conf_threshold
            <= d.confidence
            < self._config.det_conf_threshold
        ]
        high_boxes = [(d.x, d.y, d.w, d.h) for d in high]
        low_boxes = [(d.x, d.y, d.w, d.h) for d in low]
        return high, high_boxes, low_boxes

    def _predict_all_track_boxes(self) -> None:
        for track in self.tracks:
            track.predict()

    def _associate_high_confidence(
        self, predicted_track_boxes: list[BBox], high_score_boxes: list[BBox]
    ) -> AssociationResult:
        return linear_assignment_iou_match(
            predicted_track_boxes,
            high_score_boxes,
            iou_threshold=self._config.iou_match_threshold,
            max_cost=self._config.max_match_cost,
        )

    def _associate_low_confidence(
        self,
        unmatched_track_indices: list[int],
        low_score_boxes: list[BBox],
    ) -> AssociationResult:
        track_boxes = [self.tracks[i].bbox for i in unmatched_track_indices]
        return linear_assignment_iou_match(
            track_boxes,
            low_score_boxes,
            iou_threshold=self._config.iou_match_threshold_low,
            max_cost=1.0,
        )

    def _spawn_tracks_from_unmatched_high(
        self,
        unmatched_high_detection_indices: list[int],
        high_score_boxes: list[BBox],
    ) -> None:
        for det_idx in unmatched_high_detection_indices:
            self.tracks.append(self._new_track(high_score_boxes[det_idx]))

    def _remove_tracks_exceeding_max_age(self) -> None:
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self._config.max_age
        ]

    def step(self, frame_detections: list[Detection]) -> None:
        _, high_score_boxes, low_score_boxes = (
            self._partition_detections_by_confidence(frame_detections)
        )

        self._predict_all_track_boxes()
        predicted_track_boxes = [t.bbox for t in self.tracks]

        high_assoc = self._associate_high_confidence(
            predicted_track_boxes, high_score_boxes
        )

        for track_idx, det_idx in high_assoc.matched_track_to_detection:
            self.tracks[track_idx].update(high_score_boxes[det_idx])

        low_assoc = self._associate_low_confidence(
            high_assoc.unmatched_track_indices, low_score_boxes
        )

        for local_track_idx, det_idx in low_assoc.matched_track_to_detection:
            track_idx = high_assoc.unmatched_track_indices[local_track_idx]
            self.tracks[track_idx].update(low_score_boxes[det_idx])

        self._spawn_tracks_from_unmatched_high(
            high_assoc.unmatched_detection_indices, high_score_boxes
        )
        self._remove_tracks_exceeding_max_age()

    def collect_frame_results(self, frame: int) -> list[TrackResult]:
        rows: list[TrackResult] = []
        for track in self.tracks:
            if track.time_since_update > 0:
                continue
            x, y, w, h = track.bbox
            rows.append(
                TrackResult(frame=frame, track_id=track.track_id, x=x, y=y, w=w, h=h)
            )
        return rows
