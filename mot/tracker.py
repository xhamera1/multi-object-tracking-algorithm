from __future__ import annotations

from .association import match
from enum import Enum
from dataclasses import dataclass, field
from .kalman import KalmanFilter
from .types import BBox


@dataclass
class Detection:
    frame: int
    bbox: BBox
    confidence: float


class TrackState(Enum):
    Tracked = 0
    Lost = 1
    Removed = 2


@dataclass(frozen=True)
class TrackerConfig:
    confidence_threshold_high: float
    confidence_threshold_low: float
    new_track_threshold: float
    first_match_cost_max: float
    second_match_cost_max: float
    max_age: int
    min_hits: int


@dataclass
class Track:
    id: int
    bbox: BBox
    last_frame: int
    state: TrackState = TrackState.Tracked
    hits: int = 1
    kalman_filter: KalmanFilter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.kalman_filter = KalmanFilter(self.bbox)

    def predict(self) -> None:
        self.bbox = self.kalman_filter.predict()

    def update(self, det: Detection) -> None:
        self.bbox = self.kalman_filter.update(det.bbox)
        self.state = TrackState.Tracked
        self.hits += 1
        self.last_frame = det.frame

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


@dataclass
class TrackResult:
    frame: int
    track_id: int
    bbox: BBox


class Tracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.tracked_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.next_track_id = 1
        self.det_conf_threshold = config.confidence_threshold_high
        self.det_low_conf_threshold = config.confidence_threshold_low
        self.new_track_threshold = config.new_track_threshold
        self.max_cost_first = config.first_match_cost_max
        self.max_cost_second = config.second_match_cost_max
        self.max_age = config.max_age
        self.min_hits = config.min_hits

    def step(self, detections: list[Detection], frame: int) -> list[TrackResult]:
        # partition detections by confidence using two thresholds
        high_dets = [
            det for det in detections if det.confidence >= self.det_conf_threshold
        ]
        low_dets = [
            det
            for det in detections
            if self.det_low_conf_threshold <= det.confidence < self.det_conf_threshold
        ]

        # predict active and lost track positions before association
        track_pool = self.tracked_tracks + self.lost_tracks
        for track in track_pool:
            track.predict()

        # first association between active/lost tracks and high-confidence detections
        matches, unmatched_tracks, unmatched_high_dets = match(
            track_pool, high_dets, self.max_cost_first
        )

        activated_tracks: list[Track] = []
        lost_tracks: list[Track] = []

        # matched tracks are updated; lost tracks are reactivated
        for track, det in matches:
            track.update(det)
            activated_tracks.append(track)

        unmatched_active_tracks = [
            track for track in unmatched_tracks if track.state == TrackState.Tracked
        ]
        unmatched_lost_tracks = [
            track for track in unmatched_tracks if track.state == TrackState.Lost
        ]

        # second association between unmatched active tracks and low-confidence detections
        matches, unmatched_active_tracks, _ = match(
            unmatched_active_tracks, low_dets, self.max_cost_second
        )

        # update recovered tracks
        for track, det in matches:
            track.update(det)
            activated_tracks.append(track)

        # new track creation from unmatched high-confidence detections
        for det in unmatched_high_dets:
            if det.confidence < self.new_track_threshold:
                continue
            activated_tracks.append(
                Track(
                    id=self.next_track_id,
                    bbox=det.bbox,
                    last_frame=frame,
                )
            )
            self.next_track_id += 1

        # unmatched active tracks become lost or removed after the configured buffer
        for track in unmatched_active_tracks:
            if frame - track.last_frame > self.max_age:
                track.mark_removed()
            else:
                track.mark_lost()
                lost_tracks.append(track)

        # previously lost tracks keep aging until they expire
        for track in unmatched_lost_tracks:
            if frame - track.last_frame > self.max_age:
                track.mark_removed()
            else:
                lost_tracks.append(track)

        self.tracked_tracks = activated_tracks
        self.lost_tracks = lost_tracks

        return [
            TrackResult(frame=frame, track_id=track.id, bbox=track.bbox)
            for track in self.tracked_tracks
            if track.hits >= self.min_hits
        ]
