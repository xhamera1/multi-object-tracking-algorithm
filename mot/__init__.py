"""Tracker core: IoU association, Kalman motion, MOT I/O."""

from .io import load_detections, save_mot_results
from .tracker import Tracker, TrackerConfig
from .types import BBox
from .tracker import Detection, TrackResult

__all__ = [
    "BBox",
    "Detection",
    "TrackResult",
    "Tracker",
    "TrackerConfig",
    "load_detections",
    "save_mot_results",
]
