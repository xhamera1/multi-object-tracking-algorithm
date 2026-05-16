from typing import Tuple
from dataclasses import dataclass

BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    frame: int
    x: float
    y: float
    w: float
    h: float
    confidence: float


@dataclass
class TrackResult:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
