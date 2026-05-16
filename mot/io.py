from pathlib import Path
from typing import Iterable

from .types import BBox
from .tracker import Detection, TrackResult


def load_detections(det_path: Path) -> list[Detection]:
    detections: list[Detection] = []
    with det_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            frame, _, x, y, w, h, conf = line.split(",")
            detections.append(
                Detection(
                    frame=int(frame),
                    bbox=BBox(
                        x=float(x),
                        y=float(y),
                        w=float(w),
                        h=float(h),
                    ),
                    confidence=float(conf),
                )
            )
    return detections


def save_mot_results(path: Path, rows: Iterable[TrackResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            bbox = row.bbox
            handle.write(
                f"{row.frame},{row.track_id},{bbox.x:.2f},{bbox.y:.2f},{bbox.w:.2f},{bbox.h:.2f},1,-1,-1,-1\n"
            )
