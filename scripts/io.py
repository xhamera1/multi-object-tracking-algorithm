from pathlib import Path
from typing import Iterable

from scripts.types import Detection, TrackResult


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
                    x=float(x),
                    y=float(y),
                    w=float(w),
                    h=float(h),
                    confidence=float(conf),
                )
            )
    return detections


def save_mot_results(path: Path, rows: Iterable[TrackResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(
                f"{r.frame},{r.track_id},{r.x:.2f},{r.y:.2f},{r.w:.2f},{r.h:.2f},1,-1,-1,-1\n"
            )
