from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts.constants import (
    TRAIN_DATA_PATH,
    CONFIG,
    TRAIN_PREDICTIONS_PATH,
)

from mot import Tracker, TrackerConfig, load_detections, save_mot_results, sort_results


def run_train_dataset(
    data_root: Path,
    output_dir: Path,
    tracker_cfg: Mapping[str, Any],
    verbose: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for seq_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        det_file = seq_dir / "det" / "det.txt"
        if not det_file.exists():
            continue

        detections = load_detections(det_file)
        tracker = Tracker(TrackerConfig(**dict(tracker_cfg)))
        results: list = []

        by_frame: dict[int, list] = {}
        for d in detections:
            by_frame.setdefault(d.frame, []).append(d)

        if not by_frame:
            if verbose:
                print(f"[train] skipped {seq_dir.name}: no detections")
            continue

        min_frame = min(by_frame)
        max_frame = max(by_frame)
        for frame in range(min_frame, max_frame + 1):
            tracker.step(by_frame.get(frame, []))
            results.extend(tracker.collect_frame_results(frame=frame))

        out_path = output_dir / f"{seq_dir.name}.txt"
        save_mot_results(out_path, sort_results(results))
        if verbose:
            print(
                f"[train] saved {out_path} ({len(results)} rows, frames {min_frame}-{max_frame})"
            )


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    run_train_dataset(TRAIN_DATA_PATH, TRAIN_PREDICTIONS_PATH, cfg["tracker"])


if __name__ == "__main__":
    main()
