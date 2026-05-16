import yaml

from scripts.constants import (
    TEST_DATA_PATH,
    TEST_PREDICTIONS_PATH,
    CONFIG,
)

from mot import Tracker, TrackerConfig, load_detections, save_mot_results


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tracker_cfg = cfg["tracker"]

    target_sequences = {"MOT_01", "MOT_06", "MOT_07"}

    for seq_dir in sorted(
        p for p in TEST_DATA_PATH.iterdir() if p.is_dir() and p.name in target_sequences
    ):
        det_file = seq_dir / "det" / "det.txt"
        if not det_file.exists():
            continue

        detections = load_detections(det_file)
        tracker = Tracker(TrackerConfig(**tracker_cfg))
        results = []

        detections_by_frame = {}
        for d in detections:
            detections_by_frame.setdefault(d.frame, []).append(d)

        if not detections_by_frame:
            print(f"[test] skipped {seq_dir.name}: no detections")
            continue

        min_frame = min(detections_by_frame)
        max_frame = max(detections_by_frame)
        for frame in range(min_frame, max_frame + 1):
            detections = detections_by_frame.get(frame, [])
            results.extend(tracker.step(detections, frame))

        out_path = TEST_PREDICTIONS_PATH / f"{seq_dir.name}.txt"
        save_mot_results(out_path, results)
        print(
            f"[test] saved {out_path} ({len(results)} rows, frames {min_frame}-{max_frame})"
        )


if __name__ == "__main__":
    main()
