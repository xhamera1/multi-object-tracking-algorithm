from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mot.io import load_detections, save_mot_results
from mot.postprocess import sort_results
from mot.tracker import MultiObjectTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MOT tracker on test split.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to evs_mot-test.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output predictions dir.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Tracker config yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tracker_cfg = cfg["tracker"]
    runtime_cfg = cfg["runtime"]

    target_sequences = {"MOT_01", "MOT_06", "MOT_07"}

    for seq_dir in sorted(p for p in args.data_root.iterdir() if p.is_dir() and p.name in target_sequences):
        det_file = seq_dir / "det" / "det.txt"
        if not det_file.exists():
            continue

        detections = load_detections(det_file)
        tracker = MultiObjectTracker(**tracker_cfg)
        results = []

        by_frame = {}
        for d in detections:
            by_frame.setdefault(d.frame, []).append(d)

        if not by_frame:
            print(f"[test] skipped {seq_dir.name}: no detections")
            continue

        min_frame = min(by_frame)
        max_frame = max(by_frame)
        for frame in range(min_frame, max_frame + 1):
            tracker.step(by_frame.get(frame, []))
            results.extend(
                tracker.collect_frame_results(
                    frame=frame, confirmed_only=runtime_cfg.get("save_only_confirmed", True)
                )
            )

        out_path = args.output_dir / f"{seq_dir.name}.txt"
        save_mot_results(out_path, sort_results(results))
        print(f"[test] saved {out_path} ({len(results)} rows, frames {min_frame}-{max_frame})")


if __name__ == "__main__":
    main()
