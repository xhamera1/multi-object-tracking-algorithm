from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts.defaults import (
    DEFAULT_TRAIN_DATA_ROOT,
    DEFAULT_TRACKER_CONFIG,
    DEFAULT_TRAIN_PRED_DIR,
)

from mot.io import load_detections, save_mot_results
from mot.postprocess import sort_results
from mot.tracker import MultiObjectTracker


def run_train_dataset(
    data_root: Path,
    output_dir: Path,
    tracker_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    *,
    verbose: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for seq_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        det_file = seq_dir / "det" / "det.txt"
        if not det_file.exists():
            continue

        detections = load_detections(det_file)
        tracker = MultiObjectTracker(**dict(tracker_cfg))
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
            results.extend(
                tracker.collect_frame_results(
                    frame=frame, confirmed_only=runtime_cfg.get("save_only_confirmed", True)
                )
            )

        out_path = output_dir / f"{seq_dir.name}.txt"
        save_mot_results(out_path, sort_results(results))
        if verbose:
            print(f"[train] saved {out_path} ({len(results)} rows, frames {min_frame}-{max_frame})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MOT tracker on train split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_TRAIN_DATA_ROOT,
        help="Path to evs_mot-train.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TRAIN_PRED_DIR,
        help="Output predictions dir.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRACKER_CONFIG,
        help="Tracker config yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_train_dataset(args.data_root, args.output_dir, cfg["tracker"], cfg["runtime"])


if __name__ == "__main__":
    main()
