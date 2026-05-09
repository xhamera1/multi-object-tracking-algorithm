from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class TrackRow:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float


@dataclass
class GtRow:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    eval_flag: int
    cls: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize tracker predictions with IDs on image frames.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to split root (train or test).")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory with MOT prediction txt files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations"), help="Output directory.")
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional sequence names to visualize (e.g. MOT_02 MOT_03).",
    )
    parser.add_argument("--frames-per-sequence", type=int, default=4, help="How many frames to render per sequence.")
    parser.add_argument("--with-gt", action="store_true", help="Overlay GT boxes if gt/gt.txt exists.")
    return parser.parse_args()


def read_pred_rows(path: Path) -> list[TrackRow]:
    rows: list[TrackRow] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            frame, track_id, x, y, w, h, *_ = line.split(",")
            rows.append(
                TrackRow(
                    frame=int(frame),
                    track_id=int(track_id),
                    x=float(x),
                    y=float(y),
                    w=float(w),
                    h=float(h),
                )
            )
    return rows


def read_gt_rows(path: Path) -> list[GtRow]:
    rows: list[GtRow] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            frame, track_id, x, y, w, h, eval_flag, cls, _visibility = line.split(",")
            rows.append(
                GtRow(
                    frame=int(frame),
                    track_id=int(track_id),
                    x=float(x),
                    y=float(y),
                    w=float(w),
                    h=float(h),
                    eval_flag=int(eval_flag),
                    cls=int(cls),
                )
            )
    return rows


def color_for_track(track_id: int) -> tuple[float, float, float]:
    cmap = plt.get_cmap("tab20")
    return cmap(track_id % 20)[:3]


def find_image_for_frame(img_dir: Path, frame: int) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = img_dir / f"{frame:06d}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(p for p in args.data_root.iterdir() if p.is_dir())
    if args.sequences:
        allowed = set(args.sequences)
        seq_dirs = [p for p in seq_dirs if p.name in allowed]

    for seq_dir in seq_dirs:
        pred_file = args.pred_dir / f"{seq_dir.name}.txt"
        img_dir = seq_dir / "img1"
        if not pred_file.exists() or not img_dir.exists():
            continue

        pred_rows = read_pred_rows(pred_file)
        if not pred_rows:
            continue

        gt_rows = read_gt_rows(seq_dir / "gt" / "gt.txt") if args.with_gt else []
        pred_by_frame: dict[int, list[TrackRow]] = {}
        for r in pred_rows:
            pred_by_frame.setdefault(r.frame, []).append(r)
        gt_by_frame: dict[int, list[GtRow]] = {}
        for g in gt_rows:
            if g.eval_flag == 1 and g.cls == 1:
                gt_by_frame.setdefault(g.frame, []).append(g)

        all_frames = sorted(pred_by_frame.keys())
        if len(all_frames) <= args.frames_per_sequence:
            picked_frames = all_frames
        else:
            step = max(1, len(all_frames) // args.frames_per_sequence)
            picked_frames = all_frames[::step][: args.frames_per_sequence]

        cols = len(picked_frames)
        fig, axes = plt.subplots(1, cols, figsize=(7 * cols, 6), squeeze=False)
        axes = axes[0]

        for ax, frame in zip(axes, picked_frames):
            img_path = find_image_for_frame(img_dir, frame)
            if img_path:
                ax.imshow(plt.imread(img_path))
            else:
                ax.set_facecolor("black")

            for row in pred_by_frame.get(frame, []):
                color = color_for_track(row.track_id)
                rect = Rectangle((row.x, row.y), row.w, row.h, fill=False, color=color, linewidth=2.0)
                ax.add_patch(rect)
                ax.text(
                    row.x,
                    max(0, row.y - 3),
                    f"ID {row.track_id}",
                    color=color,
                    fontsize=8,
                    bbox={"facecolor": "black", "alpha": 0.4, "pad": 1, "edgecolor": "none"},
                )

            if args.with_gt:
                for g in gt_by_frame.get(frame, []):
                    rect = Rectangle((g.x, g.y), g.w, g.h, fill=False, color="lime", linewidth=1.2, linestyle="--")
                    ax.add_patch(rect)

            ax.set_title(f"{seq_dir.name} frame {frame}")
            ax.axis("off")

        legend = "Predictions with ID (colored), GT dashed green" if args.with_gt else "Predictions with ID (colored)"
        fig.suptitle(legend)
        out_path = args.output_dir / f"{seq_dir.name}_tracks_preview.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"[viz] saved {out_path}")


if __name__ == "__main__":
    main()
