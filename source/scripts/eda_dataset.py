from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class DetectionRow:
    frame: int
    x: float
    y: float
    w: float
    h: float
    confidence: float


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
    visibility: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for MOT dataset (detections + GT + visuals).")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to split root, e.g. evs_mot-train.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eda"),
        help="Directory for report and plots.",
    )
    parser.add_argument(
        "--num-sequences-to-visualize",
        type=int,
        default=3,
        help="How many sequences to draw visual examples for.",
    )
    parser.add_argument(
        "--frames-per-sequence",
        type=int,
        default=3,
        help="How many frames per sequence to visualize.",
    )
    return parser.parse_args()


def read_det_file(path: Path) -> list[DetectionRow]:
    rows: list[DetectionRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            frame, _, x, y, w, h, conf = line.split(",")
            rows.append(
                DetectionRow(
                    frame=int(frame),
                    x=float(x),
                    y=float(y),
                    w=float(w),
                    h=float(h),
                    confidence=float(conf),
                )
            )
    return rows


def read_gt_file(path: Path) -> list[GtRow]:
    rows: list[GtRow] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            frame, track_id, x, y, w, h, eval_flag, cls, visibility = line.split(",")
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
                    visibility=float(visibility),
                )
            )
    return rows


def quantiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def at(p: float) -> float:
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return sorted_vals[idx]

    return at(0.05), at(0.5), at(0.95)


def frame_consistency(frames: list[int]) -> dict[str, int]:
    if not frames:
        return {"min_frame": 0, "max_frame": 0, "missing_frame_count": 0, "first_frame_is_1": 0}
    unique = sorted(set(frames))
    expected = set(range(unique[0], unique[-1] + 1))
    missing = expected.difference(unique)
    return {
        "min_frame": unique[0],
        "max_frame": unique[-1],
        "missing_frame_count": len(missing),
        "first_frame_is_1": 1 if unique[0] == 1 else 0,
    }


def summarize_sequence(seq_dir: Path) -> dict:
    det_path = seq_dir / "det" / "det.txt"
    gt_path = seq_dir / "gt" / "gt.txt"
    detections = read_det_file(det_path)
    gt_rows = read_gt_file(gt_path)

    confs = [d.confidence for d in detections]
    widths = [d.w for d in detections]
    heights = [d.h for d in detections]
    areas = [d.w * d.h for d in detections]
    invalid_boxes = sum(1 for d in detections if d.w <= 0 or d.h <= 0)

    frame_counts: dict[int, int] = {}
    for d in detections:
        frame_counts[d.frame] = frame_counts.get(d.frame, 0) + 1
    dets_per_frame = list(frame_counts.values())

    det_frame_check = frame_consistency([d.frame for d in detections])
    gt_frame_check = frame_consistency([g.frame for g in gt_rows])

    q05, q50, q95 = quantiles(confs)
    a05, a50, a95 = quantiles(areas)
    d05, d50, d95 = quantiles([float(v) for v in dets_per_frame] or [0.0])

    gt_eval_people = [g for g in gt_rows if g.eval_flag == 1 and g.cls == 1]

    return {
        "name": seq_dir.name,
        "detections_count": len(detections),
        "gt_count": len(gt_rows),
        "gt_eval_people_count": len(gt_eval_people),
        "det_conf_mean": statistics.fmean(confs) if confs else 0.0,
        "det_conf_min": min(confs) if confs else 0.0,
        "det_conf_q05": q05,
        "det_conf_median": q50,
        "det_conf_q95": q95,
        "det_conf_max": max(confs) if confs else 0.0,
        "det_per_frame_mean": statistics.fmean(dets_per_frame) if dets_per_frame else 0.0,
        "det_per_frame_q05": d05,
        "det_per_frame_median": d50,
        "det_per_frame_q95": d95,
        "bbox_area_q05": a05,
        "bbox_area_median": a50,
        "bbox_area_q95": a95,
        "bbox_width_mean": statistics.fmean(widths) if widths else 0.0,
        "bbox_height_mean": statistics.fmean(heights) if heights else 0.0,
        "invalid_bbox_count": invalid_boxes,
        "det_frame_check": det_frame_check,
        "gt_frame_check": gt_frame_check,
    }


def find_image_for_frame(img_dir: Path, frame: int) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = img_dir / f"{frame:06d}{ext}"
        if candidate.exists():
            return candidate
    return None


def visualize_sequence(seq_dir: Path, output_dir: Path, frames_per_sequence: int = 3) -> Path | None:
    det_path = seq_dir / "det" / "det.txt"
    gt_path = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"
    if not det_path.exists() or not img_dir.exists():
        return None

    detections = read_det_file(det_path)
    gt_rows = read_gt_file(gt_path)
    frames = sorted(set(d.frame for d in detections))
    if not frames:
        return None

    if len(frames) <= frames_per_sequence:
        picked = frames
    else:
        step = max(1, len(frames) // frames_per_sequence)
        picked = frames[::step][:frames_per_sequence]

    cols = len(picked)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5), squeeze=False)
    axes = axes[0]

    det_by_frame: dict[int, list[DetectionRow]] = {}
    for d in detections:
        det_by_frame.setdefault(d.frame, []).append(d)
    gt_by_frame: dict[int, list[GtRow]] = {}
    for g in gt_rows:
        if g.eval_flag == 1 and g.cls == 1:
            gt_by_frame.setdefault(g.frame, []).append(g)

    for ax, frame in zip(axes, picked):
        img_path = find_image_for_frame(img_dir, frame)
        if img_path and img_path.exists():
            image = plt.imread(img_path)
            ax.imshow(image)
        else:
            ax.set_facecolor("black")

        for d in det_by_frame.get(frame, []):
            rect = Rectangle((d.x, d.y), d.w, d.h, fill=False, color="yellow", linewidth=1.5)
            ax.add_patch(rect)
        for g in gt_by_frame.get(frame, []):
            rect = Rectangle((g.x, g.y), g.w, g.h, fill=False, color="lime", linewidth=1.5)
            ax.add_patch(rect)

        ax.set_title(f"{seq_dir.name} frame {frame}")
        ax.axis("off")

    fig.suptitle("Yellow: detections, Green: GT(eval_flag=1,class=1)")
    out_path = output_dir / f"{seq_dir.name}_preview.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def aggregate_warnings(seq_summaries: list[dict]) -> list[str]:
    warnings: list[str] = []
    for s in seq_summaries:
        name = s["name"]
        if s["invalid_bbox_count"] > 0:
            warnings.append(f"{name}: invalid bbox count = {s['invalid_bbox_count']}")
        if s["det_frame_check"]["missing_frame_count"] > 0:
            warnings.append(
                f"{name}: missing detection frame ids = {s['det_frame_check']['missing_frame_count']}"
            )
        if s["det_conf_q05"] < 0.1:
            warnings.append(f"{name}: very low confidence tail (q05={s['det_conf_q05']:.3f})")
        if s["det_per_frame_q95"] > 20:
            warnings.append(f"{name}: dense frames likely (det/frame q95={s['det_per_frame_q95']:.1f})")
    return warnings


def write_report(path: Path, seq_summaries: list[dict], preview_paths: Iterable[Path], warnings: list[str]) -> None:
    total_det = sum(s["detections_count"] for s in seq_summaries)
    total_gt = sum(s["gt_count"] for s in seq_summaries)
    total_invalid = sum(s["invalid_bbox_count"] for s in seq_summaries)

    lines: list[str] = []
    lines.append("# EDA Report - MOT Dataset")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Detection statistics: confidence, detections-per-frame, bbox size distributions.")
    lines.append("- Sequence consistency checks: frame index continuity and value ranges.")
    lines.append("- Visual sanity checks: detections overlaid with GT on sample frames.")
    lines.append("")
    lines.append("## Global Summary")
    lines.append(f"- Sequences analyzed: {len(seq_summaries)}")
    lines.append(f"- Total detections: {total_det}")
    lines.append(f"- Total GT rows (if available): {total_gt}")
    lines.append(f"- Total invalid bboxes (w<=0 or h<=0): {total_invalid}")
    lines.append("")
    lines.append("## Sequence Summaries")
    for s in seq_summaries:
        lines.append(f"### {s['name']}")
        lines.append(
            f"- Detections: {s['detections_count']} | GT rows: {s['gt_count']} | GT eval class=1: {s['gt_eval_people_count']}"
        )
        lines.append(
            "- Confidence: "
            f"min={s['det_conf_min']:.3f}, q05={s['det_conf_q05']:.3f}, median={s['det_conf_median']:.3f}, "
            f"q95={s['det_conf_q95']:.3f}, max={s['det_conf_max']:.3f}, mean={s['det_conf_mean']:.3f}"
        )
        lines.append(
            "- Detections/frame: "
            f"q05={s['det_per_frame_q05']:.2f}, median={s['det_per_frame_median']:.2f}, "
            f"q95={s['det_per_frame_q95']:.2f}, mean={s['det_per_frame_mean']:.2f}"
        )
        lines.append(
            "- BBox area: "
            f"q05={s['bbox_area_q05']:.2f}, median={s['bbox_area_median']:.2f}, q95={s['bbox_area_q95']:.2f}; "
            f"mean width={s['bbox_width_mean']:.2f}, mean height={s['bbox_height_mean']:.2f}"
        )
        lines.append(
            "- Frame checks (DET): "
            f"min={s['det_frame_check']['min_frame']}, max={s['det_frame_check']['max_frame']}, "
            f"missing={s['det_frame_check']['missing_frame_count']}, starts_at_1={bool(s['det_frame_check']['first_frame_is_1'])}"
        )
        lines.append(
            "- Frame checks (GT): "
            f"min={s['gt_frame_check']['min_frame']}, max={s['gt_frame_check']['max_frame']}, "
            f"missing={s['gt_frame_check']['missing_frame_count']}, starts_at_1={bool(s['gt_frame_check']['first_frame_is_1'])}"
        )
        lines.append("")

    lines.append("## Observations That May Hurt Tracking")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- No major data integrity warning triggered by current checks.")
    lines.append("")
    lines.append("## Generated Visualizations")
    for preview in preview_paths:
        lines.append(f"- `{preview.name}`")
    lines.append("")
    lines.append("## Next Recommendations")
    lines.append("- Start with a robust confidence threshold search (e.g. 0.2 to 0.6).")
    lines.append("- Tune max_age jointly with IoU threshold to control FN vs IDSW trade-off.")
    lines.append("- Inspect sequences with dense frames first if ID switches are high.")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(p for p in args.data_root.iterdir() if p.is_dir() and (p / "det" / "det.txt").exists())
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence folders with det/det.txt found in: {args.data_root}")

    seq_summaries = [summarize_sequence(seq) for seq in seq_dirs]
    previews: list[Path] = []
    for seq in seq_dirs[: max(0, args.num_sequences_to_visualize)]:
        preview = visualize_sequence(seq, args.output_dir, frames_per_sequence=args.frames_per_sequence)
        if preview:
            previews.append(preview)

    warnings = aggregate_warnings(seq_summaries)
    report_path = args.output_dir / "EDA_REPORT.md"
    write_report(report_path, seq_summaries, previews, warnings)

    print(f"[EDA] analyzed sequences: {len(seq_summaries)}")
    print(f"[EDA] report: {report_path}")
    if previews:
        print(f"[EDA] previews: {len(previews)} image(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
