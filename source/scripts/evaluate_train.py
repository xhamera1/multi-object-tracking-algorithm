from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def read_gt_rows(path: Path) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    by_frame: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            frame, track_id, x, y, w, h, eval_flag, cls, _visibility = line.split(",")
            if int(eval_flag) != 1 or int(cls) != 1:
                continue
            by_frame.setdefault(int(frame), []).append(
                (int(track_id), (float(x), float(y), float(w), float(h)))
            )
    return by_frame


def read_pred_rows(path: Path) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    by_frame: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    if not path.exists():
        return by_frame
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            frame, track_id, x, y, w, h, *_rest = line.split(",")
            by_frame.setdefault(int(frame), []).append(
                (int(track_id), (float(x), float(y), float(w), float(h)))
            )
    return by_frame

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate train predictions (MOTA baseline).")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Path to train prediction .txt files.")
    parser.add_argument("--gt-root", type=Path, required=True, help="Path to train data root with gt files.")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/logs/train_eval_summary.json"),
        help="Where to save JSON evaluation summary.",
    )
    return parser.parse_args()


def match_frame(
    gt_rows: list[tuple[int, tuple[float, float, float, float]]],
    pred_rows: list[tuple[int, tuple[float, float, float, float]]],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], int, int]:
    if not gt_rows and not pred_rows:
        return [], 0, 0
    if not gt_rows:
        return [], 0, len(pred_rows)
    if not pred_rows:
        return [], len(gt_rows), 0

    cost = np.zeros((len(gt_rows), len(pred_rows)), dtype=float)
    for i, (_, gbox) in enumerate(gt_rows):
        for j, (_, pbox) in enumerate(pred_rows):
            cost[i, j] = 1.0 - bbox_iou(gbox, pbox)

    gi, pj = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    matched_gt = set()
    matched_pred = set()
    for g_idx, p_idx in zip(gi, pj):
        iou = 1.0 - cost[g_idx, p_idx]
        if iou < iou_threshold:
            continue
        matches.append((g_idx, p_idx))
        matched_gt.add(g_idx)
        matched_pred.add(p_idx)

    fn = len(gt_rows) - len(matched_gt)
    fp = len(pred_rows) - len(matched_pred)
    return matches, fn, fp


def main() -> None:
    args = parse_args()
    sequence_metrics: dict[str, dict[str, float]] = {}
    total_fn = 0
    total_fp = 0
    total_idsw = 0
    total_gt = 0
    total_frames = 0

    for seq_dir in sorted(p for p in args.gt_root.iterdir() if p.is_dir()):
        gt_file = seq_dir / "gt" / "gt.txt"
        pred_file = args.pred_dir / f"{seq_dir.name}.txt"
        if not gt_file.exists():
            continue

        gt_by_frame = read_gt_rows(gt_file)
        pred_by_frame = read_pred_rows(pred_file)
        frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
        if not frames:
            continue

        seq_fn = 0
        seq_fp = 0
        seq_idsw = 0
        seq_gt_total = 0
        previous_assignment: dict[int, int] = {}

        for frame in frames:
            gt_rows = gt_by_frame.get(frame, [])
            pred_rows = pred_by_frame.get(frame, [])
            seq_gt_total += len(gt_rows)
            matches, fn, fp = match_frame(gt_rows, pred_rows, iou_threshold=0.5)
            seq_fn += fn
            seq_fp += fp

            current_assignment: dict[int, int] = {}
            for gt_idx, pred_idx in matches:
                gt_id = gt_rows[gt_idx][0]
                pred_id = pred_rows[pred_idx][0]
                if gt_id in previous_assignment and previous_assignment[gt_id] != pred_id:
                    seq_idsw += 1
                current_assignment[gt_id] = pred_id
            previous_assignment = current_assignment

        if seq_gt_total == 0:
            seq_mota = 0.0
        else:
            seq_mota = 1.0 - (seq_fn + seq_fp + seq_idsw) / seq_gt_total

        sequence_metrics[seq_dir.name] = {
            "frames": float(len(frames)),
            "gt_total": float(seq_gt_total),
            "fn": float(seq_fn),
            "fp": float(seq_fp),
            "idsw": float(seq_idsw),
            "mota": float(seq_mota),
        }
        total_frames += len(frames)
        total_gt += seq_gt_total
        total_fn += seq_fn
        total_fp += seq_fp
        total_idsw += seq_idsw

    if not sequence_metrics:
        raise RuntimeError("No sequences were evaluated. Check --gt-root and prediction files.")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    overall_mota = 0.0 if total_gt == 0 else 1.0 - (total_fn + total_fp + total_idsw) / total_gt
    metrics_out = {
        "overall": {
            "frames": float(total_frames),
            "gt_total": float(total_gt),
            "fn": float(total_fn),
            "fp": float(total_fp),
            "idsw": float(total_idsw),
            "mota": float(overall_mota),
        },
        "per_sequence": sequence_metrics,
    }
    args.output_file.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    print("Evaluation (IoU@0.5 matching):")
    print(
        f"OVERALL -> MOTA={overall_mota:.4f}, FN={total_fn}, FP={total_fp}, "
        f"IDSW={total_idsw}, GT={total_gt}, FRAMES={total_frames}"
    )
    for seq_name, seq in sequence_metrics.items():
        print(
            f"{seq_name} -> MOTA={seq['mota']:.4f}, FN={int(seq['fn'])}, FP={int(seq['fp'])}, "
            f"IDSW={int(seq['idsw'])}, GT={int(seq['gt_total'])}, FRAMES={int(seq['frames'])}"
        )
    print(f"Saved JSON summary: {args.output_file}")


if __name__ == "__main__":
    main()
