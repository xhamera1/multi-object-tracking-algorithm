from __future__ import annotations

import argparse
import colorsys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from scripts.defaults import (
    DEFAULT_TRAIN_DATA_ROOT,
    DEFAULT_TRAIN_PRED_DIR,
    DEFAULT_VIZ_OUTPUT_DIR,
)


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
    parser = argparse.ArgumentParser(
        description="Visualize tracker predictions with IDs on image frames (MP4 per sequence, OpenCV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_TRAIN_DATA_ROOT,
        help="Path to split root (train or test).",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=DEFAULT_TRAIN_PRED_DIR,
        help="Directory with MOT prediction txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_VIZ_OUTPUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional sequence names to visualize (e.g. MOT_02 MOT_03).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=180,
        help="Max frames per video (consecutive from first prediction frame).",
    )
    parser.add_argument("--fps", type=float, default=12.0, help="Output video frame rate.")
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


def bgr_for_track(track_id: int) -> tuple[int, int, int]:
    h = (track_id % 20) / 20.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def find_image_for_frame(img_dir: Path, frame: int) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = img_dir / f"{frame:06d}{ext}"
        if candidate.exists():
            return candidate
    return None


def probe_frame_size(img_dir: Path, video_frames: list[int]) -> tuple[int, int]:
    for frame in video_frames:
        img_path = find_image_for_frame(img_dir, frame)
        if not img_path:
            continue
        im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if im is not None:
            return im.shape[1], im.shape[0]
    return 640, 480


def load_bgr_frame(img_dir: Path, frame: int, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    img_path = find_image_for_frame(img_dir, frame)
    if img_path:
        im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if im is not None:
            if im.shape[1] != w or im.shape[0] != h:
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
            return im
    return np.zeros((h, w, 3), dtype=np.uint8)


def draw_dashed_rect(
    img: np.ndarray,
    x: int,
    y: int,
    rw: int,
    rh: int,
    color: tuple[int, int, int],
    thickness: int = 1,
    dash: int = 6,
) -> None:
    x2, y2 = x + rw, y + rh

    def h_seg(xa: int, xb: int, yy: int) -> None:
        t = xa
        while t < xb:
            cv2.line(img, (t, yy), (min(t + dash, xb), yy), color, thickness, cv2.LINE_AA)
            t += dash * 2

    def v_seg(ya: int, yb: int, xx: int) -> None:
        t = ya
        while t < yb:
            cv2.line(img, (xx, t), (xx, min(t + dash, yb)), color, thickness, cv2.LINE_AA)
            t += dash * 2

    h_seg(x, x2, y)
    h_seg(x, x2, y2)
    v_seg(y, y2, x)
    v_seg(y, y2, x2)


def render_frame_bgr(
    seq_name: str,
    frame: int,
    img_dir: Path,
    size_wh: tuple[int, int],
    pred_by_frame: dict[int, list[TrackRow]],
    gt_by_frame: dict[int, list[GtRow]],
    with_gt: bool,
) -> np.ndarray:
    img = load_bgr_frame(img_dir, frame, size_wh)

    for row in pred_by_frame.get(frame, []):
        color = bgr_for_track(row.track_id)
        x1, y1 = int(row.x), int(row.y)
        x2, y2 = int(row.x + row.w), int(row.y + row.h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"ID {row.track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        tthick = 1
        (tw, th), bl = cv2.getTextSize(label, font, scale, tthick)
        ty = max(th + 6, y1 - 2)
        tx0, ty0 = x1, ty - th - bl - 4
        cv2.rectangle(img, (tx0, ty0), (tx0 + tw + 4, ty0 + th + bl + 4), (0, 0, 0), -1)
        cv2.putText(img, label, (tx0 + 2, ty0 + th + 2), font, scale, color, tthick, cv2.LINE_AA)

    if with_gt:
        lime = (0, 255, 0)
        for g in gt_by_frame.get(frame, []):
            draw_dashed_rect(img, int(g.x), int(g.y), int(g.w), int(g.h), lime, thickness=1)

    bar = f"{seq_name}  frame {frame}"
    cv2.putText(img, bar, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, bar, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def open_mp4_writer(path: Path, fps: float, size_wh: tuple[int, int]) -> cv2.VideoWriter:
    w, h = size_wh
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if writer.isOpened():
        return writer
    writer.release()
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if writer.isOpened():
        return writer
    writer.release()
    raise RuntimeError(
        f"OpenCV VideoWriter could not open {path} (tried fourcc mp4v, avc1). "
        "Rebuild OpenCV with video codecs or try another machine / conda ffmpeg-enabled build."
    )


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
        min_f, max_f = all_frames[0], all_frames[-1]
        span = max_f - min_f + 1
        n = min(args.max_frames, span)
        video_frames = list(range(min_f, min_f + n))

        size_wh = probe_frame_size(img_dir, video_frames)
        out_path = args.output_dir / f"{seq_dir.name}_tracks_preview.mp4"
        writer = open_mp4_writer(out_path, args.fps, size_wh)
        try:
            for frame in video_frames:
                bgr = render_frame_bgr(
                    seq_dir.name,
                    frame,
                    img_dir,
                    size_wh,
                    pred_by_frame,
                    gt_by_frame,
                    args.with_gt,
                )
                writer.write(bgr)
        finally:
            writer.release()

        print(f"[viz] saved {out_path}")


if __name__ == "__main__":
    main()
