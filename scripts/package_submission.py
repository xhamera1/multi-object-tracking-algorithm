from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build submission.zip from data and source.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory with MOT_01/06/07 .txt files.")
    parser.add_argument("--source-dir", type=Path, default=Path("."), help="Root source directory to include.")
    parser.add_argument("--output-dir", type=Path, default=Path("../"), help="Where to create submission.zip.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    submission_root = args.output_dir / "submission"
    data_dir = submission_root / "data"
    source_dir = submission_root / "source"

    if submission_root.exists():
        shutil.rmtree(submission_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    for seq in ("MOT_01", "MOT_06", "MOT_07"):
        src = args.pred_dir / f"{seq}.txt"
        if not src.exists():
            raise FileNotFoundError(f"Missing prediction file: {src}")
        shutil.copy2(src, data_dir / src.name)

    for item in args.source_dir.iterdir():
        if item.name == "outputs":
            continue
        dst = source_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    zip_path = args.output_dir / "submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(args.output_dir / "submission"), "zip", root_dir=args.output_dir, base_dir="submission")
    print(f"Created: {zip_path}")


if __name__ == "__main__":
    main()
