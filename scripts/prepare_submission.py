from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.constants import TEST_DATA_PATH, TEST_PREDICTIONS_PATH, CONFIG


def _should_skip_submission_copy(path: Path) -> bool:
    name = path.name
    if name == "outputs":
        return True
    if name == "submission":
        return True
    if name == "submission.zip":
        return True
    if name in {
        ".git",
        ".venv",
        "venv",
        "env",
        ".cursor",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".DS_Store",
    }:
        return True
    if name == "data":
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build submission.zip from data and source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=TEST_PREDICTIONS_PATH,
        help="Directory with MOT_01/06/07 .txt files (run_test writes here before packaging).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=TEST_DATA_PATH,
        help="Test split root passed to run_test.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG,
        help="Tracker yaml passed to run_test.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("."),
        help="Root source directory to include.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where submission/ and submission.zip are written (default: current dir / project root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_cmd = [
        sys.executable,
        "-m",
        "scripts.run_test",
        "--output-dir",
        str(args.pred_dir),
        "--data-root",
        str(args.data_root),
        "--config",
        str(args.config),
    ]
    print("[package_submission]", " ".join(test_cmd))
    subprocess.run(test_cmd, check=True)

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
        if _should_skip_submission_copy(item):
            continue
        dst = source_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    zip_path = args.output_dir / "submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(
        str(args.output_dir / "submission"),
        "zip",
        root_dir=args.output_dir,
        base_dir="submission",
    )
    print(f"Created: {zip_path}")


if __name__ == "__main__":
    main()
