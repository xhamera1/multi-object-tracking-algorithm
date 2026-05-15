from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from itertools import product
from pathlib import Path
from typing import Any

import yaml

_SOURCE_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (_SOURCE_ROOT, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.constants import (
    GRID_SEARCH_CONFIG,
    GRID_SEARCH_RESULTS_JSON,
    TRAIN_DATA_PATH,
    CONFIG,
)
from evaluate_train import evaluate_train_metrics
from run_train import run_train_dataset

TRACKER_PARAM_KEYS = frozenset(
    {
        "det_conf_threshold",
        "det_low_conf_threshold",
        "iou_match_threshold",
        "iou_match_threshold_low",
        "max_match_cost",
        "max_age",
        "min_hits",
        "next_track_id_start",
    }
)
RUNTIME_PARAM_KEYS = frozenset({"save_only_confirmed"})


def _load_parameter_grid(path: Path) -> dict[str, list[Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw or not isinstance(raw.get("parameters"), dict):
        raise ValueError("Grid YAML must contain a top-level 'parameters' mapping of key -> list.")
    grid_spec = raw["parameters"]
    out: dict[str, list[Any]] = {}
    for name, values in grid_spec.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Grid entry '{name}' must be a non-empty list.")
        out[str(name)] = list(values)

    unknown = set(out) - TRACKER_PARAM_KEYS - RUNTIME_PARAM_KEYS
    if unknown:
        raise ValueError(f"Unknown grid parameters: {sorted(unknown)}")
    return out


def _grid_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    names = sorted(grid.keys())
    lists = [grid[k] for k in names]
    return [dict(zip(names, combo)) for combo in product(*lists)]


def _apply_overrides(
    base_tracker: dict[str, Any],
    base_runtime: dict[str, Any],
    flat: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    tracker = dict(base_tracker)
    runtime = dict(base_runtime)
    for key, value in flat.items():
        if key in TRACKER_PARAM_KEYS:
            tracker[key] = value
        elif key in RUNTIME_PARAM_KEYS:
            runtime[key] = value
    return tracker, runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search tracker hyperparameters on the train split (MOTA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Domyślnie po wyszukiwaniu nadpisywany jest plik --base-config (zwykle config/default.yaml). "
            "JSON z wynikami: --output-json.\n"
            "\n"
            "Bez nadpisywania domyślnego configu (tylko JSON):\n"
            "  python -m scripts.tune_grid --no-write-config\n"
            "\n"
            "Dodatkowa kopia najlepszego YAML (np. backup):\n"
            "  python -m scripts.tune_grid --write-best-config config/best_from_grid.yaml\n"
            "\n"
            "Potem test (domyślne ścieżki):\n"
            "  python -m scripts.run_test\n"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=TRAIN_DATA_PATH,
        help="Path to evs_mot-train (det/ per sequence).",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=None,
        help="Path to ground-truth root (gt/ per sequence). Defaults to --data-root.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=CONFIG,
        help="YAML z punktem startowym; siatka nadpisuje wybrane klucze. "
        "Po tuningu domyślnie ten sam plik jest zapisywany z najlepszymi parametrami (chyba że --no-write-config).",
    )
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=GRID_SEARCH_CONFIG,
        help="YAML with parameters: { name: [v1, v2, ...] } (Cartesian product).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=GRID_SEARCH_RESULTS_JSON,
        help="Where to save all trial results and the best setting.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for per-trial prediction folders. If omitted, uses a temp dir and deletes it.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=500,
        help="Refuse to run if the grid size exceeds this (use --force to override).",
    )
    parser.add_argument("--force", action="store_true", help="Allow grids larger than --max-trials.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-sequence train logs.")
    parser.add_argument(
        "--no-write-config",
        action="store_true",
        help="Nie zapisuj najlepszego YAML do --base-config (tylko raport JSON).",
    )
    parser.add_argument(
        "--write-best-config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Dodatkowo zapisz kopię najlepszego tracker+runtime do tego pliku (nie wyłącza zapisu do --base-config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_root = args.gt_root if args.gt_root is not None else args.data_root

    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    base_tracker: dict[str, Any] = dict(base.get("tracker", {}))
    base_runtime: dict[str, Any] = dict(base.get("runtime", {}))

    grid = _load_parameter_grid(args.grid_config)
    combos = _grid_combinations(grid)
    n = len(combos)
    if n > args.max_trials and not args.force:
        raise SystemExit(
            f"Grid has {n} combinations (>{args.max_trials}). "
            "Increase --max-trials, shrink --grid-config, or pass --force."
        )

    if args.work_dir is not None:
        work_root = args.work_dir.resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        cleanup_work = False
    else:
        work_root = Path(tempfile.mkdtemp(prefix="mot_grid_"))
        cleanup_work = True

    trials: list[dict[str, Any]] = []
    try:
        for idx, flat in enumerate(combos):
            trial_dir = work_root / f"trial_{idx:05d}"
            if trial_dir.exists():
                shutil.rmtree(trial_dir)
            trial_dir.mkdir(parents=True)

            tracker_cfg, runtime_cfg = _apply_overrides(base_tracker, base_runtime, flat)
            run_train_dataset(
                args.data_root,
                trial_dir,
                tracker_cfg,
                runtime_cfg,
                verbose=not args.quiet,
            )
            metrics = evaluate_train_metrics(trial_dir, gt_root)
            mota = float(metrics["overall"]["mota"])
            trials.append(
                {
                    "index": idx,
                    "overrides": flat,
                    "tracker": tracker_cfg,
                    "runtime": runtime_cfg,
                    "mota": mota,
                    "overall": metrics["overall"],
                    "per_sequence": metrics["per_sequence"],
                }
            )
            print(
                f"[tune] trial {idx + 1}/{n} MOTA={mota:.4f} overrides={flat}",
                flush=True,
            )
    finally:
        if cleanup_work:
            shutil.rmtree(work_root, ignore_errors=True)

    trials_sorted = sorted(trials, key=lambda t: t["mota"], reverse=True)
    best = trials_sorted[0]
    report = {
        "grid_config": str(args.grid_config.resolve()),
        "base_config": str(args.base_config.resolve()),
        "num_trials": n,
        "best": {
            "mota": best["mota"],
            "overrides": best["overrides"],
            "tracker": best["tracker"],
            "runtime": best["runtime"],
            "overall": best["overall"],
            "per_sequence": best["per_sequence"],
        },
        "trials": trials_sorted,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"\nBest MOTA={best['mota']:.4f} with overrides {best['overrides']}\n"
        f"Full results: {args.output_json}"
    )

    best_yaml = {"tracker": best["tracker"], "runtime": best["runtime"]}
    text = yaml.safe_dump(best_yaml, sort_keys=False, allow_unicode=True)

    paths_to_write: list[Path] = []
    if not args.no_write_config:
        paths_to_write.append(args.base_config.resolve())
    if args.write_best_config is not None:
        paths_to_write.append(args.write_best_config.resolve())

    seen: set[Path] = set()
    for path in paths_to_write:
        if path in seen:
            continue
        seen.add(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Wrote best config: {path}")

    if not args.no_write_config:
        print(
            "Test (przykład, domyślne ścieżki): python -m scripts.run_test"
        )


if __name__ == "__main__":
    main()
