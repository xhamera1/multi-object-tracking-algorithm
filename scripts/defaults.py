"""Domyślne ścieżki względem korzenia repozytorium (cwd = projekt)."""

from pathlib import Path

DEFAULT_TRAIN_DATA_ROOT = Path("data/evs_mot-train")
DEFAULT_TEST_DATA_ROOT = Path("data/evs_mot-test")
DEFAULT_TRAIN_PRED_DIR = Path("outputs/train_predictions")
DEFAULT_TEST_PRED_DIR = Path("outputs/test_predictions")
DEFAULT_TRACKER_CONFIG = Path("config/default.yaml")
DEFAULT_GRID_CONFIG = Path("config/grid_search.yaml")
DEFAULT_EDA_OUTPUT_DIR = Path("outputs/eda")
DEFAULT_VIZ_OUTPUT_DIR = Path("outputs/visualizations")
DEFAULT_EVAL_JSON = Path("outputs/logs/train_eval_summary.json")
DEFAULT_GRID_SEARCH_JSON = Path("outputs/logs/grid_search_results.json")
