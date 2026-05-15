from pathlib import Path

TRAIN_DATA_PATH = Path("data/evs_mot-train")
TEST_DATA_PATH = Path("data/evs_mot-test")
TRAIN_PREDICTIONS_PATH = Path("outputs/train_predictions")
TEST_PREDICTIONS_PATH = Path("outputs/test_predictions")
CONFIG = Path("config/default.yaml")
GRID_SEARCH_CONFIG = Path("config/grid_search.yaml")
EDA_OUTPUT_PATH = Path("outputs/eda")
VISUALIZATIONS_OUTPUT_PATH = Path("outputs/visualizations")
EVALUATE_TRAIN_JSON = Path("outputs/logs/train_eval_summary.json")
GRID_SEARCH_RESULTS_JSON = Path("outputs/logs/grid_search_results.json")
