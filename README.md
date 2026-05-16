# Multi-object tracking

Tracking-by-detection pipeline for the EVS MOT course benchmark. Input is per-frame person detections from `det/det.txt`; output is MOT-format trajectories with stable object IDs.

Authors: Patryk Chamera, Mateusz Bielówka, Karol Bystrek

## Method

The tracker uses:

- 8D constant-velocity Kalman filter over `[x, y, w, h, vx, vy, vw, vh]`,
- Hungarian assignment with IoU cost (`1 - IoU`),
- two-stage association inspired by ByteTrack,
- track states: tracked, lost, removed.

Per frame:

1. Split detections into high-confidence and low-confidence groups.
2. Predict all tracked and lost tracks with the Kalman filter.
3. Match tracked/lost tracks to high-confidence detections.
4. Match still-unmatched active tracks to low-confidence detections.
5. Start new tracks from unmatched high-confidence detections.
6. Keep unmatched tracks as lost until `max_age`; then remove them.
7. Emit tracks after `min_hits` successful detections.

Local evaluation uses MOTA:

```text
MOTA = 1 - (FN + FP + IDSW) / GT
```

Ground truth rows are evaluated only when `eval_flag == 1` and `class == 1`.

## Repository layout

```text
mot/                 tracker library
scripts/             command-line entry points
config/default.yaml  active tracker configuration
config/grid_search.yaml
data/                local datasets, not versioned
outputs/             generated predictions, logs, visualizations
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

Expected data layout:

```text
data/
├── evs_mot-train/
│   └── <SEQUENCE>/
│       ├── det/det.txt
│       └── gt/gt.txt
└── evs_mot-test/
    └── <SEQUENCE>/
        └── det/det.txt
```

## Usage

Run commands from repository root.

```bash
python -m scripts.run_train
python -m scripts.evaluate_train
python -m scripts.run_test
python -m scripts.run_grid_search
python -m scripts.prepare_submission
```

Default paths are defined in `scripts/constants.py`:

- train predictions: `outputs/train_predictions/`
- test predictions: `outputs/test_predictions/`
- evaluation summary: `outputs/logs/train_eval_summary.json`
- grid-search results: `outputs/logs/grid_search_results.json`

## Configuration

Active settings live in `config/default.yaml`:

```yaml
tracker:
  confidence_threshold_high: 0.5
  confidence_threshold_low: 0.05
  new_track_threshold: 0.0
  first_match_cost_max: 0.8
  second_match_cost_max: 0.3
  max_age: 5
  min_hits: 1
runtime: {}
```

`config/grid_search.yaml` defines Cartesian search ranges for these tracker parameters. Grid search evaluates train MOTA and writes results to `outputs/logs/grid_search_results.json`.

## File formats

Detection input:

```text
<frame>,-1,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<confidence>
```

Tracker output:

```text
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,1,-1,-1,-1
```

Frames are 1-based. Bounding boxes use MOT `[x, y, width, height]` format.

## Submission

`scripts.prepare_submission` runs the tracker on test sequences and creates:

```text
submission.zip
└── submission/
    ├── data/
    │   ├── MOT_01.txt
    │   ├── MOT_06.txt
    │   └── MOT_07.txt
    └── source/
        └── project source files
```
