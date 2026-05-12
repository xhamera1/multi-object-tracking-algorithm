# Multi-object tracking (tracking-by-detection)

**Multiple object tracking (MOT)** from fixed public detections: each frame provides bounding boxes and detector confidence; the tracker must assign **consistent identities** across the whole sequence. This repository implements a **tracking-by-detection** pipeline with a Kalman motion model, Hungarian data association, **IoU + centroid distance** costs, **gating**, and a **two-stage** association scheme inspired by ByteTrack, tuned via **grid search** on the training split.

**Authors:** Patryk Chamera, Mateusz Bielówka, Karol Bystrek
**Context:** Advanced computer-vision course project (Codabench-style benchmark: `evs_mot-train` / `evs_mot-test`). A slide deck summarizing the approach and metrics is in [`PRESENTATION.pdf`](PRESENTATION.pdf).

---

## Problem and metric

**Input:** per-frame detections `det.txt` (`frame`, `bb_left`, `bb_top`, `bb_width`, `bb_height`, `confidence`).  
**Output:** MOT-format trajectories (`frame`, `id`, box, …) for evaluation and submission.

Quality is summarized by **MOTA** (Multiple Object Tracking Accuracy):

\[
\text{MOTA} = 1 - \frac{\mathrm{FN} + \mathrm{FP} + \mathrm{IDSW}}{\mathrm{GT}}
\]

where **FN** is missed ground-truth boxes, **FP** is predicted boxes without a match, **IDSW** counts identity switches, and **GT** is the total number of ground-truth boxes over all frames.

---

## Method

### 1. Motion model

Each track keeps an **8D constant-velocity Kalman filter** on the box: state \([x, y, w, h, v_x, v_y, v_w, v_h]\), measurement \([x, y, w, h]\). Prediction runs every frame so tracks can stay alive for several frames without a detection (`max_age`). Width and height are clamped to at least 1 pixel after predict/update to avoid degenerate boxes.

### 2. Association cost and gating

Assignments use the **Hungarian algorithm** on a **combined cost**:

\[
\text{cost} = w_{\text{iou}}(1 - \text{IoU}) + w_{c}\, d_{\text{norm}}
\]

with default weights \(w_{\text{iou}} = 0.65\), \(w_{c} = 0.35\). The normalized center distance \(d_{\text{norm}}\) scales Euclidean distance between box centers by the average box diagonal so it is roughly scale-invariant.

Even with an optimal assignment, some pairs are implausible. **Gating** rejects matches unless all of the following hold (defaults from the tuned `config/default.yaml`):

- IoU \(\geq\) `iou_match_threshold` (stage 1),
- normalized center distance \(\leq\) `max_center_distance`,
- combined cost \(\leq\) `max_match_cost`.

### 3. Two-stage association (ByteTrack-style)

- **Stage 1 — high confidence:** detections with confidence \(\geq\) `det_conf_threshold` are matched to all active tracks using the full cost and gating. Unmatched high-confidence detections spawn **new** tracks.
- **Stage 2 — low confidence:** detections in \([\) `det_low_conf_threshold`, `det_conf_threshold` \()\) are used **only** to recover **already confirmed** tracks (`hits ≥ 3`), with a stricter **IoU-only** match (`iou_match_threshold_low`). Unmatched low-confidence detections are discarded (they never create tracks), which limits false positives from noisy detections.

### 4. Track lifecycle

- **Init:** new track from an unmatched high-confidence detection.  
- **Confirm:** a track is written to the output once `hits ≥ min_hits` (default `2`), optionally gated by `runtime.save_only_confirmed`.  
- **Delete:** track removed when `time_since_update > max_age` (default `35`).

### 5. Hyperparameter search

A **Cartesian grid** over six dimensions (e.g. confidence thresholds, IoU gates, `max_age`, `min_hits`) was evaluated on the full training set with the **local MOTA** from `scripts/evaluate_train.py`; the best setting is stored in [`config/default.yaml`](config/default.yaml). See [`PRESENTATION.html`](PRESENTATION.html) for the exact search ranges (648 configurations in the deck).

---

## Results

### Training split (`evs_mot-train`)

Aggregated metrics from [`outputs/logs/train_eval_summary.json`](outputs/logs/train_eval_summary.json) (local CLEAR-style counting used by `evaluate_train.py`):

| Split | MOTA | FN | FP | IDSW | GT |
|--------|------|-----|-----|------|-----|
| **Overall** | **50.3%** | 32,400 | 3,553 | 140 | 72,638 |

Per sequence:

| Sequence | MOTA | FN | FP | IDSW |
|----------|------|-----|-----|------|
| MOT_02 | 51.2% | 21,344 | 1,853 | 0 |
| MOT_03 | 49.1% | 3,407 | 104 | 11 |
| MOT_04 | 55.3% | 2,363 | 12 | 3 |
| MOT_05 | 45.5% | 5,286 | 1,584 | 126 |

**Takeaway:** misses (**FN**) dominate; the hardest training sequence is **MOT_05** (denser scene, more FP and ID switches).

### Test split (Codabench)

Official benchmark numbers from the project deck ([`PRESENTATION.html`](PRESENTATION.html)):

| Split | MOTA | FN | FP | IDSW |
|--------|------|-----|-----|------|
| **Overall** | **42.4%** | 21,069 | 1,344 | 421 |

Per sequence:

| Sequence | MOTA | FN | FP | IDSW |
|----------|------|-----|-----|------|
| MOT_01 | 32.7% | 12,200 | 241 | 71 |
| MOT_06 | 56.5% | 3,848 | 195 | 57 |
| MOT_07 | 46.6% | 5,021 | 908 | 293 |

**Takeaway:** **MOT_01** is crowded (high FN); **MOT_07** is more dynamic with more FP and ID switches.

---

## Repository layout

```text
.
├── mot/                 # Tracker core: Kalman filter, association, I/O, geometry
├── scripts/             # CLI entry points and shared paths (defaults.py)
├── config/              # default.yaml, grid_search.yaml
├── notebooks/           # EDA (e.g. mot_eda.ipynb)
├── outputs/             # Predictions, eval JSON, EDA report, visualizations (often gitignored)
├── data/                # evs_mot-train, evs_mot-test (not shipped; place locally)
├── requirements.txt
├── pyproject.toml
└── PRESENTATION.html    # Slide-style summary (problem, method, results)
```

---

## Setup

**Python:** 3.10+ (see `pyproject.toml`).

```bash
pip install -r requirements.txt
# optional editable install
pip install -e .
```

Core runtime dependencies: `numpy`, `scipy`, `pyyaml`. Optional dev stack: `matplotlib`, `jupyter`, `ipykernel` (see optional extras in `pyproject.toml`).

**Data layout** (defaults in [`scripts/defaults.py`](scripts/defaults.py)):

- Training: `data/evs_mot-train/<SEQUENCE>/det/det.txt` (+ `img1/`, `gt/` for evaluation)
- Test: `data/evs_mot-test/<SEQUENCE>/det/det.txt`

Override paths with each script’s CLI flags where needed.

---

## Usage

Run commands from the **repository root** (current working directory = project root).

| Task | Command |
|------|---------|
| Exploratory data analysis | `python -m scripts.eda_dataset` |
| Run tracker on **train** sequences | `python -m scripts.run_train` |
| Evaluate **train** predictions (MOTA JSON) | `python -m scripts.evaluate_train` |
| Run tracker on **test** sequences (MOT_01, MOT_06, MOT_07) | `python -m scripts.run_test` |
| Visualize tracks (optional `--with-gt`) | `python -m scripts.visualize_tracks` |
| Grid search (writes best config / logs) | `python -m scripts.tune_grid` |
| Build `submission.zip` for Codabench | `python -m scripts.package_submission` |

Each module supports `--help`. Shared default paths live in `scripts/defaults.py`.

**Outputs (defaults):**

- Train predictions: `outputs/train_predictions/*.txt`
- Test predictions: `outputs/test_predictions/MOT_01.txt`, `MOT_06.txt`, `MOT_07.txt`
- Train eval summary: `outputs/logs/train_eval_summary.json`
- EDA report: `outputs/eda/EDA_REPORT.md`
- Grid search log: `outputs/logs/grid_search_results.json`

---

## Configuration

The active tracker hyperparameters live in [`config/default.yaml`](config/default.yaml). Important fields:

- **Stage 1:** `det_conf_threshold`, `iou_match_threshold`, `max_center_distance`, `max_match_cost`, `weight_iou`, `weight_center_distance`
- **Stage 2:** `det_low_conf_threshold`, `iou_match_threshold_low`
- **Lifecycle:** `max_age`, `min_hits`, `next_track_id_start`
- **Runtime:** `save_only_confirmed` under `runtime:`

The values in repo match the tuned configuration described in [`PRESENTATION.html`](PRESENTATION.html).

---

## Submission format

Codabench expects a zip archive containing:

```text
submission/
├── data/
│   ├── MOT_01.txt
│   ├── MOT_06.txt
│   └── MOT_07.txt
└── source/
    └── ...   # source used to produce the predictions
```
