# AGENTS.md

Briefing for coding agents. Human narrative → `PLAN.md` / `README.md`.

## Task

- **Input:** per-sequence `det/det.txt` (given person detections). Tracker: associate detections across time, assign stable `id`, lifecycle (init / match / delete).
- **Metric:** maximize **MOTA**; diagnose **FN, FP, IDSW** (`MOTA = 1 - sum(FN+FP+IDSW)/sum(gt)` per assignment).
- **Train:** `../data/evs_mot-train` — `det/`, `img1/`, `gt/` (GT for eval only). **Test:** `../data/evs_mot-test` — **no `gt/`**.
- **Outstanding (repo):** Etap 3 (IDSW / tuning, optional appearance). Etap 4 (final test run, format validation, `submission.zip`).

## File formats (must not drift)

**`det.txt`** (CSV):

`<frame>,-1,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>` — `frame` **1-based**; `conf` ∈ [0,1]; bbox floats OK.

**`gt.txt`** (train):

`<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<eval_flag>,<class>,<visibility>`

**Tracker output / submission** (one row per instance):

`<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,1,-1,-1,-1`

`id` unique per object for full sequence (values arbitrary). Trailing `1,-1,-1,-1` fixed.

**Submission zip:** `submission/data/MOT_01.txt`, `MOT_06.txt`, `MOT_07.txt` + `submission/source/` (full tracker sources). Build from `source/`:

`PYTHONPATH=. python scripts/package_submission.py --pred-dir outputs/test_predictions --source-dir . --output-dir ..`

## Evaluator contract

- `source/scripts/evaluate_train.py` implements **local** frame-wise Hungarian + MOTA — **not** the TrackEval package. GT rows used iff **`eval_flag == 1` and `class == 1`** (must match any new eval code).

## Code layout

- `source/config/default.yaml` — thresholds: `det_conf_threshold`, `min_hits`, `max_age`, IoU/gating, `w_iou`, `w_center`, etc.
- `source/mot/` — `io.py`, `geometry.py`, `kalman.py`, `association.py`, `tracker.py`, `postprocess.py`
- `source/scripts/` — `eda_dataset.py`, `run_train.py`, `run_test.py`, `evaluate_train.py`, `visualize_tracks.py`, `package_submission.py`
- Artifacts: `source/outputs/` (often gitignored)

## Commands (cwd = `source/`)

```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/run_train.py --data-root ../data/evs_mot-train --output-dir outputs/train_predictions_stage2
PYTHONPATH=. python scripts/evaluate_train.py --pred-dir outputs/train_predictions_stage2 --gt-root ../data/evs_mot-train --output-file outputs/logs/train_eval_stage2_summary.json
PYTHONPATH=. python scripts/run_test.py --data-root ../data/evs_mot-test --output-dir outputs/test_predictions
# optional
PYTHONPATH=. python scripts/eda_dataset.py --data-root ../data/evs_mot-train --output-dir outputs/eda
PYTHONPATH=. python scripts/visualize_tracks.py --data-root ../data/evs_mot-train --pred-dir outputs/train_predictions_stage2 --output-dir outputs/visualizations --with-gt
```

## Pitfalls

- Never assume test sequences expose `gt/`.  
- All public file formats: **1-based** `frame`.  
- Changing eval filters or output columns breaks Codabench / `evaluate_train.py` parity — keep consistent with existing parsers.
