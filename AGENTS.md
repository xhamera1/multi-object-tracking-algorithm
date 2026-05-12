# AGENTS.md

Briefing for coding agents. Human narrative → `PLAN.md` / `README.md`.

## Task

- **Input:** per-sequence `det/det.txt` (given person detections). Tracker: associate detections across time, assign stable `id`, lifecycle (init / match / delete).
- **Metric:** maximize **MOTA**; diagnose **FN, FP, IDSW** (`MOTA = 1 - sum(FN+FP+IDSW)/sum(gt)` per assignment).
- **Train:** `data/evs_mot-train` (in repo) or `../data/evs_mot-train` (sibling folder) — `det/`, `img1/`, `gt/` (GT for eval only). **Test:** `data/evs_mot-test` — **no `gt/`**.
- **Outstanding (repo):** Etap 3 (IDSW / tuning, optional appearance). Etap 4 (final test run, format validation, `submission.zip`).

## File formats (must not drift)

**`det.txt`** (CSV):

`<frame>,-1,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>` — `frame` **1-based**; `conf` ∈ [0,1]; bbox floats OK.

**`gt.txt`** (train):

`<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<eval_flag>,<class>,<visibility>`

**Tracker output / submission** (one row per instance):

`<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,1,-1,-1,-1`

`id` unique per object for full sequence (values arbitrary). Trailing `1,-1,-1,-1` fixed.

**Submission zip:** `submission/data/MOT_01.txt`, `MOT_06.txt`, `MOT_07.txt` + **`submission/source/`** (Codabench folder name for full tracker sources — copy of this project minus large artifacts). Build from **repo root**:

`python -m scripts.package_submission`

(W domyślnym ustawieniu powstają `./submission/` i `./submission.zip` w katalogu przekazanym jako `--output-dir`, domyślnie bieżący katalog / korzeń projektu.)

## Evaluator contract

- `scripts/evaluate_train.py` implements **local** frame-wise Hungarian + MOTA — **not** the TrackEval package. GT rows used iff **`eval_flag == 1` and `class == 1`** (must match any new eval code).

## Code layout

- `config/default.yaml` — thresholds: `det_conf_threshold`, `min_hits`, `max_age`, IoU/gating, `w_iou`, `w_center`, etc.
- `mot/` — `io.py`, `geometry.py`, `kalman.py`, `association.py`, `tracker.py`, `postprocess.py`
- `scripts/` — `defaults.py` (shared default paths), `eda_dataset.py`, `run_train.py`, `run_test.py`, `evaluate_train.py`, `visualize_tracks.py`, `package_submission.py`, `tune_grid.py`
- Artifacts: `outputs/` (often gitignored)

## Commands (cwd = repo root)

Run with `python -m scripts.<module>` so the `mot` package is on `sys.path`, or use `pip install -e .` once. Defaults assume `./data/evs_mot-*` and `./outputs/...` (see `scripts/defaults.py`); full option tables: `README.md`.

```bash
pip install -r requirements.txt
python -m scripts.eda_dataset
python -m scripts.run_train
python -m scripts.evaluate_train
python -m scripts.run_test
python -m scripts.visualize_tracks --with-gt
python -m scripts.tune_grid   # overwrites config/default.yaml unless --no-write-config
python -m scripts.package_submission
# Alternate output dirs / configs: pass flags (see each script's --help)
python -m scripts.run_train --output-dir outputs/train_predictions_stage2
python -m scripts.evaluate_train --pred-dir outputs/train_predictions_stage2 --output-file outputs/logs/train_eval_stage2_summary.json
```

## Pitfalls

- Never assume test sequences expose `gt/`.
- All public file formats: **1-based** `frame`.
- Changing eval filters or output columns breaks Codabench / `evaluate_train.py` parity — keep consistent with existing parsers.
