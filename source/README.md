# Source

Mamy gotowy:
- Etap 0: EDA (raport + wizualizacje),
- Etap 1 (baseline): tracker IoU + Hungarian, zapis MOT, ewaluacja MOTA.
- Etap 2: Kalman + koszt IoU/centroid + gating.

## Jak uruchomic (krotko)

1. Instalacja:
   - `pip install -r requirements.txt`
2. EDA:
   - `python scripts/eda_dataset.py --data-root ../data/evs_mot-train --output-dir outputs/eda`
3. Baseline train (predykcje):
   - `PYTHONPATH=. python scripts/run_train.py --data-root ../data/evs_mot-train --output-dir outputs/train_predictions`
4. Baseline eval (MOTA):
   - `PYTHONPATH=. python scripts/evaluate_train.py --pred-dir outputs/train_predictions --gt-root ../data/evs_mot-train --output-file outputs/logs/train_eval_summary.json`
5. Baseline test (MOT_01/06/07):
   - `PYTHONPATH=. python scripts/run_test.py --data-root ../data/evs_mot-test --output-dir outputs/test_predictions`
6. Podglad trackow z ID:
   - `PYTHONPATH=. python scripts/visualize_tracks.py --data-root ../data/evs_mot-train --pred-dir outputs/train_predictions --output-dir outputs/visualizations --with-gt`
