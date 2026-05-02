# Multi Object Tracking Algorithm

Projekt realizuje tracking-by-detection dla MOT (sledzenie wielu osob na podstawie gotowych detekcji `det.txt`).

## Co juz mamy

- Etap 0: EDA danych (statystyki, kontrole spojnosci, podglady bbox na klatkach).
- Etap 1: baseline tracker IoU + Hungarian + zapis wynikow MOT + ewaluacja MOTA.
- Etap 2: poprawa stabilnosci torow (Kalman, koszt IoU+centroid, gating).
- Wizualizacja trackow z ID na klatkach (predykcje, opcjonalnie z GT).

## Co dalej (kolejne kroki)

- Etap 3: dalsza redukcja `IDSW` (tuning wag kosztu i parametrow gatingu).
- Etap 4: finalna ewaluacja, inferencja na test i przygotowanie `submission.zip`.

## Jak uruchomic

Wszystkie komendy uruchamiaj z katalogu `source/`.

1. Instalacja:
   - `pip install -r requirements.txt`
2. EDA:
   - `PYTHONPATH=. python scripts/eda_dataset.py --data-root ../data/evs_mot-train --output-dir outputs/eda`
3. Train (Etap 2):
   - `PYTHONPATH=. python scripts/run_train.py --data-root ../data/evs_mot-train --output-dir outputs/train_predictions_stage2`
4. Ewaluacja train:
   - `PYTHONPATH=. python scripts/evaluate_train.py --pred-dir outputs/train_predictions_stage2 --gt-root ../data/evs_mot-train --output-file outputs/logs/train_eval_stage2_summary.json`
5. Test:
   - `PYTHONPATH=. python scripts/run_test.py --data-root ../data/evs_mot-test --output-dir outputs/test_predictions`
6. Wizualizacje trackow:
   - `PYTHONPATH=. python scripts/visualize_tracks.py --data-root ../data/evs_mot-train --pred-dir outputs/train_predictions_stage2 --output-dir outputs/visualizations --with-gt`

## Gdzie podgladac wyniki

- Raport EDA: `source/outputs/eda/EDA_REPORT.md`
- Podglady EDA: `source/outputs/eda/*_preview.png`
- Metryki: `source/outputs/logs/train_eval_stage2_summary.json`
- Predykcje train: `source/outputs/train_predictions_stage2/*.txt`
- Predykcje test: `source/outputs/test_predictions/MOT_01.txt`, `MOT_06.txt`, `MOT_07.txt`
- Wizualizacje trackow z ID: `source/outputs/visualizations/*_tracks_preview.png`
