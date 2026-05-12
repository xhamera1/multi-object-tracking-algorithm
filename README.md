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

## Uklad repozytorium

```text
.
├── mot/                  # pakiet trackera (Kalman, asocjacja, I/O)
├── scripts/              # CLI + defaults.py (wspolne domyslne sciezki)
├── config/               # default.yaml, grid_search.yaml
├── notebooks/            # EDA w Jupyter
├── outputs/              # predykcje, logi, wizualizacje (czesto poza Gitem)
├── requirements.txt
└── pyproject.toml        # opcjonalnie: pip install -e .
```

## Jak uruchomic

**Katalog roboczy:** korzen projektu (ten folder).

1. Srodowisko: `pip install -r requirements.txt` (opcjonalnie `pip install -e .`).
2. **Domyslne sciezki** zakladaja zbiory w `./data/evs_mot-train` i `./data/evs_mot-test` (patrz `scripts/defaults.py`). Inna lokalizacja → podaj odpowiednie flagi.

### Szybki start (bez argumentów)

| Cel | Komenda |
|-----|---------|
| EDA | `python -m scripts.eda_dataset` |
| Train (predykcje train) | `python -m scripts.run_train` |
| Ewaluacja MOTA (train) | `python -m scripts.evaluate_train` |
| Test (MOT_01 / 06 / 07) | `python -m scripts.run_test` |
| Wizualizacje (domyslnie train + `outputs/train_predictions`) | `python -m scripts.visualize_tracks --with-gt` |
| Grid search (aktualizuje `config/default.yaml`) | `python -m scripts.tune_grid` |
| Paczka `submission.zip` | `python -m scripts.package_submission` |

Każdy moduł `scripts.*` ma **inne argumenty** (ścieżki danych, katalogi wyjścia itd.). Pełny opis: `python -m scripts.<nazwa> --help`. Wspólne domyślne ścieżki w repo: `scripts/defaults.py`.

### Jupyter

`jupyter notebook notebooks/mot_eda.ipynb` (od korzenia projektu).

## Gdzie podgladac wyniki

- Raport EDA: `outputs/eda/EDA_REPORT.md`
- Metryki train: `outputs/logs/train_eval_summary.json` (lub inna ścieżka z `--output-file`)
- Grid search: `outputs/logs/grid_search_results.json`
- Predykcje train: `outputs/train_predictions/*.txt`
- Predykcje test: `outputs/test_predictions/MOT_01.txt`, `MOT_06.txt`, `MOT_07.txt`
- Wizualizacje: `outputs/visualizations/*_tracks_preview.png`
