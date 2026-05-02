# PLAN REALIZACJI PROJEKTU  
**Multiple Object Tracking (MOT) na predefiniowanych detekcjach**

## 1. Cel projektu i kryterium sukcesu

Celem projektu jest zaprojektowanie i wdrozenie algorytmu MOT, ktory:
- przyjmuje gotowe detekcje (`det.txt`) dla kazdej klatki,
- utrzymuje spojna tozsamosc obiektow w czasie (stabilne `id`),
- minimalizuje liczbe:
  - `FN` (false negatives),
  - `FP` (false positives),
  - `IDSW` (identity switches),
- osiaga mozliwie najwyzsza wartosc `MOTA` na zbiorze treningowym,
- generuje poprawny formalnie wynik dla zbioru testowego (`MOT_01`, `MOT_06`, `MOT_07`).

Kryterium sukcesu:
1. Poprawny pipeline end-to-end (wczytanie danych -> tracking -> zapis wynikow -> ewaluacja).
2. Poprawny format wyjscia zgodny z wymaganiami konkursowymi.
3. Udokumentowana metodyka i eksperymenty prowadzone na zbiorze treningowym.

---

## 2. Strategia podejscia (zalecana)

Najlepszym podejsciem dla tego typu zadania jest **iteracyjne budowanie baseline'u i jego kontrolowane ulepszanie**:

1. **Baseline funkcjonalny**: prosty tracker oparty o pozycje (IoU + Hungarian).
2. **Stabilizacja tracker'a**: model ruchu (Kalman), reguly inicjalizacji i usuwania torow.
3. **Redukcja ID switchy**: koszt laczony (ruch + podobienstwo geometryczne + ewentualnie wyglad).
4. **Strojenie hiperparametrow na train**: progi confidence, IoU, `max_age`, `min_hits`.
5. **Walidacja i raportowanie**: TrackEval + analiza przypadkow bledow.
6. **Finalna inferencja na test** i przygotowanie paczki `submission.zip`.

To podejscie minimalizuje ryzyko "duzej, trudnej do debugowania implementacji" i pozwala szybko uzyskac mierzalne postepy.

---

## 3. Architektura rozwiazania

### 3.1. Moduly logiczne

1. **Data I/O**
   - parser `det.txt`, `gt.txt`,
   - normalizacja typow i walidacja rekordow.

2. **Model obiektu (`Track`)**
   - pola: `id`, `bbox`, `age`, `time_since_update`, `hits`, historia pozycji,
   - opcjonalnie stan filtra Kalmana.

3. **Predykcja ruchu**
   - wariant A (baseline): brak predykcji, ostatnia pozycja,
   - wariant B (zalecany): Kalman (np. stan: `x, y, w, h, vx, vy, vw, vh`).

4. **Data association**
   - macierz kosztu pomiedzy torami i detekcjami,
   - rozwiazanie przypisania algorytmem Hungarian,
   - obsluga unmatched tracks / unmatched detections.

5. **Zarzadzanie cyklem zycia toru**
   - inicjalizacja nowego toru,
   - potwierdzenie toru po `min_hits`,
   - usuniecie toru po `max_age`.

6. **Ewaluacja i raporty**
   - eksport predykcji do formatu MOT Challenge,
   - metryki przez TrackEval (w szczegolnosci MOTA),
   - raport porownawczy konfiguracji.

### 3.2. Proponowana struktura katalogow (source)

```text
source/
  config/
    default.yaml
  mot/
    io.py
    geometry.py
    kalman.py
    association.py
    tracker.py
    postprocess.py
  scripts/
    run_train.py
    run_test.py
    evaluate_train.py
    package_submission.py
  outputs/
    train_predictions/
    test_predictions/
    logs/
```

---

## 4. Plan implementacji etapami

## Etap 0 - Przygotowanie i interpretacja danych
**Cel**: potwierdzic rozumienie danych i wykryc nietypowe przypadki.
**Status**: ✅ WYKONANY

Zadania:
- analiza statystyk detekcji (`confidence`, liczba detekcji/klatke, rozklad rozmiarow bbox),
- wizualizacja kilku sekwencji (det i gt na klatkach),
- sprawdzenie spojnosci indeksowania klatek i zakresow wartosci.

Deliverable:
- skrypt EDA + krotki raport obserwacji (co moze psuc tracking).

Mikroetapy zrealizowane:
- [x] `source/scripts/eda_dataset.py` - statystyki det/gt + kontrole spojnosci + ostrzezenia.
- [x] Automatyczny raport `outputs/eda/EDA_REPORT.md`.
- [x] Wizualizacje pogladowe sekwencji (`*_preview.png`) z naniesionymi det (zolty) i gt (zielony).
- [x] Aktualizacja instrukcji uruchomienia w `source/README.md`.

Uruchamianie Etapu 0:
- `cd source`
- `python scripts/eda_dataset.py --data-root <sciezka_do_evs_mot-train> --output-dir outputs/eda`

## Etap 1 - Baseline tracking (MVP)
**Cel**: miec dzialajacy system end-to-end.
**Status**: ✅ WYKONANY

Zadania:
- implementacja prostego trackera IoU + Hungarian,
- podstawowe reguly:
  - prog confidence dla detekcji,
  - `min_hits`, `max_age`,
  - zapis do wymaganego formatu wyjsciowego.

Deliverable:
- pierwsze wyniki na train,
- pierwsza wartosc MOTA (punkt odniesienia).

Mikroetapy zrealizowane:
- [x] Tracker baseline IoU + Hungarian (`source/mot/tracker.py`, `source/mot/association.py`).
- [x] Reguly bazowe: `det_conf_threshold`, `min_hits`, `max_age` (`source/config/default.yaml`).
- [x] Pelny zapis wynikow w formacie MOT (`source/mot/io.py`).
- [x] Skrypt train inference (`source/scripts/run_train.py`) z obsluga pelnego zakresu klatek.
- [x] Skrypt ewaluacji baseline MOTA (`source/scripts/evaluate_train.py`) i zapis JSON z metrykami.
- [x] Wizualizacja wynikow trackera z ID na klatkach (`source/scripts/visualize_tracks.py`).

Wyniki baseline (train, punkt odniesienia):
- `OVERALL MOTA = 0.4974`
- `FN = 33180`, `FP = 3198`, `IDSW = 131`, `GT = 72638`, `FRAMES = 3066`
- per sekwencja:
  - `MOT_02: MOTA = 0.5068`
  - `MOT_03: MOTA = 0.4687`
  - `MOT_04: MOTA = 0.5423`
  - `MOT_05: MOTA = 0.4591`

Uruchamianie Etapu 1:
- `cd source`
- `pip install -r requirements.txt`
- `PYTHONPATH=. python scripts/run_train.py --data-root ../data/evs_mot-train --output-dir outputs/train_predictions`
- `PYTHONPATH=. python scripts/evaluate_train.py --pred-dir outputs/train_predictions --gt-root ../data/evs_mot-train --output-file outputs/logs/train_eval_summary.json`
- (opcjonalnie test split) `PYTHONPATH=. python scripts/run_test.py --data-root ../data/evs_mot-test --output-dir outputs/test_predictions`

## Etap 2 - Ulepszenie stabilnosci torow
**Cel**: obnizyc `FN` i `IDSW`.
**Status**: ✅ WYKONANY

Zadania:
- dodanie Kalman filter do predykcji polozenia,
- koszt asocjacji: kombinacja IoU + dystans centroidow (z normalizacja),
- gating (odrzucanie par o zbyt duzym koszcie).

Deliverable:
- poprawa metryk wzgledem baseline,
- porownanie "przed/po" dla min. 2 sekwencji.

Mikroetapy zrealizowane:
- [x] Dodany filtr Kalmana (model stalej predkosci) do predykcji bbox (`source/mot/kalman.py`).
- [x] Asocjacja oparta o koszt laczony: `w_iou * (1-IoU) + w_center * dist_center_norm` (`source/mot/association.py`).
- [x] Dodany gating na pary track-detection: minimalne IoU, maksymalny dystans centroidow, maksymalny koszt.
- [x] Rozszerzona konfiguracja trackera o parametry kosztu i gating (`source/config/default.yaml`).
- [x] Ewaluacja po zmianach i zapis wynikow do `outputs/logs/train_eval_stage2_summary.json`.

Porownanie metryk (PRZED -> PO):
- Overall MOTA: `0.4974 -> 0.4989` (poprawa)
- Overall FN: `33180 -> 33034` (mniej o 146)
- Overall IDSW: `131 -> 107` (mniej o 24)
- Overall FP: `3198 -> 3261` (wzrost o 63; akceptowalny przy spadku FN/IDSW)

Porownanie per-sequence (min. 2):
- `MOT_03`:
  - MOTA: `0.4687 -> 0.4746`
  - FN: `3580 -> 3553`
  - IDSW: `23 -> 13`
- `MOT_05`:
  - MOTA: `0.4591 -> 0.4596`
  - FN: `5532 -> 5483`
  - IDSW: `108 -> 92`

Uruchamianie Etapu 2:
- `cd source`
- `pip install -r requirements.txt`
- `PYTHONPATH=. python scripts/run_train.py --data-root ../data/evs_mot-train --output-dir outputs/train_predictions_stage2`
- `PYTHONPATH=. python scripts/evaluate_train.py --pred-dir outputs/train_predictions_stage2 --gt-root ../data/evs_mot-train --output-file outputs/logs/train_eval_stage2_summary.json`

## Etap 3 - Ograniczanie identity switchy
**Cel**: poprawa stabilnosci ID.
**Status**: ⏳ W TRAKCIE

Zadania:
- analiza fragmentow z najwieksza liczba `IDSW`,
- tuning wag kosztu asocjacji,
- ewentualnie dodanie cech wygladu (prosty embedding/cosine), jesli czas pozwoli.

Deliverable:
- wersja "best config" dla train.

## Etap 4 - Ewaluacja i przygotowanie rozwiazania
**Cel**: profesjonalna finalizacja projektu.
**Status**: ⏳ NIE ROZPOCZETY

Zadania:
- ewaluacja TrackEval na train i interpretacja wynikow,
- uruchomienie inferencji na test (`MOT_01`, `MOT_06`, `MOT_07`),
- walidacja formatu plikow wynikowych,
- przygotowanie `submission/` (`data/` + `source/`) i archiwum `.zip`.

Deliverable:
- gotowa paczka do oddania,
- dokumentacja uruchomienia i reprodukowalnosci.

---

## 5. Hiperparametry i strojenie (co i jak stroic)

Kluczowe parametry:
- `det_conf_threshold` - usuwa slabe detekcje (trade-off FP/FN),
- `iou_match_threshold` - rygor przypisania tor-detekcja,
- `max_age` - ile klatek tor moze "przezyc" bez dopasowania,
- `min_hits` - kiedy tor staje sie potwierdzony,
- wagi kosztu (np. `w_iou`, `w_center`, opcjonalnie `w_app`).

Metoda strojenia:
1. Ustalic mocny baseline.
2. Tuning pojedynczego parametru przy zamrozonych pozostalych.
3. Zapisywac wyniki eksperymentow (konfiguracja -> metryki -> komentarz).
4. Nie wybierac konfiguracji po pojedynczej sekwencji - patrzec na srednia i stabilnosc.

---

## 6. Metryki i kryteria techniczne

Priorytet metryk:
1. `MOTA` (glowna metryka),
2. skladowe: `FN`, `FP`, `IDSW` (diagnoza przyczyn).

Kryteria jakosci implementacji:
- reprodukowalnosc (stale seedy, wersjonowane configi),
- modularnosc (separacja I/O, asocjacji, modelu ruchu),
- czytelny logging (parametry, czasy, metryki),
- pelna automatyzacja uruchomienia (jedna komenda dla train/test/eval).

---

## 7. Ryzyka projektowe i plan minimalizacji

1. **Nadmierna liczba FP przez slabe detekcje**
   - rozwiazanie: tuning `det_conf_threshold`, ewentualna filtracja po rozmiarze bbox.

2. **Wysokie IDSW przy occlusions i skrzyzowaniach trajektorii**
   - rozwiazanie: lepszy koszt asocjacji, Kalman + gating, delikatne zwiekszenie `max_age`.

3. **Przeuczenie konfiguracji do pojedynczych sekwencji train**
   - rozwiazanie: tuning na wielu sekwencjach i porownania przekrojowe.

4. **Bledy formalne formatu wyjsciowego**
   - rozwiazanie: skrypt walidacyjny formatu uruchamiany przed pakowaniem.

---

## 8. Harmonogram (propozycja)

- **Dzien 1**: EDA + wizualizacje + przygotowanie parserow.
- **Dzien 2-3**: baseline IoU + Hungarian + zapis wynikow.
- **Dzien 4-5**: Kalman + ulepszona asocjacja + tuning.
- **Dzien 6**: analiza bledow, poprawki IDSW, finalny tuning.
- **Dzien 7**: inferencja na test, walidacja formatu, paczka `submission.zip`, dokumentacja.

Jesli czasu jest mniej: priorytetem jest solidny baseline + Kalman + poprawna ewaluacja i paczkowanie.

---

## 9. Definicja "Done"

Projekt uznaje sie za gotowy, gdy:
- [ ] Algorytm dziala dla wszystkich wymaganych sekwencji.
- [ ] Wyniki sa zapisane w poprawnym formacie MOT.
- [ ] TrackEval uruchamia sie bez bledow i zwraca metryki.
- [ ] Istnieje wybrana, uzasadniona konfiguracja finalna.
- [ ] Przygotowano `submission/data/*.txt`.
- [ ] Przygotowano `submission/source/*` z kodem i instrukcja uruchomienia.
- [ ] Powstal finalny `submission.zip` gotowy do wyslania.

---

## 10. Rekomendacja koncowa

Najbardziej profesjonalne podejscie do tego projektu to:
1. szybko zbudowac dzialajacy baseline,
2. mierzyc kazda zmiane metrykami (nie "na oko"),
3. priorytetyzowac redukcje `IDSW` i `FN`,
4. utrzymac powtarzalny pipeline od treningu do paczki submission.

W praktyce to daje najlepszy kompromis miedzy jakoscia wyniku, kontrola ryzyka i czasem realizacji.

---

## 11. Status realizacji (live)

- [x] Etap 0 - Przygotowanie i interpretacja danych.
- [x] Etap 1 - Baseline tracking (MVP).
- [x] Etap 2 - Ulepszenie stabilnosci torow.
- [ ] Etap 3 - Ograniczanie identity switchy.
- [ ] Etap 4 - Ewaluacja i przygotowanie rozwiazania.

Zasada prowadzenia projektu:
- po kazdym zakonczonym etapie aktualizujemy ten plan,
- jesli pojawiaja sie zadania posrednie, dopisujemy je jako mikroetapy pod odpowiednim etapem,
- po kazdym etapie dokumentujemy jak uruchomic kod i jakie artefakty powinny sie pojawic.
