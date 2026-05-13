"""Generate a standalone HTML pitch-deck presentation for the MOT project.

Usage (from repo root):
    python -m scripts.generate_presentation_html
    python -m scripts.generate_presentation_html --output PRESENTATION.html
    python -m scripts.generate_presentation_html --pdf               # also saves PDF via playwright
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Slide definitions
# Each dict: kicker, title, subtitle (optional), body (raw HTML string)
# ---------------------------------------------------------------------------

SLIDES: list[dict] = [
    # 01 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Zaawansowane Algorytmy Wizyjne",
        "title": "Multi-Object<br>Tracking",
        "subtitle": "Mateusz Bielówka & Karol Bystrek & Patryk Chamera",
        "body": """
        """,
    },

    # 02 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Problem",
        "title": "Tracking-by-detection",
        "body": """
        <p class="lead">Dostajemy gotowe <code>bbox + confidence</code>
        na każdej klatce.<br> Zadanie: przypisać <em>stabilne ID</em> tej samej osobie
        przez całą sekwencję.</p>
        <div class="grid two mt32">
          <div class="card">
            <b>Wejście</b>
            <span>per-frame detekcje z kwalifikacją pewności</span>
          </div>
          <div class="card">
            <b>Wyjście</b>
            <span>trajektorie — każdy bbox otagowany trwałym <code>id</code></span>
          </div>
          <div class="card warn">
            <b>Zasłonięcia &amp; luki</b>
            <span>osoba może zniknąć na kilka klatek i wrócić</span>
          </div>
          <div class="card warn">
            <b>Identity switch</b>
            <span>skrzyżowanie torów → błędna zamiana ID</span>
          </div>
        </div>
        """,
    },

    # 03 ─────────────────────────────────────────────────────────────────────
    # {
    #     "kicker": "Dane",
    #     "title": "Format MOT",
    #     "body": """
    #     <div class="split mt24">
    #       <div>
    #         <p class="mono label mb8">det/det.txt — wejście</p>
    #         <pre>&lt;frame&gt;, -1, bb_left, bb_top, bb_width, bb_height, conf</pre>
    #         <p class="note">frame 1-based · conf ∈ [0, 1] · współrzędne float</p>
    #       </div>
    #       <div>
    #         <p class="mono label mb8">MOT_XX.txt — wyjście</p>
    #         <pre>&lt;frame&gt;, id, bb_left, bb_top, bb_width, bb_height, 1, -1, -1, -1</pre>
    #         <p class="note">trailing <code>1,-1,-1,-1</code> stałe · id unikalne w sekwencji</p>
    #       </div>
    #     </div>
    #     <div class="info-row mt28">
    #       <div class="pill">Train: <code>gt/gt.txt</code> dostępne</div>
    #       <div class="pill">Test: <strong>brak ground truth</strong></div>
    #       <div class="pill">GT filtr: <code>eval_flag==1 &amp;&amp; class==1</code></div>
    #     </div>
    #     """,
    # },

    # 04 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Metryka",
        "title": "MOTA",
        "body": """
        <div class="formula-block mt20">
          MOTA = 1 − <span class="frac"><span class="num">FN + FP + IDSW</span><span class="den">GT</span></span>
        </div>
        <div class="grid four mt32">
          <div class="metric-card fn">
            <strong>FN</strong>
            <span>False Negative<br><em>zgubiony obiekt GT</em></span>
          </div>
          <div class="metric-card fp">
            <strong>FP</strong>
            <span>False Positive<br><em>duch bez pokrycia w GT</em></span>
          </div>
          <div class="metric-card sw">
            <strong>IDSW</strong>
            <span>ID Switch<br><em>zmiana tożsamości</em></span>
          </div>
          <div class="metric-card gt">
            <strong>GT</strong>
            <span>Ground Truth<br><em>suma bboxów referencyjnych</em></span>
          </div>
        </div>
        """,
    },

    # 05 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Tak to zrobiliśmy",
        "title": "Budowa trackera",
        "body": """
        <ol class="timeline mt28">
          <li>
            <span class="mono step">01</span>
            <div>
              <b>Baseline — IoU + algorytm węgierski</b>
              <span>Każda detekcja przypisana do najlepszego toru globalnie (minimalizacja sumy kosztów).
              IoU mierzy stopień nakładania bboxów — punkt startowy każdego trackera TBD.</span>
            </div>
          </li>
          <li>
            <span class="mono step">02</span>
            <div>
              <b>Filtr Kalmana — predykcja ruchu</b>
              <span>8-wymiarowy model stałej prędkości (pozycja + prędkość) przewiduje bbox
              na każdej klatce. Tor pozostaje aktywny nawet gdy detekcja chwilowo zanika
              (zasłonięcie, niska jakość kadru).</span>
            </div>
          </li>
          <li>
            <span class="mono step">03</span>
            <div>
              <b>Koszt asocjacji — samo IoU</b>
              <span>Macierz kosztów oparta wyłącznie na <code>1 − IoU</code>.
              Algorytm węgierski minimalizuje sumaryczny koszt przypisania torów do detekcji.</span>
            </div>
          </li>
          <li>
            <span class="mono step">04</span>
            <div>
              <b>Gating — dwie niezależne bramy akceptacji</b>
              <span>Algorytm węgierski zawsze zwróci pary — nawet bezsensowne, gdy nie ma lepszych.
              Dwie bramy (min. IoU, max. koszt łączny) odrzucają
              przypisania poniżej progu jakości.</span>
            </div>
          </li>
          <li>
            <span class="mono step">05</span>
            <div>
              <b>Dwuetapowa asocjacja — high-conf i low-conf</b>
              <span>Podejście inspirowane ByteTrack: pewne detekcje (Etap 1) decydują o nowych torach,
              słabe (Etap 2) ratują <em>już potwierdzone</em> tory bez ryzyka tworzenia FP.
              Rozdzielenie ról zmniejsza jednocześnie FN i FP.</span>
            </div>
          </li>
          <li>
            <span class="mono step">06</span>
            <div>
              <b>Grid search — systematyczna optymalizacja hiperparametrów</b>
              <span>Kartezjański iloczyn 6 parametrów (648 kombinacji), każda uruchomiona
              na całym zbiorze treningowym. Kryterium: MOTA z lokalnego ewaluatora.
              Deterministyczny, w pełni odtwarzalny.</span>
            </div>
          </li>
        </ol>
        """,
    },

    # 06 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Komponent",
        "title": "Filtr Kalmana",
        "body": """
        <p class="lead" style="font-size:16px; margin-bottom:18px">
          Rekurencyjny estymator stanu — łączy model ruchu (predykcję) z obserwacją (detekcją),
          ważąc je według ich niepewności. Dzięki predykcji tracker utrzymuje aktywny tor przez
          całe <code>max_age</code> klatek bez detekcji, zamiast natychmiast tracić obiekt.
          Model <b>stałej prędkości (CV)</b> — prosty, ale wystarczający dla ruchu pieszych.
        </p>
        <div class="split">
          <div>
            <p class="mono label mb8">wektor stanu (8D)</p>
            <pre>x = [x, y, w, h, vx, vy, vw, vh]</pre>
            <p class="note">Pozycja lewego górnego rogu + rozmiar bbox + ich prędkości.
            Δt = 1 klatka.</p>
            <p class="mono label mb8 mt20">pomiar (4D)</p>
            <pre>z = [x, y, w, h]   ← tylko bbox z detekcji</pre>
          </div>
          <div>
            <p class="mono label mb8">macierze modelu</p>
            <pre>F  → I₈ + [pos ← pos + vel] na off-diag
H  → I₄ | 0₄   (obserwujemy tylko pozycję)
Q  → diag: pos×0.05,  vel×0.01
R  → I₄        (szum pomiaru)
P₀ → 10·I₈    (duża startowa niepewność)</pre>
          </div>
        </div>
        <div class="callout mt20">
          Predykcja utrzymuje tor aktywny przez <code>max_age = 35</code> klatek bez detekcji.
          Po każdym kroku: clamp <code>w, h ≥ 1</code> (zapobiega degeneracji bbox).
        </div>
        """,
    },

    # 07 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Komponent",
        "title": "Koszt asocjacji",
        "body": """
        <p class="lead" style="font-size:16px; margin-bottom:16px">
          Algorytm węgierski wyznacza <b>globalne optymalne przypisanie
          one-to-one</b> torów do detekcji — minimalizuje sumaryczny koszt, nie każdą parę z osobna.
          Koszt oparty wyłącznie na <em>nakładaniu bboxów</em> (IoU) — prosty, szybki i skuteczny.
        </p>
        <div class="formula-block formula-sm">
          cost = 1 − IoU
        </div>
        <div class="split mt20">
          <div>
            <p class="mono label mb8">macierz kosztów</p>
            <pre>for each (track, det):
  cost[i, j] = 1.0 - iou(track_box, det_box)</pre>
            <p class="note">IoU = 0 gdy brak nakładania → koszt = 1 (maksymalny).</p>
          </div>
          <div>
            <p class="mono label mb8">dwie bramy (gating) — po przypisaniu</p>
            <div class="gate-list">
              <div class="gate"><code>IoU ≥ 0.15</code> — minimalne nakładanie</div>
              <div class="gate"><code>cost ≤ 0.92</code> — łączny koszt</div>
            </div>
            <p class="note">Pary poniżej progów → odrzucone jako unmatched, mimo wyniku Węgierskiego.</p>
          </div>
        </div>
        """,
    },

    # 08 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Kluczowy pomysł",
        "title": "Asocjacja dwuetapowa",
        "body": """
        <p class="lead" style="font-size:16px; margin-bottom:16px">
          Podejście inspirowane <b>ByteTrack</b>: detekcje dzielimy według pewności i traktujemy różnie.
          Pewne detekcje budują nowe tory i wykonują główne przypisanie; słabe służą <em>wyłącznie</em>
          do ratowania już istniejących, potwierdzonych torów — bez ryzyka tworzenia FP z szumu detektora.
        </p>
        <div class="two-stage">
          <div class="stage stage1">
            <p class="mono label">Etap 1 — high conf</p>
            <p class="stage-thresh">conf ≥ <strong>0.30</strong></p>
            <ul>
              <li>Wszystkie aktywne tory vs detekcje high-conf</li>
              <li>Koszt mieszany: <code>w·(1−IoU) + w·d_norm</code></li>
              <li>Trzy bramy: IoU, dystans centrum, koszt łączny</li>
              <li>Niesparowane high dety → <strong>nowe tory</strong></li>
            </ul>
          </div>
          <div class="stage-arrow">→</div>
          <div class="stage stage2">
            <p class="mono label">Etap 2 — low conf recovery</p>
            <p class="stage-thresh">0.05 ≤ conf &lt; 0.30</p>
            <ul>
              <li>Tylko <em>potwierdzone</em> tory (hits ≥ 3) — bez nowych</li>
              <li>Tylko IoU (brama: <code>IoU ≥ 0.60</code>, rygorystyczna)</li>
              <li>Niesparowane low dety → <strong>odrzucone</strong> (nie tworzą torów)</li>
            </ul>
          </div>
        </div>
        <p class="note mt16">Rozdzielenie ról: Etap 1 zarządza nowymi bytami (FP/FN),
        Etap 2 ratuje zasłonięte / przejściowo słabe obiekty bez ryzyka tworzenia duchów.</p>
        """,
    },

    # 09 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Cykl życia",
        "title": "Init · Confirm · Delete",
        "body": """
        <p class="lead" style="font-size:16px; margin-bottom:20px">
          Zarządzanie cyklem życia to filtr jakości torów: nowe tory muszą
          „udowodnić" istnienie przez kilka kolejnych klatek zanim pojawią się w wynikach
          (eliminacja FP z chwilowych błędów detekcji), a zbyt długo niewidziane tory
          są usuwane (uwalniamy ID, eliminujemy duchy).
        </p>
        <div class="lifecycle">
          <div class="lc-node">
            <span class="mono label">INIT</span>
            <p>Nowa high-conf detekcja bez pary tworzy tor.<br>
            <code>hits = 1</code>, <code>time_since_update = 0</code>.<br>
            Nie emitowany jeszcze w wynikach.</p>
          </div>
          <div class="lc-arrow">→</div>
          <div class="lc-node accent-node">
            <span class="mono label">CONFIRMED</span>
            <p>Tor pojawia się w wynikach gdy <code>hits ≥ min_hits (2)</code>.<br>
            Kwalifikuje do low-conf rescue (Etap 2) po <code>hits ≥ 3</code>.<br>
            Każda pomyślna asocjacja: <code>hits += 1</code>.</p>
          </div>
          <div class="lc-arrow">→</div>
          <div class="lc-node">
            <span class="mono label">DELETED</span>
            <p>Tor usunięty gdy <code>time_since_update &gt; max_age (35)</code>.<br>
            Filtr Kalmana „trzyma" pozycję przez 35 klatek bez detekcji.<br>
            Zwolnione ID nie są ponownie używane.</p>
          </div>
        </div>
        """,
    },

    # 10 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Optymalizacja",
        "title": "Grid Search",
        "body": """
        <p class="lead">Parametry wpływają na wzajemne kompromisy FN ↔ FP ↔ IDSW.
        Wybrałem przeszukanie siatki (vs random search) — przestrzeń jest mała, każdy wymiar mały,
        wynik deterministyczny.</p>
        <div class="grid-params mt28">
          <div class="gp-header mono">parametr</div>
          <div class="gp-header mono">wartości</div>
          <div class="gp-header mono">rola</div>

          <div class="gp-name"><code>det_conf_threshold</code></div>
          <div class="gp-vals"><code>[0.2, 0.25, 0.3]</code></div>
          <div class="gp-desc">próg high-conf → bilans FP/FN</div>

          <div class="gp-name"><code>det_low_conf_threshold</code></div>
          <div class="gp-vals"><code>[0.05, 0.1, 0.15]</code></div>
          <div class="gp-desc">dolny próg Etapu 2</div>

          <div class="gp-name"><code>iou_match_threshold</code></div>
          <div class="gp-vals"><code>[0.15, 0.2, 0.25]</code></div>
          <div class="gp-desc">rygor bramy IoU Etap 1</div>

          <div class="gp-name"><code>iou_match_threshold_low</code></div>
          <div class="gp-vals"><code>[0.4, 0.5, 0.6]</code></div>
          <div class="gp-desc">rygor bramy IoU Etap 2</div>

          <div class="gp-name"><code>max_age</code></div>
          <div class="gp-vals"><code>[30, 35]</code></div>
          <div class="gp-desc">przeżywalność toru bez detekcji</div>

          <div class="gp-name"><code>min_hits</code></div>
          <div class="gp-vals"><code>[2, 3]</code></div>
          <div class="gp-desc">klatki do potwierdzenia emisji</div>
        </div>
        <p class="note mt16">3×3×3×3×2×2 = <strong>648 kombinacji</strong>.
        Kryterium: <code>overall["mota"]</code> z lokalnego ewaluatora.</p>
        """,
    },

    # 11 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Najlepsza konfiguracja",
        "title": "config/default.yaml",
        "body": """
        <div class="config-grid mt24">
          <div class="cfg-group">
            <p class="mono label mb8">progi pewności</p>
            <div class="cfg-row"><code>det_conf_threshold</code><strong>0.30</strong></div>
            <div class="cfg-row"><code>det_low_conf_threshold</code><strong>0.05</strong></div>
          </div>
          <div class="cfg-group">
            <p class="mono label mb8">asocjacja — etap 1</p>
            <div class="cfg-row"><code>iou_match_threshold</code><strong>0.15</strong></div>
            <div class="cfg-row"><code>max_match_cost</code><strong>0.92</strong></div>
          </div>
          <div class="cfg-group">
            <p class="mono label mb8">asocjacja — etap 2</p>
            <div class="cfg-row"><code>iou_match_threshold_low</code><strong>0.60</strong></div>
          </div>
          <div class="cfg-group">
            <p class="mono label mb8">cykl życia</p>
            <div class="cfg-row"><code>max_age</code><strong>35</strong></div>
            <div class="cfg-row"><code>min_hits</code><strong>2</strong></div>
            <div class="cfg-row"><code>save_only_confirmed</code><strong>true</strong></div>
          </div>
        </div>
        """,
    },

    # 12 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Wyniki — train",
        "title": "MOTA 50.3%",
        "body": """
        <div class="results-top mt20">
          <div class="big-result">
            <span class="mono label">Overall MOTA</span>
            <strong class="big-num accent">50.3<span style="font-size:.5em">%</span></strong>
          </div>
          <div class="err-pills">
            <div class="err-pill fn-pill"><span class="mono">FN</span><strong>32 400</strong><em>dominujący błąd</em></div>
            <div class="err-pill fp-pill"><span class="mono">FP</span><strong>3 553</strong></div>
            <div class="err-pill sw-pill"><span class="mono">IDSW</span><strong>140</strong></div>
            <div class="err-pill gt-pill"><span class="mono">GT</span><strong>72 638</strong></div>
          </div>
        </div>
        <div class="seq-table mt28">
          <div class="seq-header mono">sekwencja</div>
          <div class="seq-header mono">MOTA</div>
          <div class="seq-header mono">FN</div>
          <div class="seq-header mono">FP</div>
          <div class="seq-header mono">IDSW</div>

          <div>MOT_02</div><div class="accent-text">51.2%</div><div>21 344</div><div>1 853</div><div>0</div>
          <div>MOT_03</div><div>49.1%</div><div>3 407</div><div>104</div><div>11</div>
          <div>MOT_04</div><div class="accent-text">55.3%</div><div>2 363</div><div>12</div><div>3</div>
          <div>MOT_05</div><div class="warn-text">45.5%</div><div>5 286</div><div>1 584</div><div>126</div>
        </div>
        <p class="note mt16">FN dominuje — tracker traci osoby zasłonięte lub słabo widoczne.
        MOT_05 ma więcej FP i IDSW: gęstsza scena.</p>
        """,
    },

    # 13 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Wyniki — test (Codabench)",
        "title": "MOTA 42.4%",
        "body": """
        <div class="results-top mt20">
          <div class="big-result">
            <span class="mono label">Overall MOTA</span>
            <strong class="big-num accent">42.4<span style="font-size:.5em">%</span></strong>
          </div>
          <div class="err-pills">
            <div class="err-pill fn-pill"><span class="mono">FN</span><strong>21 069</strong><em>dominujący błąd</em></div>
            <div class="err-pill fp-pill"><span class="mono">FP</span><strong>1 344</strong></div>
            <div class="err-pill sw-pill"><span class="mono">IDSW</span><strong>421</strong></div>
          </div>
        </div>
        <div class="seq-table mt28">
          <div class="seq-header mono">sekwencja</div>
          <div class="seq-header mono">MOTA</div>
          <div class="seq-header mono">FN</div>
          <div class="seq-header mono">FP</div>
          <div class="seq-header mono">IDSW</div>

          <div>MOT_01</div><div class="warn-text">32.7%</div><div>12 200</div><div>241</div><div>71</div>
          <div>MOT_06</div><div class="accent-text">56.5%</div><div>3 848</div><div>195</div><div>57</div>
          <div>MOT_07</div><div>46.6%</div><div>5 021</div><div>908</div><div>293</div>
        </div>
        <p class="note mt16">MOT_01 — scena o wysokiej gęstości, tracker traci więcej obiektów (FN dominuje).
        MOT_07 — wyraźnie więcej FP i IDSW: dynamiczna scena, częste skrzyżowania torów.</p>
        """,
    },

    # 14 ─────────────────────────────────────────────────────────────────────
    {
        "kicker": "Koniec",
        "title": "Dziękujemy za uwagę",
        "subtitle": "Mateusz Bielówka & Karol Bystrek & Patryk Chamera",
        "body": "",
    },
]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg:          #faf8f4;
  --panel:       #ffffff;
  --ink:         #19170f;
  --muted:       #6b6660;
  --line:        #e6dfd4;
  --accent:      #1d4ed8;
  --accent-soft: #eff4ff;
  --green:       #0d7a56;
  --green-soft:  #edfaf4;
  --amber:       #92400e;
  --amber-soft:  #fffbeb;
  --red-soft:    #fef2f2;
  --red:         #991b1b;
  --serif:       'Playfair Display', Georgia, 'Times New Roman', serif;
  --mono:        'JetBrains Mono', 'Fira Code', 'SFMono-Regular', Consolas, monospace;
  --sans:        'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-snap-type: y mandatory; }

body {
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
}

/* ── slide shell ──────────────────────────────────────────────── */
.slide {
  min-height: 100vh;
  padding: 56px 72px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  scroll-snap-align: start;
  position: relative;
  overflow: hidden;
}

.slide::before {
  content: "";
  position: absolute;
  inset: 24px;
  border: 1px solid var(--line);
  border-radius: 28px;
  pointer-events: none;
}

.slide-num {
  position: absolute;
  right: 52px;
  bottom: 40px;
  color: #c4bbb0;
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: .1em;
}

.content {
  max-width: 1100px;
  margin: 0 auto;
  width: 100%;
  position: relative;
  z-index: 1;
}

/* ── typography ───────────────────────────────────────────────── */
.kicker {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .22em;
  color: var(--accent);
  margin-bottom: 18px;
}

h1 {
  font-family: var(--serif);
  font-size: clamp(44px, 6.5vw, 82px);
  font-weight: 900;
  line-height: .95;
  letter-spacing: -.04em;
  margin-bottom: 20px;
}

.lead {
  font-size: 19px;
  line-height: 1.55;
  color: var(--muted);
  max-width: 820px;
}

p.note {
  font-size: 14px;
  color: var(--muted);
  line-height: 1.5;
}

code {
  font-family: var(--mono);
  font-size: .88em;
  background: #f0ede8;
  border-radius: 4px;
  padding: 1px 5px;
}

pre {
  font-family: var(--mono);
  background: #1a1814;
  color: #e8e4dc;
  padding: 16px 20px;
  border-radius: 14px;
  font-size: 14px;
  line-height: 1.55;
  overflow-x: auto;
  tab-size: 2;
}

.mono { font-family: var(--mono); }
.label {
  font-family: var(--mono);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .16em;
  color: var(--muted);
}

.mb8  { margin-bottom: 8px; }
.mt16 { margin-top: 16px; }
.mt20 { margin-top: 20px; }
.mt24 { margin-top: 24px; }
.mt28 { margin-top: 28px; }
.mt32 { margin-top: 32px; }

/* ── grid helpers ─────────────────────────────────────────────── */
.grid { display: grid; gap: 16px; }
.two   { grid-template-columns: repeat(2, 1fr); }
.three { grid-template-columns: repeat(3, 1fr); }
.four  { grid-template-columns: repeat(4, 1fr); }

/* ── card ─────────────────────────────────────────────────────── */
.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 22px 24px;
  font-size: 17px;
}
.card b { display: block; font-size: 18px; margin-bottom: 6px; }
.card span { display: block; color: var(--muted); font-size: 15px; line-height: 1.4; }
.card.warn { border-color: #fcd34d; background: var(--amber-soft); }
.card.warn b { color: var(--amber); }

/* ── split ────────────────────────────────────────────────────── */
.split {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

/* ── callout ──────────────────────────────────────────────────── */
.callout {
  background: var(--green-soft);
  border: 1px solid #b2e8d4;
  border-radius: 18px;
  padding: 18px 22px;
  font-size: 16px;
  color: var(--green);
  line-height: 1.5;
}

/* ── info pills ───────────────────────────────────────────────── */
.info-row { display: flex; gap: 12px; flex-wrap: wrap; }
.pill {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 100px;
  padding: 8px 18px;
  font-size: 14px;
  color: var(--muted);
}
.pill strong { color: var(--ink); }

/* ── hero row (slide 01) ──────────────────────────────────────── */
.hero-row {
  display: flex;
  gap: 40px;
  margin-top: 44px;
  flex-wrap: wrap;
}
.hero-stat {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.big-num {
  font-family: var(--serif);
  font-size: 72px;
  font-weight: 900;
  letter-spacing: -.06em;
  line-height: 1;
  color: var(--ink);
}
.big-num.accent { color: var(--accent); }
.accent-text { color: var(--accent); font-weight: 600; }
.warn-text   { color: var(--amber);  font-weight: 600; }

/* ── formula block ────────────────────────────────────────────── */
.formula-block {
  font-family: var(--serif);
  font-size: 36px;
  font-weight: 700;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 24px 32px;
  display: inline-block;
  letter-spacing: -.02em;
}
.formula-block em { font-style: normal; color: var(--accent); }
.formula-sm { font-size: 26px; }
.frac {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  vertical-align: middle;
  margin: 0 4px;
}
.frac .num {
  border-bottom: 2px solid var(--ink);
  padding-bottom: 3px;
  font-size: .72em;
  white-space: nowrap;
}
.frac .den { font-size: .72em; padding-top: 3px; }

/* ── metric cards ─────────────────────────────────────────────── */
.metric-card {
  border-radius: 20px;
  padding: 22px;
  text-align: center;
}
.metric-card strong {
  display: block;
  font-family: var(--mono);
  font-size: 30px;
  font-weight: 700;
  margin-bottom: 8px;
}
.metric-card span { font-size: 14px; color: var(--muted); line-height: 1.4; }
.metric-card em { font-style: normal; display: block; font-size: 12px; margin-top: 4px; }
.fn { background: var(--red-soft);   border: 1px solid #fecaca; }
.fn strong { color: var(--red); }
.fp { background: var(--amber-soft); border: 1px solid #fcd34d; }
.fp strong { color: var(--amber); }
.sw { background: #faf5ff; border: 1px solid #e9d5ff; }
.sw strong { color: #6b21a8; }
.gt { background: var(--accent-soft); border: 1px solid #bfdbfe; }
.gt strong { color: var(--accent); }

/* ── timeline ─────────────────────────────────────────────────── */
.timeline {
  list-style: none;
  display: grid;
  gap: 10px;
}
.timeline li {
  display: flex;
  align-items: flex-start;
  gap: 20px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 16px 20px;
}
.timeline .step {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--accent);
  background: var(--accent-soft);
  border-radius: 8px;
  padding: 4px 8px;
  white-space: nowrap;
  margin-top: 2px;
  flex-shrink: 0;
}
.timeline li b { display: block; font-size: 16px; margin-bottom: 3px; }
.timeline li span { color: var(--muted); font-size: 14px; line-height: 1.4; }

/* ── gate list ────────────────────────────────────────────────── */
.gate-list { display: grid; gap: 8px; margin-bottom: 12px; }
.gate {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 14px;
}

/* ── two-stage ────────────────────────────────────────────────── */
.two-stage {
  display: grid;
  grid-template-columns: 1fr 40px 1fr;
  gap: 12px;
  align-items: center;
}
.stage {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 24px 26px;
}
.stage1 { border-color: #bfdbfe; background: var(--accent-soft); }
.stage2 { border-color: #b2e8d4; background: var(--green-soft); }
.stage-thresh {
  font-family: var(--mono);
  font-size: 20px;
  font-weight: 700;
  margin: 8px 0 14px;
  color: var(--ink);
}
.stage ul { padding-left: 16px; }
.stage li { font-size: 14px; color: var(--muted); margin-bottom: 6px; line-height: 1.4; }
.stage li strong { color: var(--ink); }
.stage-arrow {
  text-align: center;
  font-size: 28px;
  color: var(--muted);
}

/* ── lifecycle ────────────────────────────────────────────────── */
.lifecycle {
  display: grid;
  grid-template-columns: 1fr 32px 1fr 32px 1fr;
  gap: 8px;
  align-items: center;
}
.lc-node {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 22px;
  min-height: 130px;
}
.lc-node .label { margin-bottom: 10px; }
.lc-node p { font-size: 14px; color: var(--muted); line-height: 1.5; }
.lc-node p code { font-size: 12px; }
.accent-node { border-color: #bfdbfe; background: var(--accent-soft); }
.lc-arrow { text-align: center; font-size: 22px; color: var(--muted); }

/* ── grid search param table ──────────────────────────────────── */
.grid-params {
  display: grid;
  grid-template-columns: auto auto 1fr;
  gap: 6px 20px;
  align-items: center;
}
.gp-header {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .16em;
  color: var(--muted);
  padding-bottom: 6px;
  border-bottom: 1px solid var(--line);
}
.gp-name  { padding: 8px 0; }
.gp-vals  { padding: 8px 0; }
.gp-desc  { font-size: 14px; color: var(--muted); padding: 8px 0; }
.grid-params > div:not(.gp-header) { border-bottom: 1px solid #f0ede8; }

/* ── config grid ──────────────────────────────────────────────── */
.config-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
}
.cfg-group {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 20px 22px;
}
.cfg-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 7px 0;
  border-bottom: 1px solid #f0ede8;
  font-size: 14px;
}
.cfg-row:last-child { border-bottom: none; }
.cfg-row strong {
  font-family: var(--mono);
  font-weight: 700;
  color: var(--accent);
}

/* ── results ──────────────────────────────────────────────────── */
.results-top {
  display: flex;
  gap: 40px;
  align-items: flex-start;
  flex-wrap: wrap;
}
.big-result { display: flex; flex-direction: column; gap: 4px; }
.err-pills {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  align-items: flex-end;
  padding-bottom: 8px;
}
.err-pill {
  border-radius: 16px;
  padding: 12px 18px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 90px;
}
.err-pill span { font-size: 11px; }
.err-pill strong { font-family: var(--serif); font-size: 26px; font-weight: 700; }
.err-pill em { font-style: normal; font-size: 11px; }
.fn-pill { background: var(--red-soft);   border: 1px solid #fecaca; color: var(--red); }
.fp-pill { background: var(--amber-soft); border: 1px solid #fcd34d; color: var(--amber); }
.sw-pill { background: #faf5ff;           border: 1px solid #e9d5ff; color: #6b21a8; }
.gt-pill { background: var(--accent-soft);border: 1px solid #bfdbfe; color: var(--accent); }

.seq-table {
  display: grid;
  grid-template-columns: auto repeat(4, 1fr);
  gap: 6px 20px;
  font-size: 15px;
  align-items: center;
}
.seq-header {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .14em;
  color: var(--muted);
  padding-bottom: 6px;
  border-bottom: 1px solid var(--line);
}
.seq-table > div:not(.seq-header) {
  padding: 6px 0;
  border-bottom: 1px solid #f0ede8;
}

/* ── print / PDF ──────────────────────────────────────────────── */
@media print {
  html { scroll-snap-type: none; }
  body { background: white; }
  .slide {
    width: 100vw;
    height: 100vh;
    min-height: 0;
    break-after: page;
    page-break-after: always;
    padding: 48px 64px;
  }
}
@page {
  size: 16in 9in landscape;
  margin: 0;
}
"""

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

TEMPLATE = """\
<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-Object Tracking — Pitch Deck</title>
  <style>{css}</style>
</head>
<body>
  <main>{slides}</main>
</body>
</html>
"""


def _render_slide(slide: dict, index: int, total: int) -> str:
    kicker   = slide["kicker"]
    title    = slide["title"]
    subtitle = slide.get("subtitle", "")
    body     = slide.get("body", "")
    sub_html = f'<p class="lead" style="margin-bottom:0">{subtitle}</p>' if subtitle else ""
    return (
        f'<section class="slide">'
        f'<div class="content">'
        f'<div class="kicker">{kicker}</div>'
        f'<h1>{title}</h1>'
        f'{sub_html}'
        f'{body}'
        f'</div>'
        f'<span class="slide-num">{index:02d} / {total:02d}</span>'
        f'</section>'
    )


def build_html() -> str:
    total  = len(SLIDES)
    slides = "\n".join(_render_slide(s, i, total) for i, s in enumerate(SLIDES, 1))
    return TEMPLATE.format(css=CSS, slides=slides)


# ---------------------------------------------------------------------------
# PDF via playwright
# ---------------------------------------------------------------------------

def _export_pdf(html_path: Path, pdf_path: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        print("playwright not installed — skipping PDF export.")
        print("  Install: pip install playwright && playwright install chromium")
        return

    print(f"Generating PDF via playwright → {pdf_path}")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page    = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(pdf_path),
            print_background=True,
            width="16in",
            height="9in",
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        )
        browser.close()
    print(f"PDF saved: {pdf_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate standalone HTML pitch-deck for the MOT project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("PRESENTATION.html"),
        help="Output HTML file path.",
    )
    p.add_argument(
        "--pdf",
        action="store_true",
        help="Also export PDF (requires playwright: pip install playwright && playwright install chromium).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    html = build_html()
    args.output.write_text(html, encoding="utf-8")
    print(f"HTML saved: {args.output}")
    if args.pdf:
        pdf_path = args.output.with_suffix(".pdf")
        _export_pdf(args.output, pdf_path)


if __name__ == "__main__":
    main()
