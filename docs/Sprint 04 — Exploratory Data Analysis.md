# Sprint 3 — Exploratory Data Analysis (NASA C-MAPSS FD004)

**Project:** Predictive Maintenance — Turbofan Engine RUL Prediction
**Dataset:** NASA C-MAPSS (FD004 subset)
**Sprint focus:** Exploratory Data Analysis (EDA) — understanding the data before any preprocessing or modeling decisions are locked in.

---

## 1. Sprint Goal

Establish a solid, evidence-based understanding of the raw dataset — engine lifetimes, sensor behavior, operating conditions, and the RUL target — before committing to feature engineering or model design. Every notebook ends with an **Observations** section and an **Engineering Decisions** section, so this sprint produces a paper trail of *why* later preprocessing choices are made, not just *what* they are.

## 2. Deliverables

Seven notebooks were produced under `notebooks/`, each with a single, focused objective:

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_data_overview.ipynb` | First look at the raw data: shape, schema, validation, engine/sensor counts |
| 02 | `02_engine_lifetime_analysis.ipynb` | Deep dive into how long engines run before failure |
| 03 | `03_sensor_variability.ipynb` | Which sensors carry signal vs. which are flat/near-constant |
| 04 | `04_sensor_degradation_analysis.ipynb` | How sensor readings evolve across normalized engine life (0–100%) |
| 05 | `05_sensor_correlation_analysis.ipynb` | Redundancy between sensors |
| 06 | `06_operating_conditions_analysis.ipynb` | Effect of the 3 operational settings on sensors |
| 07 | `07_rul_analysis.ipynb` | Distribution and shape of the RUL target itself |

Supporting code used across all notebooks: `src/data/loader.py` (`DataLoader`) and `src/data/validator.py` (`DataValidator`), driven by paths in `src/config/config.py`. Several notebooks export intermediate artifacts to `reports/` (`highly_correlated_sensors.csv`, `operational_setting_sensor_correlation.csv`, `rul_summary.csv`).

---

## 3. Notebook-by-Notebook Summary

### 01 — Data Overview
- Loaded train/test/RUL data via `DataLoader` and validated it with `DataValidator`.
- **Findings:** 249 engines in train, 248 in test; 21 sensor columns; 3 operational setting columns; no missing values; all columns numeric; engine lifetimes vary considerably (right-skewed).
- **Decisions:** No preprocessing needed yet. Future models must handle variable-length sequences. Next step → investigate sensor variability.

### 02 — Engine Lifetime Analysis
- Computed per-engine lifetime (`max time_in_cycles` per `unit_number`) and examined its distribution (histogram, boxplot, percentiles, shortest-lived engines).
- **Findings:** Lifetimes range ~130–540+ cycles, most engines fail between 150–300 cycles. Distribution is unimodal and right-skewed. A handful of statistical outliers (>~440 cycles) exist on the long-life side only; no abnormally short-lived engines.
- **Decisions:**
  1. Preserve all engines — long-lived outliers look legitimate, not data errors.
  2. Sequence models (e.g., LSTM) will need a variable-length-sequence strategy (sliding windows / padding).
  3. RUL must be computed **per engine**, not from a global max cycle.
  4. No lifetime-based filtering of observations.
  5. Proceed to sensor-level investigation.

### 03 — Sensor Variability
- Computed variance, std, unique-value counts, and full summary stats per sensor; visualized with a variance bar chart and raw/normalized boxplots.
- **Findings:** Variance spans ~2.2×10⁻⁵ (Sensor 16) to >113,000 (Sensor 9). Sensor 16 and Sensor 19 are near-binary/low-cardinality (2 unique values) — likely discrete states rather than continuous measurements. Sensors 9, 7, 8, 18 show the highest variability. High variance ≠ high predictive value on its own.
- **Decisions:**
  1. Keep all sensors — no removal based on variance alone.
  2. Investigate low-cardinality sensors (16, 19) further.
  3. Move to correlation analysis to check redundancy.
  4. Move to temporal/degradation analysis — variability over time matters more than raw variance.
  5. Defer any feature elimination until multiple lines of evidence are combined.

### 04 — Sensor Degradation Analysis
- Normalized each engine's cycle count to a 0–100% "life percentage" (20 life-stage bins) so engines of different lifespans are comparable; plotted average sensor value per life stage (raw and Min-Max normalized) and quantified start→end change per sensor.
- **Findings:** Most sensors change only mildly across the lifecycle (degradation is gradual, not abrupt). Largest absolute changes: **Sensor 16 (0.115), Sensor 14 (0.076), Sensor 11 (0.046), Sensor 4 (0.036), Sensor 3 (0.024)**. Moderate movers: Sensors 17, 9, 1, 6, 5, 20, 21. Nearly stable: Sensors 10, 15, 13, 19. Degradation signal appears **distributed across multiple sensors**, not concentrated in one.
- **Decisions:**
  1. Prioritize high-change sensors (16, 14, 11) in later feature work.
  2. Keep moderate-change sensors — models may exploit interactions.
  3. Don't drop stable sensors (10, 15) yet — reassess after correlation/importance analysis.
  4. Feature selection will combine variance, lifecycle trend, correlation, model importance, and SHAP — not any single criterion.
  5. Proceed to correlation analysis.

### 05 — Sensor Correlation Analysis
- Built the full sensor×sensor correlation matrix (heatmap), extracted pairs with |r| ≥ 0.90, and counted each sensor's "strong connections."
- **Findings:** Substantial redundancy across sensors. Most connected group (12 strong correlations each): **Sensors 2, 7, 9, 12, 20, 21**. Moderately connected: 3, 4, 10, 17. Weakly connected (unique information): **13, 14, 15, 16, 19** — Sensor 16 in particular has only one strong correlation, reinforcing that it behaves differently from the rest of the fleet of sensors and should not be dropped without further checks.
- **Decisions:**
  1. Keep all sensors — correlation alone isn't grounds for removal.
  2. Document the redundant sensor group (2, 7, 9, 12, 20, 21) for later feature-selection revisit.
  3. Preserve weakly-connected sensors (16, 13, 19) — likely unique signal.
  4. Delay feature elimination until variance + lifecycle + correlation + model importance + SHAP are all considered together.
  5. Note that dimensionality reduction / model-based selection may be useful in the feature engineering stage.

### 06 — Operating Conditions Analysis
- Profiled the 3 operational settings (distributions, pairwise scatterplots, correlation among settings, and correlation between settings and all 21 sensors).
- **Findings:** Operational Setting 1 varies over a wide continuous range (0–~42). Setting 2 varies continuously but over a much smaller range and is **strongly linearly related to Setting 1** (possible redundancy). Setting 3 is nearly constant (mostly at 100) and largely uncorrelated with the other two — it contributes little variability.
- **Decisions:**
  1. Keep all three settings for now; evaluate Setting 3's contribution during feature selection.
  2. Investigate whether Settings 1 & 2 are redundant enough to drop one.
  3. Retain operational settings as model inputs — they may explain sensor variance independent of degradation.
  4. Future work must separate **operating-condition effects** from **true degradation effects** to avoid misleading feature-importance results (important given FD004's multiple operating regimes).

### 07 — RUL (Target) Analysis
- Computed ground-truth RUL per row (`max_cycle − time_in_cycles`), examined its distribution (histogram, boxplot, binned groups: 0–20, 21–50, 51–100, 101–150, >150), plotted RUL-vs-cycle trajectories for sample engines, and tested a cap at **125 cycles**.
- **Findings:** RUL is right-skewed and clearly imbalanced — most samples fall in the >150 "healthy" bucket, while the operationally critical 0–20 cycle failure zone is comparatively under-represented. Healthy engines show similar, low-information sensor signatures, so distinguishing e.g. 250 vs. 450 remaining cycles is inherently hard; the signal strengthens as engines approach failure.
- **Decisions:**
  1. **Adopt RUL capping at 125 cycles** for model training (raw RUL kept unchanged during EDA).
  2. Apply the cap only at the preprocessing stage, not to the analytical dataset.
  3. Capping focuses model capacity on the degradation region, where predictions are actionable for maintenance planning.
  4. Capping stabilizes optimization by preventing very large RUL values from dominating the regression loss.
  5. A 125-cycle cap is consistent with common practice in published C-MAPSS literature (typically 125–130), improving comparability.

---

## 4. Consolidated Cross-Notebook Decisions

Pulling together the "Engineering Decisions" from all seven notebooks, the following EDA-stage conclusions carry into Sprint 4 (Feature Engineering / Preprocessing):

1. **No rows or engines removed.** Long-lived engines are legitimate, not outliers/errors.
2. **No sensors removed yet.** Decisions about Sensor 16 (near-constant, low correlation) and Sensor 19 (low cardinality) are explicitly deferred, not made — they carry potentially unique signal despite low variance.
3. **RUL must be computed per engine** and **capped at 125 cycles** before training.
4. **Variable-length sequences** are a first-class concern for any sequence model (LSTM/Transformer) — needs padding or sliding-window design.
5. **Redundant sensor groups identified** for future dimensionality reduction: `{2, 7, 9, 12, 20, 21}` (strongly inter-correlated).
6. **High-priority degradation sensors** to weight in feature engineering: `16, 14, 11, 4, 3` (largest lifecycle change).
7. **Operational Settings 1 & 2 are likely redundant**; Setting 3 contributes little variance (mostly constant at 100) — flagged for feature-selection review.
8. **Operating-condition effects must be separated from degradation effects** before trusting any feature-importance results — critical for FD004 since it spans multiple operating regimes (unlike simpler C-MAPSS subsets).
9. **Final feature selection is explicitly deferred** to a later stage that will combine: variance, lifecycle-trend change, correlation, model-based feature importance, and SHAP explainability — no single EDA metric is used in isolation.

---

## 5. Artifacts Produced

- `reports/highly_correlated_sensors.csv` — sensor pairs with |r| ≥ 0.90
- `reports/operational_setting_sensor_correlation.csv` — max |correlation| of each sensor with the 3 operational settings
- `reports/rul_summary.csv` — RUL summary statistics (min/max/mean/median/std)

## 6. Open Questions Carried Into Next Sprint

- Should Sensor 16 and Sensor 19 be treated as categorical/discrete features rather than continuous sensor readings?
- Should Operational Setting 3 be dropped, or does it interact meaningfully with sensors under specific regimes?
- What is the best strategy (sliding window vs. padding) for handling variable engine lifespans in sequence models?
- How should operating-regime clustering (using Settings 1–3) be used to normalize/condition sensor readings before degradation modeling?

---

*End of Sprint 3 documentation.*
