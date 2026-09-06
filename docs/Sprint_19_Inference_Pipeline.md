# Production Phase 1 — Consolidated Inference Pipeline

---

# Goal

Move from "predict RUL" being logic scattered across analysis scripts (`bucket_error_analysis.py`,
various test-eval notebooks, `promote_regime_aware_model.py` each re-implementing the
same chain) to **one class that owns the entire raw-input-to-prediction path**. This
is the foundation every later production step (API, containerization, monitoring)
depends on — none of that is buildable on top of copy-pasted logic.

---

# What Was Built

**`src/inference/pipeline.py` — `InferencePipeline`**

Loads every artifact once at construction:
- `best_model.pkl` (via `BaseTrainer.load`)
- `regime_normalizer.pkl` — only if the model needs it (see below)
- `feature_scaler.pkl`
- `selected_features.json`

Exposes one method that matters:

```python
pipeline = InferencePipeline()
predictions = pipeline.predict(raw_engine_readings)
```

**Auto-detects whether the current canonical model needs regime-aware normalization**
by reading the `requires_regime_normalizer` flag `promote_regime_aware_model.py`
writes into `best_params.json` — callers never need to know or care which
preprocessing path is currently canonical. This directly closes the operational risk
flagged in Sprint 17: predicting with the wrong preprocessing path for a
regime-aware model silently produces wrong results. Now that risk lives in exactly
one place, checked automatically, instead of being something every caller has to
remember.

## Input / Output Contract

**In**: a DataFrame with one row per (engine, cycle) — same shape as `test_FD004`
(`unit_number`, `time_in_cycles`, 3 operational settings, 21 sensors). Covers each
engine's full history up to "now," not just the latest cycle — lag/rolling/diff
features need the prior cycles to be computable.

**Out**: one row per engine —

| Column | Meaning |
|---|---|
| `unit_number` | Engine ID |
| `predicted_RUL` | The prediction |
| `n_cycles_seen` | How much history this engine had |
| `short_history_warning` | `True` if fewer than 4 cycles (lag features undefined) |
| `regime` | Detected operating regime (only if the model requires regime normalization) |

The prediction is for each engine's **most recent cycle** — matching the official
evaluation protocol used since Sprint 13, not a per-cycle prediction for the whole
history.

---

# Verification (Not Just "It Runs")

Running without an error isn't the bar for something meant to sit in a production
path. Four checks, all against real data:

1. **Exact match to the already-validated pipeline**: ran `InferencePipeline` against
   the real official `test_FD004`, compared to `RUL_FD004` — MAE 19.307, identical to
   the value independently computed by `bucket_error_analysis.py`'s manually-scripted
   version of the same chain. Confirms the consolidation is mathematically identical
   to the logic it replaces, not just similar.
2. **Single-engine vs. batch consistency**: predicting for one engine in isolation
   produced the exact same value (140.969712 for unit 5) as that engine's row when
   predicting for the full batch — confirms no cross-engine leakage in the
   groupby-based logic.
3. **Short-history handling**: an engine with only 2 cycles (fewer than the 4 needed
   for the longest lag feature) still gets a prediction (CatBoost handles the
   resulting NaNs natively, same as always in this project) but is correctly flagged
   via `short_history_warning`, not silently treated as equally reliable.
4. **Input validation**: a DataFrame missing a required sensor column, and an empty
   DataFrame, both raise a clear `CustomException` naming the problem rather than
   failing deep inside feature engineering with a confusing error.

---

# What This Enables Next

Every later production phase depends on this existing:
- **Phase 2** (model packaging / MLflow Model Registry) wraps this pipeline as a
  versioned, promotable unit instead of a mutable file.
- **Phase 3** (API) becomes a thin wrapper: parse the request, call
  `pipeline.predict()`, format the response.
- **Phase 4** (testing) gets a real target — this class, with the four checks above
  formalized as actual test cases rather than one-off verification.

---

# Decisions

- `InferencePipeline` is the only sanctioned way to go from raw sensor data to a
  prediction going forward. Any new analysis script should import and use it rather
  than re-implementing the chain again.
- The `requires_regime_normalizer` auto-detection means promoting a future model
  (regime-aware or not) via `promote_regime_aware_model.py`-style scripts requires no
  changes to this class — it reads the flag, not a hardcoded assumption.
- Formal unit tests (Phase 4) are the next place this verification work should live
  permanently, not just as one-off checks in this session.
