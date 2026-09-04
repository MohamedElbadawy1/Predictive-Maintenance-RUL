"""
Promote the regime-aware tuned CatBoost model to canonical status.

Real results from the full regime-aware tuning run (not a smoke test):

    Model      Stage              MAE        RMSE       R2        MAPE
    catboost   tuned_individual   19.132785  25.367345  0.783540  28.446220
    ensemble   tuned_ensemble     19.247538  25.335927  0.784076  28.010054
    xgboost    tuned_individual   19.428778  25.299473  0.784697  28.744324
    lightgbm   tuned_individual   19.579705  25.629934  0.779035  28.435882

    Prior canonical best (non-regime-aware, globally-scaled ensemble): 19.226

Solo regime-aware CatBoost (19.133) beats every other option, including
the regime-aware ensemble (19.248) — the ensemble actually underperforms
its own best member here, dragged down by the weaker XGBoost/LightGBM
members even with inverse-MAE weighting. See the sprint doc for the
full discussion; the short version is CatBoost benefited far more from
regime-aware normalization than the other two did, for reasons not yet
fully understood.

Promoting the INDIVIDUAL CatBoost model, not the ensemble.

IMPORTANT — this changes the inference pipeline, not just the model
file: any code that loads best_model.pkl and predicts must now ALSO
apply regime_normalizer.pkl to the raw sensor readings BEFORE feature
engineering. Predicting with the old (non-regime-aware) preprocessing
path will silently produce wrong results — the model expects
regime-normalized sensor inputs, not globally-scaled ones.
"""
import json
import shutil
from src.config.config import MODELS_DIR, SCALERS_DIR, SELECTED_FEATURES_PATH

# 1. Promote the model itself
shutil.copy(MODELS_DIR / "catboost_tuned_regime_aware.pkl", MODELS_DIR / "best_model.pkl")

# 2. Promote the feature list and scaler that go with it
shutil.copy(MODELS_DIR / "selected_features_regime_aware.json", SELECTED_FEATURES_PATH)
shutil.copy(SCALERS_DIR / "feature_scaler_regime_aware.pkl", SCALERS_DIR / "feature_scaler.pkl")

# 3. regime_normalizer.pkl is already at its canonical path (MODELS_DIR /
#    "regime_normalizer.pkl") — the tuning script saved it there directly,
#    no rename needed. Just confirm it exists.
regime_normalizer_path = MODELS_DIR / "regime_normalizer.pkl"
assert regime_normalizer_path.exists(), (
    "regime_normalizer.pkl not found — required for inference with this "
    "model. Re-run tune_and_ensemble_regime_aware.py if missing."
)

# 4. Record what was promoted and why, with the real metrics
with open(MODELS_DIR / "catboost_best_params_regime_aware.json") as f:
    winning = json.load(f)

best_params_record = {
    "params": winning["params"],
    "metrics": {
        "MAE": 19.132785, "RMSE": 25.367345, "R2": 0.783540, "MAPE": 28.446220,
    },
    "normalization": "regime_aware",
    "requires_regime_normalizer": True,
    "regime_normalizer_path": str(regime_normalizer_path),
    "note": (
        "Promoted after the real regime-aware tuning run beat every other "
        "option, including the regime-aware ensemble (19.248) and the "
        "prior canonical non-regime-aware ensemble (19.226). Solo "
        "CatBoost, not the ensemble. Inference requires applying "
        "regime_normalizer.pkl to raw sensor readings before feature "
        "engineering — this is a different preprocessing path than the "
        "pre-regime-aware model used."
    ),
    "comparison": {
        "this_model_test_MAE": 19.132785,
        "regime_aware_ensemble_test_MAE": 19.247538,
        "prior_canonical_ensemble_test_MAE": 19.226,
    },
}

with open(MODELS_DIR / "best_params.json", "w") as f:
    json.dump(best_params_record, f, indent=2)

with open(MODELS_DIR / "best_model_name.txt", "w") as f:
    f.write("catboost_regime_aware_tuned")

print("Promoted regime-aware CatBoost to canonical best_model.pkl")
print(f"  Test MAE: 19.133 (was 19.226)")
print(f"  Also promoted: selected_features.json, feature_scaler.pkl")
print(f"  regime_normalizer.pkl confirmed at: {regime_normalizer_path}")
print("\nReminder: inference now requires regime_normalizer.pkl as an extra")
print("preprocessing step before feature engineering.")
