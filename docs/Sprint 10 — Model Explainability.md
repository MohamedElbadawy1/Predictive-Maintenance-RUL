# Sprint 9 — Model Explainability

## Objective

The goal of this sprint was to understand how the best-performing traditional machine learning model makes its predictions.

Instead of treating the model as a black box, we analyzed feature importance to answer questions such as:

- Which features contribute the most to Remaining Useful Life (RUL) prediction?
- Are engineered features more useful than the original sensor measurements?
- Which feature engineering techniques are effective?
- Which features contribute little or no predictive value?
- How can these findings guide future model improvements?

The insights gained in this sprint will drive the feature selection strategy in the next sprint.

---

# Sprint Deliverables

Implemented:

```
src/
└── explainability/
    ├── feature_importance.py
    ├── visualization.py
    └── shap_explainer.py
```

Notebook:

```
notebooks/
└── 10_model_explainability.ipynb
```

Documentation:

```
docs/
└── Sprint_9_Model_Explainability.md
```

---

# Explainability Pipeline

```
Best Model
      │
      ▼
Feature Importance
      │
      ▼
Importance Visualization
      │
      ▼
Feature Family Analysis
      │
      ▼
Engineering Decisions
      │
      ▼
Feature Selection (Next Sprint)
```

---

# Feature Importance Analysis

The trained CatBoost model was analyzed using its built-in feature importance scores.

The results showed a clear ranking of the most influential features.

## Top Observations

### 1. `time_in_cycles` is the most important feature

The engine cycle count dominates all other features.

This is expected because Remaining Useful Life naturally decreases as the engine accumulates operating cycles.

Although this feature provides significant predictive power, it also raises an engineering question:

> Should the model rely heavily on cycle count, or should it learn degradation primarily from sensor behavior?

This question will be investigated experimentally during the Feature Selection sprint.

---

### 2. Sensor 13 is the most informative sensor

The following features consistently appeared among the most important:

- sensor_13
- sensor_13_lag_1
- sensor_13_lag_2
- sensor_13_lag_3

This indicates that both the current measurement and the recent history of Sensor 13 contain valuable degradation information.

---

### 3. Lag Features dominate the model

Multiple lag features appear in the highest-ranked features, including:

- sensor_13_lag_1
- sensor_13_lag_2
- sensor_13_lag_3
- sensor_11_lag_1
- sensor_15_lag_1
- sensor_4_lag_1
- sensor_6_lag_3

This validates the design decision made during Sprint 6 to include temporal information for traditional machine learning models.

---

### 4. Rolling Statistics contribute relatively little

Rolling Mean and Rolling Standard Deviation features appeared much lower in the ranking.

This suggests that the chosen rolling window size may not be optimal or that CatBoost already captures temporal patterns using lag features.

---

### 5. Rate of Change contributes very little

Rate-of-change features have minimal importance.

This supports the earlier engineering concern that numerical derivatives can amplify sensor noise and may not provide additional predictive information.

---

# Feature Family Contribution

To better understand the contribution of each feature engineering strategy, feature importance values were grouped by feature family.

| Feature Family | Total Importance |
|----------------|----------------:|
| Lag Features | **53.999** |
| Raw Features | **43.732** |
| Rolling Features | **1.939** |
| Rate of Change | **0.330** |

## Interpretation

### Lag Features

Lag Features provide the largest contribution to the model.

This demonstrates that recent sensor history is more informative than the current measurement alone.

---

### Raw Features

Raw sensor values remain highly informative and continue to contribute substantially to prediction performance.

The best-performing model relies on both original sensor measurements and engineered temporal features.

---

### Rolling Features

Rolling statistics contribute only a small percentage of the overall importance.

Future experiments will investigate whether different rolling window sizes improve their usefulness.

---

### Rate of Change

Rate-of-change features contribute almost no predictive information for the current model.

These features are candidates for removal during the next sprint.

---

# Engineering Decisions

Based on the explainability analysis, the following engineering decisions were made.

## Decision 1

Keep Lag Features.

The explainability results strongly justify keeping lag features in the feature engineering pipeline.

---

## Decision 2

Keep Raw Sensor Features.

Raw sensor measurements continue to provide valuable information and should remain in the baseline feature set.

---

## Decision 3

Rolling Features require further investigation.

Rather than removing them immediately, different rolling window sizes will be evaluated during future experiments.

The rolling window size will remain a tunable hyperparameter.

---

## Decision 4

Rate-of-Change Features are candidates for removal.

Their extremely low contribution suggests they may unnecessarily increase feature dimensionality without improving predictive performance.

They will be evaluated during Feature Selection.

---

## Decision 5

Investigate the dependence on `time_in_cycles`.

Although this feature is highly predictive, experiments will compare models trained with and without it to determine whether the model can learn degradation primarily from sensor behavior.

---

# Limitations

This sprint used feature importance derived from CatBoost.

Although feature importance provides a useful global ranking, it does not explain:

- Whether a feature increases or decreases predicted RUL.
- Non-linear feature interactions.
- Local explanations for individual predictions.

These limitations motivate the use of SHAP analysis in future explainability experiments.

---

# What Comes Next

The explainability results directly influence the next sprint.

## Sprint 10 — Feature Selection & Optimization

Objectives:

- Remove features with negligible importance.
- Retrain all baseline models.
- Compare performance before and after feature reduction.
- Measure improvements in training time.
- Evaluate whether removing weak features improves generalization.
- Freeze the final feature set for future experiments.

Possible experiments include:

- Removing Rate-of-Change features.
- Evaluating different rolling window sizes.
- Comparing models with and without `time_in_cycles`.
- Removing zero-importance features.
- Comparing feature subsets based on importance thresholds.

The outcome of Sprint 10 will define the final feature space used for hyperparameter optimization and deep learning models.

---

# Sprint Summary

Sprint 9 successfully transformed the best-performing CatBoost model from a black-box predictor into an interpretable model.

The analysis confirmed that temporal information captured through lag features provides the greatest predictive value, while rolling statistics and rate-of-change features contribute relatively little under the current configuration.

These findings establish a clear, evidence-based roadmap for feature selection and model optimization in the following sprint.