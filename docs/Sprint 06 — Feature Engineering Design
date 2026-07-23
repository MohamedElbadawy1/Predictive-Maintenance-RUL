# Sprint 5 — Feature Engineering Design

## Overview

Sprint 5 is dedicated entirely to the architectural design of the feature engineering pipeline.

No implementation is performed during this sprint.

Instead, the objective is to establish a clear, modular, and reusable feature engineering strategy that supports multiple model families while preventing data leakage and maintaining software engineering best practices.

The resulting design will guide the implementation of future sprints without requiring major architectural changes.

---

# Objectives

This sprint answers the following design questions:

- Which original features should be preserved?
- Which engineered features should be created?
- Which feature ideas should be postponed?
- How will feature engineering differ between traditional machine learning models and sequence models?
- How will data leakage be prevented?
- How should the preprocessing pipeline be organized?
- Which engineering decisions should be fixed before implementation?

---

# Available Features

The NASA CMAPSS dataset contains the following input variables.

## Engine Information

- unit_number
- time_in_cycles

## Operational Settings

- operational_setting_1
- operational_setting_2
- operational_setting_3

## Sensor Measurements

- sensor_1
- sensor_2
- ...
- sensor_21

Total input columns:

26 features

---

# Features to Preserve

## Operational Settings

All three operational settings will initially be retained.

Although Operational Setting 3 shows limited variability, it may still contain useful information for certain operating regimes.

Removing features will be postponed until evidence from trained models becomes available.

---

## Sensor Measurements

All sensor measurements will be preserved during the first implementation.

Reasons:

- Low variance does not necessarily imply low predictive value.
- Correlation alone is insufficient for feature elimination.
- Feature importance will be evaluated after baseline models have been trained.

---

## Engine Identifier

The engine identifier (`unit_number`) will remain available during preprocessing.

It is required for:

- Rolling statistics
- Lag feature generation
- Sequence construction
- Group-based operations

It will not be used directly as a model input.

---

## Time in Cycles

The cycle number will also be preserved.

Reasons:

- Temporal ordering
- Rolling calculations
- Lag generation
- Sequence construction

---

# Planned Engineered Features

## Rolling Mean

Purpose

Capture long-term degradation trends while reducing sensor noise.

Window Size

Status:

To Be Determined (TBD)

The rolling window size will be treated as a hyperparameter.

Possible values:

- 5
- 10
- 20
- 30

Status

Planned for Sprint 6.

---

## Rolling Standard Deviation

Purpose

Measure signal stability throughout engine degradation.

Window Size

TBD

Status

Planned for Sprint 6.

---

## Lag Features

Examples

Sensor(t−1)

Sensor(t−2)

Sensor(t−3)

Purpose

Allow traditional machine learning models to capture temporal dependencies.

Status

Planned for Sprint 6.

---

## Rate of Change

Example

ΔSensor = Sensor(t) − Sensor(t−1)

Purpose

Capture degradation speed rather than absolute sensor values.

Considerations

Derivative-based features amplify sensor noise.

Their usefulness will be evaluated experimentally.

Status

Planned for Sprint 6.

---

# Features Postponed

## Engine Age

Example

Current Cycle / Maximum Cycle

Status

Postponed.

Reason

The maximum cycle is only known during training.

It is unavailable during inference and would therefore introduce data leakage if used directly.

An alternative representation may be investigated in a future sprint.

---

## Interaction Features

Examples

Sensor2 × OperationalSetting1

Sensor9 / Sensor4

Status

Postponed.

Reason

Will be considered after feature importance analysis.

---

## Frequency-Domain Features

Examples

- FFT
- Wavelets

Status

Future work.

Reason

Increase complexity without first establishing a strong baseline.

---

## Polynomial Features

Status

Future work.

Reason

Tree-based models already capture nonlinear relationships effectively.

---

## Dimensionality Reduction

Examples

- PCA
- ICA
- Autoencoders

Status

Future work.

Reason

Interpretability is preferred during the early stages of the project.

---

# Traditional ML vs Sequence Models

Feature engineering requirements differ significantly depending on the model family.

## Traditional Machine Learning

Examples

- XGBoost
- LightGBM
- CatBoost
- Random Forest

Characteristics

These models cannot learn temporal dependencies directly.

Required engineered features

- Rolling Mean
- Rolling Standard Deviation
- Lag Features
- Rate of Change

Window generation is not required.

---

## Sequence Models

Examples

- LSTM
- GRU

Characteristics

These models naturally learn temporal dependencies.

Therefore:

- Lag Features are unnecessary.
- Rolling features are optional.
- Window generation is mandatory.

Instead of additional engineered features, these models require a sequence representation of the data.

---

# Data Leakage Prevention

Preventing data leakage is a primary architectural requirement.

The following rules will be enforced.

## Rule 1

Feature engineering may only use the current observation and previous observations.

Future engine cycles must never be accessed.

---

## Rule 2

Rolling statistics must use historical observations only.

Centered rolling windows are prohibited.

---

## Rule 3

Lag features may only reference previous cycles.

Future values are never allowed.

---

## Rule 4

Scaling parameters must be fitted using the training dataset only.

The fitted scaler will then be applied to validation and testing datasets.

---

## Rule 5

Validation splitting will be performed by engine rather than by individual observations.

This prevents information leakage between engine trajectories.

---

## Rule 6

Sequence generation will preserve chronological order.

Random shuffling before sequence generation is prohibited.

---

# Scaling Strategy

Scaling requirements depend on the selected model.

Status

To be determined after baseline experiments.

Current expectation

Traditional ML

- Scaling may not be required.

Sequence Models

- Scaling will almost certainly be required.

The most appropriate scaler (StandardScaler, MinMaxScaler, RobustScaler, etc.) will be selected experimentally.

---

# Feature Registry

| Feature | Traditional ML | LSTM / GRU | Planned Sprint |
|----------|:--------------:|:----------:|:--------------:|
| Original Sensors | ✅ | ✅ | Existing |
| Operational Settings | ✅ | ✅ | Existing |
| Rolling Mean | ✅ | Optional | Sprint 6 |
| Rolling Std | ✅ | Optional | Sprint 6 |
| Lag Features | ✅ | ❌ | Sprint 6 |
| Rate of Change | ✅ | Optional | Sprint 6 |
| Sequence Builder | ❌ | ✅ | Sprint 7 |
| Engine Age | Postponed | Postponed | TBD |
| Interaction Features | TBD | TBD | Future |

---

# Proposed Pipeline

## Traditional Machine Learning Pipeline

```
Raw Training Dataset
        │
        ▼
Target Generation
        │
        ▼
Feature Engineering
        │
        ▼
Train / Validation Split (by engine)
        │
        ▼
Scaling (Fit on Train Only)
        │
        ├──────────────┐
        ▼              ▼
Transform Train   Transform Validation
        │              │
        └──────┬───────┘
               ▼
Model Training
```

---

## Sequence Model Pipeline

```
Raw Training Dataset
        │
        ▼
Target Generation
        │
        ▼
Train / Validation Split (by engine)
        │
        ▼
Scaling (Fit on Train Only)
        │
        ▼
Sequence Builder
        │
        ▼
LSTM / GRU
```

---

# Engineering Decisions

## Decision 1

All original sensors will be retained during the first implementation.

---

## Decision 2

Feature elimination will rely on evidence rather than assumptions.

---

## Decision 3

Traditional machine learning and sequence models will use independent preprocessing pipelines.

---

## Decision 4

All engineered features must be causal.

Future observations are never permitted during feature generation.

---

## Decision 5

Rolling window sizes will be treated as hyperparameters.

---

## Decision 6

Scaling strategy will be selected after baseline experiments.

---

## Decision 7

Sequence generation is considered a data representation step rather than feature engineering.

---

## Decision 8

Feature engineering components will remain modular, reusable, and independently testable.

---

# Planned Architecture

```
src/
│
├── preprocessing/
│   ├── rul_generator.py
│   ├── feature_engineer.py
│   ├── sequence_generator.py
│   └── preprocessor.py
```

---

# Future Sprint Plan

## Sprint 6

Traditional Machine Learning Feature Pipeline

Implementation:

- Rolling Mean
- Rolling Standard Deviation
- Lag Features
- Rate of Change

---

## Sprint 7

Sequence Pipeline

Implementation:

- Sequence Builder
- Window Generation
- LSTM / GRU Input Preparation

---

# Sprint Outcome

Sprint 5 establishes the architectural foundation for feature engineering.

Rather than implementing transformations immediately, this sprint defines a modular design that separates target engineering, feature engineering, sequence generation, and preprocessing into independent components.

This architecture promotes maintainability, extensibility, reproducibility, and prevents data leakage while supporting both traditional machine learning models and deep learning approaches.