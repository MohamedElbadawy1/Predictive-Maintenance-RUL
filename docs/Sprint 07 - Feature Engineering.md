# Sprint 6 — Traditional Machine Learning Feature Engineering

**Status:** Completed

---

# Sprint Goal

The objective of this sprint was to build the first version of a reusable feature engineering pipeline for traditional machine learning models.

Unlike previous sprints that focused on data understanding and target engineering, this sprint introduces the first stage of transforming raw sensor measurements into richer temporal features that better represent engine degradation patterns.

The implementation follows a modular, object-oriented design to ensure maintainability and extensibility throughout the project lifecycle.

---

# Objectives

This sprint aimed to:

- Build a reusable `FeatureEngineer` class.
- Generate temporal features for traditional ML models.
- Preserve the original dataset without modification.
- Compute features independently for each engine.
- Maintain compatibility with future preprocessing and modeling pipelines.
- Validate the implementation using unit tests.

---

# Implemented Components

## FeatureEngineer

**Location**

```
src/preprocessing/feature_engineer.py
```

### Public API

```python
engineer = FeatureEngineer(
    sensor_columns=SENSOR_COLUMNS,
    rolling_window=5,
    lags=[1, 2, 3],
)

feature_df = engineer.transform(train_df)
```

The class follows a scikit-learn inspired interface by exposing a single public method:

```python
transform()
```

All feature generation logic is encapsulated within private helper methods.

---

# Generated Features

Version 1 supports four temporal feature types.

---

## 1. Rolling Mean

Example:

```
sensor_2_roll_mean_5
```

Purpose:

- Smooth short-term sensor fluctuations.
- Capture the local trend of sensor behavior.
- Reduce the effect of measurement noise.

---

## 2. Rolling Standard Deviation

Example:

```
sensor_2_roll_std_5
```

Purpose:

- Measure local variability.
- Detect unstable operating conditions.
- Capture changes in signal consistency.

---

## 3. Lag Features

Examples:

```
sensor_2_lag_1
sensor_2_lag_2
sensor_2_lag_3
```

Purpose:

- Preserve temporal memory.
- Allow tree-based models to observe previous sensor values.
- Approximate sequential information without recurrent neural networks.

---

## 4. Rate of Change (Difference)

Example:

```
sensor_2_diff
```

Purpose:

- Capture degradation speed.
- Highlight abrupt sensor changes.
- Estimate first-order temporal derivatives.

---

# Engineering Design

The implementation follows the Single Responsibility Principle.

```
FeatureEngineer
│
├── transform()
├── _validate_input()
├── _rolling_mean()
├── _rolling_std()
├── _lag_features()
└── _rate_of_change()
```

Each feature type is implemented independently to simplify maintenance and future extensions.

---

# Processing Strategy

All temporal features are generated independently for each engine using:

```python
groupby("unit_number")
```

This prevents information leakage between different engines and preserves the temporal structure of each degradation trajectory.

---

# NaN Handling

Temporal feature generation naturally introduces missing values.

Examples:

- Rolling statistics require previous observations.
- Lag features require previous cycles.
- First differences are undefined for the first cycle.

These missing values are **intentionally preserved**.

The `FeatureEngineer` does **not** perform:

- Imputation
- Row removal
- Data cleaning

Handling missing values is deferred to the preprocessing pipeline in Sprint 7.

---

# Input Validation

The class validates:

- Non-empty DataFrame.
- Presence of the grouping column.
- Existence of all sensor columns.
- Positive rolling window size.
- Valid lag values.

Invalid configurations raise the project's custom exception.

---

# Unit Testing

A dedicated test suite was implemented.

**Location**

```
tests/test_feature_engineer.py
```

Implemented tests include:

- Return a new DataFrame.
- Preserve the original DataFrame.
- Maintain row count.
- Generate all expected feature columns.
- Verify lag features are calculated independently for each engine.
- Verify difference features are calculated independently for each engine.
- Validate invalid rolling window.
- Validate invalid lag values.
- Detect missing sensor columns.
- Detect missing grouping column.
- Detect empty DataFrame.

These tests ensure both functional correctness and architectural integrity.

---

# Results

The feature engineering pipeline successfully generated:

| Feature Type | Count |
|--------------|------:|
| Rolling Mean | 21 |
| Rolling Standard Deviation | 21 |
| Lag Features | 63 |
| Rate of Change | 21 |
| **Total New Features** | **126** |

The transformed dataset:

- Preserves all original rows.
- Preserves all original columns.
- Appends newly engineered features.
- Never modifies the input DataFrame.

---

# Engineering Decisions

The following architectural decisions were adopted during this sprint:

- Use an object-oriented design.
- Follow a scikit-learn inspired API (`transform()`).
- Compute temporal features independently for each engine.
- Preserve NaN values.
- Avoid feature selection during this stage.
- Avoid scaling during this stage.
- Avoid data splitting during this stage.
- Separate feature engineering from sequence generation.
- Keep the pipeline modular and extensible.

---

# Limitations

Current implementation does not include:

- Feature selection.
- Scaling.
- Dimensionality reduction.
- Window generation.
- Sequence creation.
- Advanced statistical features.
- Frequency-domain features.

These capabilities will be introduced in future sprints.

---

# Lessons Learned

This sprint demonstrated the importance of separating feature engineering from other preprocessing tasks.

By keeping each component focused on a single responsibility, the overall pipeline becomes easier to understand, test, maintain, and extend.

The addition of automated unit tests significantly increases confidence in the correctness of the implementation and provides a foundation for future development.

---

# Next Sprint

## Sprint 7 — Data Splitting & Scaling Pipeline

The next sprint will introduce a preprocessing pipeline responsible for:

- Train/Validation splitting by engine.
- Preventing data leakage.
- Fitting scalers using only the training data.
- Transforming validation data.
- Preparing datasets for baseline machine learning models.

Feature engineering will remain unchanged and will be integrated into this preprocessing pipeline.