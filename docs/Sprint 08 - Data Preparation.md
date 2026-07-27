# Sprint 8 — Data Preparation

## Objective

The objective of this sprint is to transform the engineered dataset into a training-ready dataset while preventing data leakage and maintaining a modular, reusable, and production-ready preprocessing pipeline.

Unlike previous sprints that focused on data understanding, target generation, and feature engineering, this sprint prepares the data for machine learning models.

---

# Motivation

After completing Feature Engineering (Sprint 6), the dataset still cannot be used directly for model training.

Several preprocessing steps are required before training any machine learning model:

- Separate features from the target.
- Split the dataset into Training and Validation sets.
- Ensure that complete engine trajectories remain together.
- Scale numerical features correctly.
- Combine all preparation steps into a reusable pipeline.

These operations form the final preprocessing stage before model training.

---

# Sprint Architecture

```
Engineered Dataset
        │
        ▼
FeatureSelector
        │
        ▼
DataSplitter
        │
        ▼
FeatureScaler
        │
        ▼
Prepared Training Dataset
```

---

# Implemented Components

## 1. FeatureSelector

**Location**

```
src/preprocessing/feature_selector.py
```

### Responsibility

Separate:

- Feature matrix (X)
- Target vector (y)

while removing columns that should not be used for model training.

### Example

```python
selector = FeatureSelector(
    target_column="RUL",
    drop_columns=["unit_number"],
)

X, y = selector.transform(df)
```

---

### Engineering Decisions

#### Removed Columns

| Column | Reason |
|----------|--------|
| unit_number | Engine identifier, not a predictive feature. |

#### Target

```
RUL
```

is separated from the feature matrix and returned independently.

---

## 2. DataSplitter

**Location**

```
src/preprocessing/data_splitter.py
```

### Responsibility

Split the dataset into **Training** and **Validation** datasets while ensuring that every engine remains entirely within a single dataset.

Instead of splitting individual rows, the splitter treats each engine as a single group.

---

## Splitting Strategy

The implementation uses **GroupShuffleSplit** from scikit-learn.

```python
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)
```

Each engine is considered one independent group.

Conceptually:

```
Engine IDs
      │
      ▼
GroupShuffleSplit
      │
      ├── Training Engines
      └── Validation Engines
```

---

## Why GroupShuffleSplit?

The NASA C-MAPSS dataset consists of complete degradation trajectories.

Each engine contains a chronological sequence of observations.

If rows were randomly split, observations from the same engine could appear in both Training and Validation datasets, allowing the model to indirectly observe part of the validation engine during training.

This would produce unrealistically optimistic evaluation metrics.

Using **GroupShuffleSplit** guarantees:

- Entire engines remain together.
- No engine appears in both datasets.
- Temporal order inside every engine is preserved.
- The split is reproducible using `random_state`.

---

## Temporal Order Preservation

Although engine groups are randomly assigned to either Training or Validation, the rows within each engine are never shuffled.

Example:

Original engine trajectory

```
Engine 15

Cycle

1
2
3
4
5
```

After splitting

```
Engine 15

Cycle

1
2
3
4
5
```

The chronological degradation history remains unchanged.

---

## Engineering Decisions

- Split by engine rather than individual rows.
- Use `GroupShuffleSplit` instead of implementing custom splitting logic.
- Preserve complete degradation trajectories.
- Prevent engine-level data leakage.
- Use a fixed random seed for reproducibility.

---

## 3. FeatureScaler

**Location**

```
src/preprocessing/feature_scaler.py
```

### Responsibility

Scale numerical features before model training.

Pipeline:

```
Training Data
      │
      ▼
Fit Scaler
      │
      ▼
Transform Training
      │
      ▼
Transform Validation
```

---

## Default Scaler

The default implementation uses

```
StandardScaler
```

The class accepts any scikit-learn compatible scaler.

Examples:

```python
FeatureScaler(StandardScaler())

FeatureScaler(MinMaxScaler())

FeatureScaler(RobustScaler())
```

---

## Why Fit Only on Training Data?

Scaling statistics (mean and standard deviation) must only be computed from the Training dataset.

Using Validation statistics would introduce information leakage.

Correct workflow:

```
Training Data

↓

Fit Scaler

↓

Transform Training

↓

Transform Validation
```

---

## Engineering Decisions

- Fit only on Training data.
- Never fit using Validation data.
- Preserve column names.
- Preserve DataFrame index.
- Save fitted scaler for inference.

---

## 4. DataPreparationPipeline

**Location**

```
src/preprocessing/preparation_pipeline.py
```

### Responsibility

Coordinate all preprocessing components into one reusable pipeline.

The pipeline itself contains no preprocessing logic.

Its only responsibility is orchestration.

Pipeline:

```
FeatureSelector

↓

DataSplitter

↓

FeatureScaler

↓

Prepared Dataset
```

---

### Example

```python
pipeline = DataPreparationPipeline(

    splitter=DataSplitter(),

    selector=FeatureSelector(
        target_column="RUL",
        drop_columns=["unit_number"],
    ),

    scaler=FeatureScaler(),

)

X_train, X_val, y_train, y_val = pipeline.prepare(df)
```

---

# Complete Data Preparation Workflow

```
Engineered Dataset
        │
        ▼
Feature Selection
        │
        ▼
Engine-Based Train/Validation Split
        │
        ▼
Feature Scaling
        │
        ▼
Training Ready Dataset
```

---

# Data Leakage Prevention

Preventing data leakage was one of the primary goals of this sprint.

The following safeguards were implemented:

- Target values are separated before training.
- Complete engines are treated as independent groups.
- `GroupShuffleSplit` ensures no engine appears in both datasets.
- Feature scaling is fitted only on the Training dataset.
- Validation data is never used during any fitting process.

These design decisions make the evaluation process much closer to real deployment, where predictions are performed on completely unseen engines.

---

# Unit Tests

The following unit tests were implemented.

---

## FeatureSelector

- Feature/Target separation
- Target removal
- Column removal
- Invalid inputs

---

## DataSplitter

- Returns Training and Validation DataFrames
- No overlapping engines
- Row count preserved
- Invalid split ratio
- Missing engine column
- Empty DataFrame

---

## FeatureScaler

- Fit
- Transform
- Fit + Transform
- Save scaler
- Transform before fitting
- Empty DataFrame
- Shape preservation
- Column preservation
- Index preservation

---

## DataPreparationPipeline

- Complete pipeline execution
- Correct output types
- Target removed
- Engine identifier removed
- Matching feature and target dimensions

---

# Notebook

Created

```
notebooks/09_data_preparation.ipynb
```

The notebook demonstrates the complete preprocessing workflow:

- Loading the dataset
- Data validation
- RUL generation
- Feature engineering
- Feature selection
- Engine-based dataset splitting
- Feature scaling
- Final training-ready datasets

---

# Integration Test

A complete integration test was created.

```
pipeline_test.py
```

The script executes the complete preprocessing pipeline.

```
Load Data

↓

Validate Data

↓

Generate RUL

↓

Feature Engineering

↓

Feature Selection

↓

Group-Based Train/Validation Split

↓

Feature Scaling

↓

Training Ready Dataset
```

This provides a quick regression test whenever modifications are made to the preprocessing pipeline.

---

# Engineering Decisions

## Single Responsibility Principle

Each class has one clear responsibility.

| Class | Responsibility |
|--------|---------------|
| FeatureSelector | Separate features and target |
| DataSplitter | Group-aware dataset splitting |
| FeatureScaler | Feature normalization |
| DataPreparationPipeline | Coordinate preprocessing workflow |

---

## Group-Aware Splitting

Instead of implementing a custom splitting algorithm, the project relies on scikit-learn's **GroupShuffleSplit**.

Benefits:

- Prevents engine-level data leakage.
- Uses a well-tested implementation.
- Easily extendable to GroupKFold and grouped cross-validation.
- Produces reproducible experiments through a configurable random seed.

---

## Modularity

Each preprocessing component can be independently:

- Tested
- Reused
- Extended
- Replaced

without affecting the remaining pipeline.

---

## Extensibility

The preprocessing pipeline was designed to support future enhancements, including:

- Additional feature selection techniques.
- Alternative scaling strategies.
- PCA.
- Group-aware cross-validation.
- Hyperparameter optimization pipelines.
- End-to-end sklearn Pipelines.

---

# Sprint 7 Deliverables

Completed components:

- ✅ FeatureSelector
- ✅ DataSplitter
- ✅ FeatureScaler
- ✅ DataPreparationPipeline
- ✅ Integration Pipeline
- ✅ Unit Tests
- ✅ Demonstration Notebook
- ✅ Sprint Documentation

---

# Project Progress

```
Sprint 1  ✅ Data Understanding

Sprint 2  ✅ Data Loading & Validation

Sprint 3  ✅ Exploratory Data Analysis

Sprint 4  ✅ Target Engineering

Sprint 5  ✅ Feature Engineering Design

Sprint 6  ✅ Traditional Feature Engineering

Sprint 7  ✅ Data Preparation

Sprint 8  🚀 Baseline Model Training
```

---

# Next Sprint

## Sprint 8 — Baseline Model Training

The next sprint focuses on building the first traditional machine learning models.

Planned models:

- Random Forest
- XGBoost
- LightGBM
- CatBoost

These models will use the complete preprocessing pipeline developed during Sprints 2–7.

The objective is to establish strong baseline performance before introducing explainability techniques and deep learning architectures.