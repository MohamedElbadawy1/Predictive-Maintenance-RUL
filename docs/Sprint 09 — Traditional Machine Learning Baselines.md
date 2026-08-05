# Sprint 8 — Traditional Machine Learning Baselines

---

# Objective

The goal of this sprint was to build the first complete end-to-end machine learning training pipeline for the NASA C-MAPSS Remaining Useful Life (RUL) prediction project.

This sprint focuses on establishing strong baseline models using traditional machine learning algorithms before moving to deep learning approaches such as LSTM and GRU.

Rather than optimizing model performance immediately, the objective is to create a reusable, extensible, and maintainable training framework.

---

# Sprint Scope

This sprint includes:

- Regression evaluation framework
- Generic model training interface
- Model factory
- Traditional ML benchmarking framework
- Training four baseline models
- Model comparison
- Experiment artifact generation

---

# Project Structure

```
src/
│
├── evaluation/
│   └── regression_evaluator.py
│
├── models/
│   ├── base_trainer.py
│   ├── model_factory.py
│   ├── traditional_ml_benchmark.py
│
├── experiments/
│   └── experiment_tracker.py
```

---

# Components

## 1. RegressionEvaluator

### Purpose

Provides a unified interface for evaluating regression models.

### Calculated Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Mean Absolute Percentage Error (MAPE)

### Design Decision

Every model uses exactly the same evaluation class.

This guarantees fair comparisons between different algorithms and avoids duplicated metric calculations.

---

## 2. BaseTrainer

### Purpose

Provides a common interface for all traditional machine learning models.

### Responsibilities

- Train a model
- Generate predictions
- Save trained models
- Load trained models

### Design Decision

Instead of creating a separate trainer for every algorithm, BaseTrainer wraps any scikit-learn compatible estimator.

This significantly reduces duplicated code.

---

## 3. ModelFactory

### Purpose

Creates machine learning models using a common interface.

Example:

```python
model = ModelFactory.create("catboost")
```

Supported models:

- Random Forest
- XGBoost
- LightGBM
- CatBoost

### Design Decision

The benchmark never directly imports individual models.

Adding a new model only requires updating ModelFactory.

---

## 4. TraditionalMLBenchmark

### Purpose

Automates benchmarking across multiple traditional ML algorithms.

Workflow:

```
Create Model

↓

Train

↓

Predict

↓

Evaluate

↓

Store Metrics

↓

Compare Models
```

Returns

- Metrics DataFrame
- Best Trainer

### Design Decision

Training logic is implemented once and reused for every model.

---

## 5. ExperimentTracker

### Purpose

Automatically saves experiment artifacts.

Artifacts include:

- benchmark_results.csv
- best_model.pkl
- best_model_name.txt
- feature_importance.csv
- summary.json

Each experiment is stored inside a timestamped directory.

Example:

```
artifacts/

└── experiments/

    └── 2026-08-04_15-30-22/

        benchmark_results.csv

        feature_importance.csv

        best_model.pkl

        best_model_name.txt

        summary.json
```

### Design Decision

Every experiment should be reproducible.

Instead of overwriting results, each run is preserved independently.

---

# Baseline Models

The following algorithms were trained without hyperparameter tuning.

- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

These models serve as reference baselines for future optimization.

---

# Baseline Results

| Model | MAE | RMSE | R² | MAPE | Training Time (s) |
|------|------:|------:|------:|------:|------:|
| CatBoost | **12.90** | **19.11** | **0.7814** | **24.99** | 31.44 |
| LightGBM | 13.21 | 19.76 | 0.7662 | 26.32 | 8.35 |
| XGBoost | 13.36 | 19.78 | 0.7659 | 26.80 | 6.06 |
| Random Forest | 13.45 | 19.91 | 0.7628 | 28.35 | 159.15 |

---

# Observations

## CatBoost

- Best overall performance.
- Highest R².
- Lowest MAE.
- Lowest RMSE.
- Lowest MAPE.

CatBoost is selected as the baseline model for future experiments.

---

## Random Forest

Although Random Forest produced competitive results, its training time was significantly longer than the boosting-based models while providing lower predictive performance.

It is unlikely to be the preferred model for future optimization.

---

## Gradient Boosting Models

CatBoost, LightGBM, and XGBoost consistently outperformed Random Forest.

This confirms that gradient boosting methods are well suited for the NASA C-MAPSS regression problem.

---

# Engineering Decisions

## Unified Training Interface

Every traditional ML model follows the same workflow.

```
Model

↓

Train

↓

Predict

↓

Evaluate
```

This avoids duplicated training code.

---

## Model Factory Pattern

Model creation is centralized.

Future models can be added without modifying the benchmark pipeline.

---

## Separate Experiment Tracking

Experiment management is isolated from training.

The benchmark focuses only on training and evaluation, while ExperimentTracker is responsible for persistence.

This follows the Single Responsibility Principle.

---

## Reproducibility

Every experiment is stored independently using timestamped directories.

This allows future comparisons between:

- Feature engineering versions
- Hyperparameter tuning experiments
- Deep learning models
- Different datasets

without overwriting previous results.

---

# Limitations

The models were intentionally trained using default hyperparameters.

No optimization was performed in this sprint.

The objective was to establish reliable baselines rather than maximize predictive performance.

---

# Deliverables

Completed:

- RegressionEvaluator
- BaseTrainer
- ModelFactory
- TraditionalMLBenchmark
- ExperimentTracker
- Four baseline models
- Model comparison
- Automatic experiment artifact generation

---

# Next Sprint

Sprint 9 — Model Analysis & Optimization

Planned tasks:

- Residual analysis
- Prediction error analysis
- Feature importance analysis
- SHAP explainability
- Hyperparameter optimization using Optuna
- Benchmark comparison with optimized models

The optimized traditional models will serve as the final benchmark before transitioning to sequence-based deep learning models (LSTM and GRU).