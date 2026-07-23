# Sprint 4 — Target Engineering

## Overview

Sprint 4 focuses exclusively on generating the prediction target for the NASA CMAPSS training dataset.

Unlike many machine learning datasets, the CMAPSS training data does not contain a Remaining Useful Life (RUL) column. Therefore, the target variable must be computed before any preprocessing or model development.

To ensure modularity, reusability, and maintainability, the target generation process was isolated into a dedicated `RULGenerator` class.

This design allows multiple target representations (e.g., raw, capped, normalized, or piecewise RUL) to be generated through a single, reusable interface.

---

# Objectives

The objectives of this sprint were to:

- Generate the Remaining Useful Life (RUL) target.
- Support both raw and capped RUL generation.
- Preserve the original dataset without modification.
- Design a reusable and extensible target engineering component.
- Validate the correctness of the generated target.
- Document the engineering decisions before moving to preprocessing.

---

# Understanding Remaining Useful Life (RUL)

Remaining Useful Life (RUL) represents the number of operating cycles remaining before an engine reaches failure.

For each engine:

RUL = Maximum Engine Cycle − Current Cycle

For example:

| Engine | Cycle | RUL |
|---------|------:|----:|
| 1 | 1 | 191 |
| 1 | 2 | 190 |
| 1 | 3 | 189 |
| ... | ... | ... |
| 1 | 192 | 0 |

Every engine therefore finishes with an RUL of zero.

---

# Why the Dataset Does Not Include RUL

The NASA CMAPSS training dataset contains complete run-to-failure trajectories.

Since every engine is observed until failure, the Remaining Useful Life can be computed directly from the recorded engine cycles.

This allows researchers to generate different target representations depending on the modeling strategy.

---

# RULGenerator Class

Location:

```text
src/preprocessing/rul_generator.py
```

## Responsibilities

The `RULGenerator` class is responsible for:

- Computing raw RUL.
- Applying optional RUL capping.
- Returning a new DataFrame.
- Preserving the original DataFrame.
- Validating input data.
- Logging target generation events.

The class is **not** responsible for:

- Feature scaling.
- Data cleaning.
- Feature engineering.
- Saving datasets.
- Model training.

This separation follows the **Single Responsibility Principle (SRP)**.

---

# Public API

```python
generator = RULGenerator(train_df)

raw_df = generator.generate()

capped_df = generator.generate(cap=125)
```

The interface remains simple while supporting future extensions.

---

# Raw RUL

Raw RUL preserves the complete Remaining Useful Life for every engine.

Example:

| Cycle | Raw RUL |
|------:|--------:|
| 1 | 191 |
| 2 | 190 |
| ... | ... |
| 192 | 0 |

This representation preserves the full degradation trajectory.

---

# Capped RUL

Very large Remaining Useful Life values are difficult to predict because healthy engines exhibit little observable degradation.

To reduce unnecessary variation, an upper limit can be applied.

Example (Cap = 125):

| Cycle | Raw RUL | Capped RUL |
|------:|--------:|-----------:|
| 1 | 191 | 125 |
| 2 | 190 | 125 |
| ... | ... | ... |
| 70 | 125 | 125 |
| 71 | 124 | 124 |
| ... | ... | ... |
| 192 | 0 | 0 |

The capped representation introduces a plateau during the healthy operating phase while preserving the degradation region.

---

# Why Choose a Cap of 125?

Exploratory Data Analysis (Sprint 3) showed that:

- Very high RUL values are comparatively rare.
- Healthy engines exhibit minimal degradation signals.
- Distinguishing between extremely healthy engines (e.g., 300 vs. 450 cycles remaining) provides limited practical value.
- Maintenance decisions primarily depend on accurately predicting engines approaching failure.

Applying a cap of **125 cycles** allows the model to concentrate on the degradation region while reducing the influence of extreme target values.

This choice is also consistent with many published studies using the NASA CMAPSS dataset.

---

# Validation

The generated target was validated by verifying:

- Every engine ends with RUL = 0.
- RUL decreases by one at every operating cycle.
- Raw RUL correctly follows the engine lifetime.
- Capped RUL preserves the degradation region.
- The original DataFrame remains unchanged.
- The generated DataFrame contains a new `RUL` column.

---

# Software Engineering Decisions

## Decision 1 — Preserve the Original Dataset

The original training DataFrame is never modified.

The generator always returns a new DataFrame.

---

## Decision 2 — Optional Target Capping

RUL capping is optional.

```python
generator.generate()
```

returns raw RUL, while

```python
generator.generate(cap=125)
```

returns capped RUL.

---

## Decision 3 — Input Validation

The class validates:

- Empty DataFrames.
- Missing required columns.
- Invalid cap values.

Invalid inputs raise a `CustomException`.

---

## Decision 4 — Append the Target Column

The generated `RUL` column is appended as the final column of the DataFrame.

This preserves the original dataset structure and simplifies feature-target separation.

---

## Decision 5 — Extensible Design

The class was designed to support future target representations without changing its public interface.

Possible future extensions include:

- Normalized RUL.
- Piecewise Linear RUL.
- Classification labels.
- Multi-task targets.

---

# Notebook Deliverable

Notebook:

```text
notebooks/08_rul_generation.ipynb
```

The notebook demonstrates:

- Raw RUL generation.
- Capped RUL generation.
- Distribution comparison.
- Validation examples.
- Engineering observations.
- Target engineering decisions.

---

# Sprint Deliverables

Completed components:

- ✅ `src/preprocessing/rul_generator.py`
- ✅ `notebooks/08_rul_generation.ipynb`
- ✅ Target validation
- ✅ Raw RUL generation
- ✅ Capped RUL generation
- ✅ Sprint documentation

---

# Sprint Outcome

Sprint 4 successfully established a reusable and extensible target engineering pipeline for the NASA CMAPSS project.

The Remaining Useful Life target can now be generated consistently across experiments while preserving the integrity of the original dataset.

This modular design provides a solid foundation for the next sprint, where preprocessing and feature engineering components will be integrated into a complete machine learning pipeline.
