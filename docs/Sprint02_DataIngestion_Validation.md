# Sprint 2 – Data Ingestion & Data Validation

## Objective

Build the first stage of the machine learning pipeline by implementing:

- DataLoader
- DataValidator

This stage focuses **only** on loading and validating the raw NASA CMAPSS datasets.

No preprocessing, cleaning, feature engineering, or RUL calculation is performed at this stage.

---

# Project Components

## 1. DataLoader (`src/data/loader.py`)

### Responsibilities

- Read the training dataset.
- Read the testing dataset.
- Read the RUL dataset.
- Assign the correct column names.
- Return ready-to-use Pandas DataFrames.
- Log loading operations.
- Raise a custom exception if loading fails.

### Not Responsible For

- Cleaning data
- Feature engineering
- Data validation
- Computing Remaining Useful Life (RUL)

---

## 2. DataValidator (`src/data/validator.py`)

### Responsibilities

Validate the integrity and quality of the loaded datasets without modifying them.

### Current Validation Checks

### Dataset Structure

- File successfully loaded
- Correct number of columns
- Correct column names

### Data Quality

- Missing values
- Duplicate rows
- Numeric data types

### Domain Validation

- `unit_number` contains only positive values
- `time_in_cycles` contains only positive values
- Every engine starts from cycle 1

---

# Design Philosophy

The validator follows a **read-only** approach.

Its responsibility is to **detect** problems, not fix them.

If issues are found, they are reported through a validation report.

Cleaning and preprocessing will be implemented in later pipeline stages.

---

# Validation Report Format

Each dataset returns a report with the following structure:

```python
{
    "valid": bool,
    "errors": list[str],
    "warnings": list[str]
}
```

Example:

```python
{
    "valid": True,
    "errors": [],
    "warnings": []
}
```

---

# Logging

Both DataLoader and DataValidator use the custom logging system.

Example:

```text
2026-07-16 19:15:47 | INFO | loader.py | Reading train_FD004.txt
2026-07-16 19:15:48 | INFO | loader.py | train_FD004.txt Loaded Successfully
2026-07-16 19:15:49 | INFO | validator.py | Validating training dataset...
```

This provides complete traceability during pipeline execution.

---

# Test Run

Pipeline executed successfully using the **FD004** dataset.

Console output:

```text
2026-07-16 19:15:47 | INFO | loader.py | Reading train_FD004.txt
2026-07-16 19:15:48 | INFO | loader.py | train_FD004.txt Loaded Successfully
2026-07-16 19:15:48 | INFO | loader.py | Reading test_FD004.txt
2026-07-16 19:15:49 | INFO | loader.py | test_FD004.txt Loaded Successfully
2026-07-16 19:15:49 | INFO | loader.py | Reading RUL_FD004.txt
2026-07-16 19:15:49 | INFO | loader.py | RUL_FD004.txt Loaded Successfully
2026-07-16 19:15:49 | INFO | validator.py | Validating training dataset...
2026-07-16 19:15:49 | INFO | validator.py | Validating testing dataset...
2026-07-16 19:15:49 | INFO | validator.py | Validating RUL dataset...
```

Validation report:

```python
{
    "train": {
        "valid": True,
        "errors": [],
        "warnings": []
    },
    "test": {
        "valid": True,
        "errors": [],
        "warnings": []
    },
    "rul": {
        "valid": True,
        "errors": [],
        "warnings": [
            "Duplicate rows found."
        ]
    }
}
```

---

# Observations

The training and testing datasets passed all validation checks.

The RUL dataset reported duplicate rows.

This is treated as a warning rather than an error because duplicate RUL values can occur naturally in the NASA CMAPSS dataset and do not necessarily indicate corrupted data. No modification is performed during validation.

---

# Current Pipeline

```
Raw NASA Dataset
        │
        ▼
+----------------+
|  DataLoader    |
+----------------+
        │
        ▼
+----------------+
| DataValidator  |
+----------------+
        │
        ▼
Validation Report
```

---

# Next Stage

Sprint 3 will focus on **Exploratory Data Analysis (EDA)**.
