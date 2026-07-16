# Data Understanding — NASA C-MAPSS (FD004)

**Sprint 1 — Deliverable**
**Dataset source:** [Kaggle — NASA Turbofan Engine Degradation Simulation](https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation)

> This document is purely for understanding. It contains no model training, no plots, and no XGBoost.
> The only goal here: **fully understand the data before writing a single line of modeling code.**

---

## 1. Dataset Overview

### 1.1 What is C-MAPSS?

**C-MAPSS** stands for **Commercial Modular Aero-Propulsion System Simulation**.
It's a simulation tool developed by NASA (specifically NASA's Prognostics Center of Excellence) to realistically simulate the performance and degradation of commercial turbofan aircraft engines.

- The simulated engine is a ~90,000 lb thrust-class turbofan, made up of five main rotating components:
  - **Fan**
  - **LPC** — Low Pressure Compressor
  - **HPC** — High Pressure Compressor
  - **HPT** — High Pressure Turbine
  - **LPT** — Low Pressure Turbine

- The dataset was originally introduced in the paper:
  *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"* — Saxena, Goebel, Simon & Eklund (2008), presented at the PHM 2008 Data Challenge.

- The idea: each engine (unit) starts with a different degree of initial wear and unknown manufacturing variation, then runs through consecutive operational cycles until it degrades and eventually reaches **failure**. Throughout this process, sensor readings and operational settings are recorded at every cycle.

- This is **not real data from physical engines** — it's **simulated** data, but designed to be statistically realistic. That's why it became the de facto global **benchmark** for Prognostics and Health Management (PHM) research and RUL prediction.

---

### 1.2 What is RUL?

**RUL = Remaining Useful Life**.

- It's the number of remaining **operational cycles** before an engine reaches failure.
- It's the **target/label** that any model (later on) tries to predict.
- In the training files: we know the full lifetime of each engine (from start to failure), so RUL for any row = (total cycles until failure) − (current cycle number).
- In the test files: the data is **truncated** at a random point before failure, and the task is to predict RUL at the last available cycle, then compare it against the true value provided in `RUL_FD004.txt`.

This is the core of the problem: a **regression** task aimed at predicting a number (remaining cycles) from a sequence of sensor readings.

---

### 1.3 Why are there four datasets (FD001–FD004)?

Because NASA wanted to test algorithms under different levels of **complexity**, based on two key variables:

| Variable | Possible values |
|---|---|
| Number of Operating Conditions | 1 or 6 |
| Number of Fault Modes | 1 or 2 |

Combining these gives 4 combinations = 4 datasets.

### 1.4 What's the difference between them?

| Dataset | Operating Conditions | Fault Modes | Train Units | Test Units |
|---|---|---|---|---|
| **FD001** | 1 | 1 (HPC degradation) | 100 | 100 |
| **FD002** | 6 | 1 (HPC degradation) | 260 | 259 |
| **FD003** | 1 | 2 | 100 | 100 |
| **FD004** | 6 | 2 | 248/249 | 248/249 |

- **Fault Mode 1** is typically: degradation in the HPC (High Pressure Compressor).
- **Fault Mode 2**: an additional degradation (often in the Fan) occurring in parallel with the first.
- More **Operating Conditions** means sensor readings vary due to operating context (altitude, speed, etc.) — not only due to degradation — which makes it harder to separate the "true degradation signal" from "noise caused by changing conditions."

### 1.5 Why did we choose FD004 for our project?

- FD004 is the **hardest and most realistic** of the four: 6 different operating conditions + 2 simultaneous fault modes.
- Relatively larger dataset (~248 training engines and a large number of rows), giving more room to learn from.
- A real challenge: the model has to learn to separate the effect of "changing operating conditions" from "actual degradation" — which mirrors exactly what real predictive-maintenance systems face in aircraft.
- Choosing FD004 means any good result we get later carries **higher research and practical value** than choosing the easier FD001.

---

## 2. Files

### 2.1 `train_FD004.txt`

- Contains complete **run-to-failure** data: each engine (`unit_id`) is recorded from the start of its operation all the way to actual failure.
- Since we know the failure point, RUL for every row can be computed retroactively (no need for an external file).
- Will be used for model training later.

### 2.2 `test_FD004.txt`

- Same format as the train file, but each engine here is **truncated** at a random point **before** reaching failure.
- We don't know from the file itself how many cycles remain for each engine — that's exactly what we'll predict later.

### 2.3 `RUL_FD004.txt`

- A small separate file, with number of lines equal to the number of engines in `test_FD004.txt`.
- Each line represents the **ground truth** RUL value at the last recorded cycle for each engine in the test file.
- Used only for evaluation after a future prediction step — not for training.

---

## 3. Features

### 3.1 How many columns?

Each row in the train/test files contains **26 columns**, with no header inside the file, and values separated by spaces.

### 3.2 What does each column mean?

| # | Column name | Description |
|---|---|---|
| 1 | `unit_number` | Engine ID |
| 2 | `time_in_cycles` | Current operational cycle number for this engine |
| 3 | `operational_setting_1` | Operational setting 1 |
| 4 | `operational_setting_2` | Operational setting 2 |
| 5 | `operational_setting_3` | Operational setting 3 |
| 6–26 | `sensor_measurement_1` … `sensor_measurement_21` | 21 readings from different sensors |

### 3.3 What are the Operational Settings?

Three values describing the **flight/operating context** the engine was running under at that moment, corresponding to concepts such as:
- **Altitude**
- **Mach Number** (speed)
- **Throttle Resolver Angle (TRA)**

In FD001 and FD003 these are nearly constant throughout (single operating condition), while in FD002 and FD004 (including our project) they **vary across 6 different combinations** — a major source of complexity and noise that will need to be handled later (naturally, but not now — just understanding for now).

### 3.4 What are the Sensors?

21 numeric readings representing the engine's physical state, mostly temperatures, pressures, and rotational speeds at different locations in the engine:

| Sensor | Symbol | Description | Unit |
|---|---|---|---|
| s1 | T2 | Total temperature at fan inlet | °R |
| s2 | T24 | Total temperature at LPC outlet | °R |
| s3 | T30 | Total temperature at HPC outlet | °R |
| s4 | T50 | Total temperature at LPT outlet | °R |
| s5 | P2 | Pressure at fan inlet | psia |
| s6 | P15 | Total pressure in bypass-duct | psia |
| s7 | P30 | Total pressure at HPC outlet | psia |
| s8 | Nf | Physical fan speed | rpm |
| s9 | Nc | Physical core speed | rpm |
| s10 | epr | Engine pressure ratio | — |
| s11 | Ps30 | Static pressure at HPC outlet | psia |
| s12 | phi | Ratio of fuel flow to Ps30 | pps/psi |
| s13 | NRf | Corrected fan speed | rpm |
| s14 | NRc | Corrected core speed | rpm |
| s15 | BPR | Bypass ratio | — |
| s16 | farB | Burner fuel-air ratio | — |
| s17 | htBleed | Bleed enthalpy | — |
| s18 | Nf_dmd | Demanded fan speed | rpm |
| s19 | PCNfR_dmd | Demanded corrected fan speed | rpm |
| s20 | W31 | HPT coolant bleed | lbm/s |
| s21 | W32 | LPT coolant bleed | lbm/s |

### 3.5 Are all sensors useful?

**No.** This is an important observation, confirmed across multiple research papers that used the same dataset:

- Some sensors have values that are **nearly constant / flat** throughout an engine's lifetime and don't reflect actual degradation, so they carry little to no useful information for predicting RUL.
- The sensors commonly reported as constant/uninformative (based on multiple studies using this dataset) are: **s1, s5, s6, s10, s16, s18, s19**.
- The rest (~14 sensors) tend to show a clear increasing or decreasing trend correlated with degradation progress, making them strong candidates to be used as features later.
- **Important note:** this is a theoretical observation drawn from the literature, not a result of actually analyzing our own data yet. In the next Sprint (EDA/Statistics) this needs to be verified directly on our specific FD004 copy (since some papers analyzed this on FD001 only) — **without plotting anything yet if we want to keep the strict rule, just via `.describe()`, `.nunique()`, or `.std()`** if a quick verification step is needed.

---

## 4. Summary

- **C-MAPSS**: NASA's simulation of turbofan aircraft engine degradation.
- **RUL**: Remaining cycles before failure — the target variable.
- We chose **FD004** because it's the most realistic and complex (6 operating conditions × 2 fault modes).
- 3 files: `train` (complete), `test` (truncated), `RUL` (ground truth for evaluation).
- 26 columns: `unit_number`, `time_in_cycles`, 3 operational settings, 21 sensor measurements.
- Not all sensors are useful — some are nearly constant and carry no degradation signal, to be verified empirically in the next step.

---

## References

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. PHM 2008.
- Kaggle Dataset: NASA Turbofan Engine Degradation Simulation — https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation
- NASA Prognostics Center of Excellence — C-MAPSS Data Set documentation
