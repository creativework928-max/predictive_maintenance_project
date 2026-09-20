# Predictive Maintenance — Task 2: Feature Engineering

A professional, leakage-safe, end-to-end feature-engineering pipeline for predictive maintenance using the **UCI AI4I 2020 Predictive Maintenance Dataset**.

The project is designed around production-oriented principles:

* reproducible feature engineering
* leakage-safe train/validation/test processing
* physically meaningful engineering features
* statistical feature-quality analysis
* professional visualizations
* automated HTML reporting
* automated DOCX documentation
* offline and real-time/online feature generation
* deterministic execution
* schema validation
* export of engineered datasets
* extensibility to real industrial time-series data

---

## 1. Project Overview

Predictive maintenance aims to identify abnormal machine behavior and predict failures before they occur.

The feature-engineering stage is critical because raw sensor measurements often do not contain enough information for a machine-learning model to distinguish:

* normal operating conditions
* gradual degradation
* abnormal operating states
* rapid changes in sensor measurements
* relationships between physical variables
* machine operating regimes
* potential failure precursors

This project implements **Task 2: Feature Engineering** as a reusable pipeline rather than as a collection of isolated notebook transformations.

The pipeline takes raw machine data, validates the schema, creates engineered features, removes invalid or redundant features, evaluates feature quality, generates visualizations, and exports datasets suitable for subsequent machine-learning tasks.

---

# 2. Dataset

## UCI AI4I 2020 Predictive Maintenance Dataset

The primary reference dataset is:

**AI4I 2020 Predictive Maintenance Dataset**

Dataset source:

**UCI Machine Learning Repository — Dataset ID 601**

Official source:

https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdat

The dataset contains **10,000 observations** and **14 original variables** and is intended for predictive-maintenance classification/regression experiments.

The dataset is synthetic rather than a direct collection of industrial machine telemetry. Therefore, this project deliberately avoids incorrectly treating the `UID` field as a genuine physical timestamp.

---

# 3. Important Dataset Interpretation

A major design decision in this project is the distinction between:

### Reference/AI4I mode

The AI4I dataset contains physically meaningful machine variables such as:

* air temperature
* process temperature
* rotational speed
* torque
* tool wear
* product type

These variables can be used to construct engineering features based on machine physics and operating conditions.

However, the `UID` field should **not automatically be interpreted as a true chronological machine timestamp**.

Therefore, the pipeline does not create artificial temporal lags simply by sorting the dataset by `UID`.

### Real industrial time-series mode

For a real deployment dataset, the preferred schema contains at least:

```text
machine_id
timestamp
sensor_1
sensor_2
...
target
```

With genuine timestamps and machine identifiers, the pipeline can safely generate:

* lag features
* rolling statistics
* exponentially weighted statistics
* rates of change
* temporal anomaly features
* machine-specific operating histories

All temporal features must be calculated using observations available **at or before the prediction time**.

---

# 4. Task 2 Objectives

The feature-engineering pipeline covers the following feature families.

## 4.1 Time-Based Features

For datasets containing a genuine timestamp, the pipeline can derive features such as:

* year
* month
* day
* day of week
* hour
* minute
* elapsed time
* cyclical hour representation
* cyclical day-of-week representation

Example:

```text
hour_sin
hour_cos
dow_sin
dow_cos
```

Cyclical encoding prevents artificial discontinuities such as:

```text
23 → 0
```

being interpreted as a large numerical jump.

---

# 5. Lag Features

For real machine time-series data, historical sensor measurements can be transformed into lag variables:

```text
sensor_lag_1
sensor_lag_2
sensor_lag_3
sensor_lag_5
sensor_lag_10
```

These features allow a model to learn relationships such as:

> "The current machine condition depends partly on the sensor state several observations ago."

Lag features are generated separately for each machine where a `machine_id` is available.

---

# 6. Rolling Statistics

The pipeline supports rolling-window statistics including:

### Rolling mean

```text
rolling_mean_w5
rolling_mean_w10
rolling_mean_w20
```

### Rolling standard deviation

```text
rolling_std_w5
rolling_std_w10
rolling_std_w20
```

### Rolling minimum

```text
rolling_min_w5
```

### Rolling maximum

```text
rolling_max_w5
```

### Rolling median

```text
rolling_median_w5
```

Rolling statistics capture local operating behavior rather than relying only on individual sensor measurements.

---

# 7. Exponentially Weighted Moving Averages

Exponentially weighted moving averages give greater importance to recent observations.

Example:

```text
ewm_mean_alpha_01
ewm_mean_alpha_03
```

These features are particularly useful for detecting gradual machine degradation.

---

# 8. Sensor Difference Features

Differences between physically related sensors can reveal machine behavior that individual sensors cannot capture.

For AI4I:

```text
temperature_gap = process_temperature - air_temperature
```

This is an important physically meaningful feature.

Additional sensor differences can be created when appropriate columns exist.

---

# 9. Sensor Ratio Features

Ratios can describe relative operating conditions.

Examples include:

```text
temperature_ratio
torque_speed_ratio
```

Ratio generation must be protected against:

* division by zero
* extremely small denominators
* infinite values
* invalid numerical values

Invalid ratios are converted to missing values and subsequently handled by the feature-quality pipeline.

---

# 10. Physical Engineering Features

The project includes domain-informed physical transformations.

## 10.1 Mechanical Power

Mechanical power is derived from torque and rotational speed.

The standard relationship is:

```text
P = τ × ω
```

where:

```text
ω = 2π × RPM / 60
```

Therefore:

```text
mechanical_power_w =
    torque_nm * 2π * rotational_speed_rpm / 60
```

This feature provides a physically interpretable representation of machine loading.

---

# 11. Torque × Rotational-Speed Interaction

A direct interaction feature is also useful:

```text
torque_speed_interaction =
    torque_nm * rotational_speed_rpm
```

This captures joint effects between mechanical load and rotational speed.

---

# 12. Tool-Wear Transformations

Tool wear is an important degradation-related variable.

The pipeline supports transformations such as:

```text
tool_wear_squared
tool_wear_sqrt
tool_wear_log1p
```

These transformations allow downstream models to learn nonlinear relationships.

---

# 13. Rate-of-Change Features

For genuine time-series data, rate-of-change features can identify rapidly changing operating conditions.

Examples:

```text
sensor_diff_1
sensor_pct_change_1
sensor_rate_of_change
```

A rate-of-change feature can be expressed as:

```text
(current_value - previous_value) / elapsed_time
```

The implementation must account for:

* missing previous values
* zero elapsed time
* irregular sampling
* machine boundaries

---

# 14. Z-Score and Anomaly Features

The project supports statistical anomaly indicators.

A standard z-score is:

```text
z = (x - μ) / σ
```

where:

* `x` = current measurement
* `μ` = reference mean
* `σ` = reference standard deviation

Large absolute values can indicate unusual machine behavior.

For leakage-safe processing, statistics used to construct global anomaly features are fitted using the training data and then applied to validation/test data.

---

# 15. Interaction Features

Feature interactions allow the model to represent combinations of operating variables.

Examples include:

```text
torque × rotational_speed
temperature × torque
temperature × tool_wear
speed × tool_wear
```

Only meaningful interactions should be retained. Creating every possible pair of columns can create unnecessary dimensionality and increase overfitting risk.

---

# 16. Machine Operating Conditions

The pipeline can derive operating-condition features such as:

* low-speed indicator
* high-speed indicator
* high-torque indicator
* high-tool-wear indicator
* high-temperature indicator
* power operating regime
* temperature operating regime

These features transform continuous measurements into interpretable operating states.

---

# 17. Missing-Value Handling

Real production data commonly contains missing values.

The feature-quality pipeline therefore checks for:

* missing values
* missing-value percentage
* infinite values
* invalid numerical values
* columns with excessive missingness

The feature-engineering process should never silently hide data-quality problems.

Missing values are handled using transformations fitted on the training partition wherever fitting is required.

---

# 18. Constant and Duplicate Feature Removal

Constant features contain no predictive information.

For example:

```text
feature_a = 1
feature_a = 1
feature_a = 1
...
```

Such a feature should be removed.

The pipeline also checks for duplicate feature columns.

This reduces:

* unnecessary memory usage
* redundant information
* numerical instability
* model complexity

---

# 19. Feature Quality Analysis

The project produces a feature-quality report containing information such as:

* feature name
* data type
* number of unique values
* missing count
* missing percentage
* infinite count
* mean
* standard deviation
* minimum
* maximum
* median
* constant-feature status
* duplicate-feature status

This report helps determine whether engineered features are suitable for downstream modeling.

---

# 20. Correlation Analysis

The project generates correlation visualizations to identify relationships between features.

Correlation analysis is useful for detecting:

* strongly correlated feature groups
* redundant features
* physically meaningful relationships
* possible multicollinearity

Correlation should not automatically be interpreted as causation.

---

# 21. Feature Distribution Visualization

The visualization module produces professional distribution plots for selected features.

Typical visualizations include:

* histograms
* density distributions
* boxplots
* target-conditioned distributions

The plots are saved under:

```text
reports/figures/
```

---

# 22. Rolling-Statistics Visualization

Where genuine time-series data is available, rolling statistics can be visualized against time.

Typical examples:

```text
raw sensor
rolling mean
rolling standard deviation
```

These plots help demonstrate how the engineered features capture machine behavior over time.

---

# 23. Feature Importance Visualization

Feature importance visualization is included as a diagnostic step.

Importance may be calculated using a suitable tree-based model or another supported estimator.

The purpose at this stage is not to finalize the production model.

Instead, feature importance is used to answer questions such as:

* Which engineered features contain useful signal?
* Are physical features informative?
* Are temporal features useful?
* Are there suspiciously predictive leakage variables?
* Are redundant features dominating the model?

---

# 24. Leakage Prevention

Leakage prevention is one of the most important design principles of this project.

The following rule is enforced:

> A feature at prediction time must only use information that would have been available at prediction time.

Examples of dangerous leakage include:

```text
future sensor value
future rolling mean
future failure status
failure-mode labels
post-failure measurements
statistics calculated using the complete dataset
```

The AI4I dataset includes failure-mode indicators such as:

```text
TWF
HDF
PWF
OSF
RNF
```

These describe specific failure mechanisms and can directly reveal the target.

Therefore, they must not be used as ordinary predictive features when predicting:

```text
Machine failure
```

---

# 25. Train / Validation / Test Processing

The pipeline follows a leakage-safe workflow:

```text
Raw data
   │
   ├── Train
   ├── Validation
   └── Test
          │
          ▼
Feature engineering
          │
          ▼
Quality checks
          │
          ▼
Exports
```

Transformations that require fitting are fitted using the training data.

The fitted transformation logic is then applied to:

```text
validation
test
```

without recalculating training-dependent parameters from those partitions.

---

# 26. Dataset Splitting

The default configuration is:

```text
Training:    70%
Validation:  15%
Testing:     15%
```

with:

```text
random_state = 42
```

For classification tasks, stratified splitting is used where appropriate.

The resulting files are:

```text
data/processed/train_features.csv
data/processed/validation_features.csv
data/processed/test_features.csv
```

---

# 27. Exported Engineered Dataset

The complete engineered dataset is exported to:

```text
data/processed/engineered_features.csv
```

The partitioned datasets are exported to:

```text
data/processed/train_features.csv
data/processed/validation_features.csv
data/processed/test_features.csv
```

These files can subsequently be consumed by Task 3/model-development workflows.

---

# 28. Real-Time Feature Generation

A dedicated online feature-generation component is included:

```text
src/realtime_features.py
```

Its purpose is to demonstrate how the offline feature definitions can be reproduced when new machine observations arrive.

A real-time system should follow:

```text
Sensor event
     │
     ▼
Validation
     │
     ▼
State update
     │
     ▼
Online feature calculation
     │
     ▼
Model inference
     │
     ▼
Prediction
```

The online implementation must maintain only the historical state required to calculate future features.

It must not recompute future information.

---

# 29. Project Structure

```text
predictive_maintenance_project/
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   │
│   └── processed/
│       ├── engineered_features.csv
│       ├── train_features.csv
│       ├── validation_features.csv
│       └── test_features.csv
│
├── notebooks/
│   └── task_2_feature_engineering.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── feature_quality.py
│   ├── visualization.py
│   ├── report_generator.py
│   └── realtime_features.py
│
├── reports/
│   ├── figures/
│   ├── feature_engineering_report.html
│   └── feature_engineering_documentation.docx
│
├── scripts/
│   └── run_task2.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 30. Installation

## 30.1 Clone or create the project

Place the project in a dedicated directory:

```bash
mkdir predictive_maintenance_project
cd predictive_maintenance_project
```

---

## 30.2 Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 30.3 Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

# 31. Obtaining the AI4I Dataset

The recommended source is the official UCI Machine Learning Repository.

The dataset can be downloaded manually and placed at:

```text
data/raw/ai4i2020.csv
```

Alternatively, it can be retrieved programmatically using the UCI repository package.

Example:

```python
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=601)

X = dataset.data.features
y = dataset.data.targets

data = X.copy()

for column in y.columns:
    data[column] = y[column]

data.to_csv(
    "data/raw/ai4i2020.csv",
    index=False
)
```

The authoritative dataset documentation should always be preferred when verifying column definitions and dataset provenance.

---

# 32. Expected AI4I Columns

The raw AI4I dataset normally contains columns corresponding to:

```text
UDI
Product ID
Type
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
Machine failure
TWF
HDF
PWF
OSF
RNF
```

The project normalizes these into consistent internal names where required.

---

# 33. Running Task 2

The main entry point is:

```bash
python scripts/run_task2.py
```

The script performs the complete Task 2 workflow.

---

# 34. Basic Execution

```bash
python scripts/run_task2.py
```

The default input is:

```text
data/raw/ai4i2020.csv
```

The default processed-data directory is:

```text
data/processed/
```

and reports are generated under:

```text
reports/
```

---

# 35. Specify the Raw Dataset

```bash
python scripts/run_task2.py \
    --raw-data data/raw/ai4i2020.csv
```

On Windows PowerShell:

```powershell
python scripts/run_task2.py `
    --raw-data data/raw/ai4i2020.csv
```

---

# 36. Specify the Output Directory

```bash
python scripts/run_task2.py \
    --output-dir data/processed
```

---

# 37. Configure Dataset Splits

The default split is:

```text
70% train
15% validation
15% test
```

It can be explicitly specified:

```bash
python scripts/run_task2.py \
    --train-size 0.70 \
    --validation-size 0.15 \
    --test-size 0.15 \
    --random-state 42
```

The split proportions must sum to:

```text
1.0
```

---

# 38. Skip Reports

For faster development runs:

```bash
python scripts/run_task2.py --skip-reports
```

---

# 39. Skip Visualizations

For a feature-only execution:

```bash
python scripts/run_task2.py --skip-visualizations
```

---

# 40. Complete Recommended Run

For the complete Task 2 pipeline:

```bash
python scripts/run_task2.py
```

After execution, inspect:

```text
data/processed/
reports/
reports/figures/
```

---

# 41. Expected Outputs

A successful run should produce:

```text
data/processed/
├── engineered_features.csv
├── train_features.csv
├── validation_features.csv
└── test_features.csv
```

Reports:

```text
reports/
├── figures/
├── feature_engineering_report.html
├── feature_engineering_documentation.docx
└── task2_execution_summary.json
```

The exact report filenames may depend on the report-generator configuration.

---

# 42. HTML Report

The HTML report is designed to provide an executive and technical overview of Task 2.

It should contain sections such as:

```text
Executive Summary
Dataset Overview
Feature Engineering Summary
Feature Quality
Missing Values
Feature Correlations
Feature Distributions
Rolling Statistics
Feature Importance
Leakage Checks
Train/Validation/Test Summary
Generated Artifacts
Conclusion
```

The report should be viewable in any modern browser.

Open:

```text
reports/feature_engineering_report.html
```

---

# 43. DOCX Documentation

The DOCX report provides a formal documentation artifact suitable for:

* academic submission
* technical review
* project documentation
* stakeholder review
* engineering handover

Open:

```text
reports/feature_engineering_documentation.docx
```

---

# 44. Notebook

The notebook provides an interactive walkthrough of Task 2:

```text
notebooks/task_2_feature_engineering.ipynb
```

The notebook should mirror the production code rather than containing an independent implementation.

Recommended workflow:

```text
Notebook
   │
   ├── Explore
   ├── Explain
   └── Validate
          │
          ▼
Production source modules
```

This prevents the common problem where notebook code and production code produce different features.

---

# 45. Reproducibility

The pipeline uses deterministic configuration where applicable.

The default random seed is:

```text
42
```

To reproduce an experiment:

```bash
python scripts/run_task2.py \
    --random-state 42
```

The execution summary should record:

* execution timestamp
* input path
* output path
* row counts
* feature counts
* split proportions
* random seed
* detected schema
* generated artifacts
* validation results

---

# 46. Feature Naming Convention

Engineered features should use descriptive, machine-readable names.

Examples:

```text
temperature_gap
mechanical_power_w
torque_speed_interaction
tool_wear_squared
tool_wear_log1p
rotational_speed_lag_1
torque_rolling_mean_w5
torque_rolling_std_w5
```

Feature names should avoid:

* unexplained abbreviations
* spaces where unnecessary
* target information
* future-state terminology
* ambiguous names

---

# 47. Feature Engineering Principles

The project follows these principles:

### Principle 1 — Physical meaning

Whenever possible, engineered variables should correspond to a meaningful machine property.

### Principle 2 — No leakage

Future or post-failure information must never be used.

### Principle 3 — Reproducibility

The same raw input and configuration should produce the same engineered output.

### Principle 4 — Train-only fitting

Statistics requiring fitting are estimated from training data.

### Principle 5 — Online compatibility

Features intended for production should be reproducible from streaming observations.

### Principle 6 — Explicit validation

Feature quality and schema problems should be reported rather than silently ignored.

### Principle 7 — Maintainability

Feature logic belongs in reusable Python modules rather than being buried in a notebook.

---

# 48. AI4I Failure-Mode Leakage Warning

The AI4I dataset contains several failure-mode columns:

```text
TWF
HDF
PWF
OSF
RNF
```

These are closely related to the target.

For a model whose target is:

```text
Machine failure
```

these fields should generally be excluded from predictive input features unless the modeling objective explicitly requires them.

Using them without justification can produce an artificially high model performance that would not represent a realistic early-warning predictive-maintenance system.

---

# 49. Production Data Recommendation

For a real industrial deployment, the preferred raw data schema is:

```text
machine_id
timestamp
sensor_1
sensor_2
sensor_3
...
failure
```

Example:

```text
machine_id,timestamp,temperature,speed,torque,tool_wear,failure
M001,2026-01-01 08:00:00,301.2,1450,42.1,120,0
M001,2026-01-01 08:01:00,301.4,1452,42.8,121,0
M001,2026-01-01 08:02:00,302.1,1455,43.2,122,0
```

With this structure, temporal features can be generated correctly.

---

# 50. Real-Time Architecture

The intended future production architecture is:

```text
                 ┌─────────────────────┐
                 │ Machine / Sensors   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Event / Data Stream │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Validation     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Online Feature      │
                 │ Generation          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Predictive Model   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Failure Probability│
                 │ / Health Score     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Alert / Maintenance │
                 └─────────────────────┘
```

Task 2 provides the feature-generation layer required between raw machine telemetry and the future prediction model.

---

# 51. Validation Checklist

Before considering Task 2 complete, verify:

```text
[ ] Raw dataset exists
[ ] Raw dataset schema validated
[ ] Target column identified
[ ] AI4I schema correctly detected
[ ] Failure-mode leakage columns excluded
[ ] No artificial timestamp generated from UID
[ ] Train/validation/test split completed
[ ] Split sizes are correct
[ ] Feature transformations fit correctly
[ ] Validation transformation uses training-fitted state
[ ] Test transformation uses training-fitted state
[ ] Feature schemas match
[ ] Missing values checked
[ ] Infinite values checked
[ ] Constant features checked
[ ] Duplicate features checked
[ ] Correlations analyzed
[ ] Distributions visualized
[ ] Rolling statistics visualized where applicable
[ ] Feature importance calculated
[ ] Leakage checks completed
[ ] Engineered dataset exported
[ ] Train dataset exported
[ ] Validation dataset exported
[ ] Test dataset exported
[ ] HTML report generated
[ ] DOCX report generated
[ ] Execution summary generated
```

---

# 52. Quality Gates

A production-oriented Task 2 execution should fail or warn when:

* the input dataset does not exist
* required columns are missing
* duplicate column names exist
* target cannot be identified
* train/validation/test proportions are invalid
* generated feature schemas differ
* target leakage is detected
* infinite values remain unexpectedly
* a feature transformation generates invalid numerical values
* output artifacts cannot be written

These checks are intended to make failures explicit instead of silently producing unreliable data.

---

# 53. Testing

Run the project tests with:

```bash
pytest
```

For coverage:

```bash
pytest --cov=src
```

Code formatting:

```bash
black src scripts
```

Static analysis:

```bash
ruff check src scripts
```

---

# 54. Development Workflow

Recommended development process:

```text
1. Obtain authentic dataset
        ↓
2. Validate raw schema
        ↓
3. Explore raw variables
        ↓
4. Implement feature definitions
        ↓
5. Validate feature quality
        ↓
6. Check leakage
        ↓
7. Visualize features
        ↓
8. Export engineered datasets
        ↓
9. Generate HTML report
        ↓
10. Generate DOCX documentation
        ↓
11. Validate complete pipeline
        ↓
12. Proceed to model development
```

---

# 55. Version Control

Generated files that should normally be excluded from Git include:

```text
.venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
reports/figures/
data/processed/
```

The raw dataset may also be excluded depending on project/distribution requirements.

Refer to:

```text
.gitignore
```

for the project's complete ignore policy.

---

# 56. Security and Data Governance

For real industrial deployment:

* never commit credentials
* never commit API keys
* never commit private machine identifiers
* do not expose sensitive telemetry
* validate incoming sensor values
* authenticate data sources
* maintain audit logs
* version feature definitions
* version trained models
* monitor data drift
* monitor feature drift
* monitor prediction performance

The AI4I dataset itself is suitable for experimentation, but real industrial data may contain commercially sensitive information.

---

# 57. Limitations

This project should not be interpreted as a production predictive-maintenance system solely because it uses an end-to-end engineering pipeline.

Important limitations include:

1. AI4I is synthetic.
2. AI4I does not represent every industrial machine type.
3. Synthetic failure mechanisms may not reflect real equipment behavior.
4. Model performance on AI4I should not be interpreted as field performance.
5. Real deployment requires actual timestamped telemetry.
6. Real deployment requires machine-specific historical state.
7. Thresholds should be calibrated using domain knowledge and operational data.
8. Monitoring and retraining are required after deployment.

---

# 58. Recommended Next Task

After Task 2 has been validated, the next stage should consume:

```text
data/processed/train_features.csv
data/processed/validation_features.csv
data/processed/test_features.csv
```

The next modeling stage should preserve the same leakage-safe boundaries.

A recommended future architecture is:

```text
Task 1
Data acquisition / validation
        ↓
Task 2
Feature engineering
        ↓
Task 3
Model training
        ↓
Task 4
Model evaluation
        ↓
Task 5
Deployment
        ↓
Task 6
Real-time monitoring
```

---

# 59. Citation

The primary dataset should be cited as:

> AI4I 2020 Predictive Maintenance Dataset. (2020). UCI Machine Learning Repository. DOI: 10.24432/C5HS5C.

Dataset license:

> CC BY 4.0

Official repository:

https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdat

---

# 60. Final Task 2 Definition of Done

Task 2 is considered complete when the project can execute:

```bash
python scripts/run_task2.py
```

and successfully produce:

```text
data/processed/engineered_features.csv
data/processed/train_features.csv
data/processed/validation_features.csv
data/processed/test_features.csv
```

together with:

```text
reports/feature_engineering_report.html
reports/feature_engineering_documentation.docx
reports/task2_execution_summary.json
```

and the execution confirms:

```text
✓ Dataset loaded
✓ Schema validated
✓ Feature engineering completed
✓ Leakage checks completed
✓ Feature quality checks completed
✓ Train/validation/test transformations completed
✓ Engineered datasets exported
✓ Visualizations generated
✓ HTML report generated
✓ DOCX documentation generated
✓ Execution summary generated
```

At that point, the feature-engineering stage is ready to serve as the controlled input to the downstream predictive-maintenance modeling pipeline.
