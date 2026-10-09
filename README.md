<h1 align="center">Satellite Anomaly Time-to-Event Prediction</h1>

<p align="center">
  <b>Machine Learning for Predicting the Time Until the Next Recorded Satellite Anomaly</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=flat-square&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Domain-Satellite%20Analytics-6A5ACD?style=flat-square">
</p>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation](#2-motivation)
3. [Problem Statement](#3-problem-statement)
4. [Research Background](#4-research-background)
5. [Project Objectives](#5-project-objectives)
6. [System Architecture](#6-system-architecture)
7. [Data Sources](#7-data-sources)
8. [Raw Data Requirements](#8-raw-data-requirements)
9. [Data Preparation Pipeline](#9-data-preparation-pipeline)
10. [Satellite Identifier Normalization](#10-satellite-identifier-normalization)
11. [GPS Satellite Mapping](#11-gps-satellite-mapping)
12. [Time-to-Event Construction](#12-time-to-event-construction)
13. [Filtering Strategy](#13-filtering-strategy)
14. [Feature Engineering](#14-feature-engineering)
15. [Final Dataset](#15-final-dataset)
16. [Exploratory Dataset Statistics](#16-exploratory-dataset-statistics)
17. [Machine Learning Pipeline](#17-machine-learning-pipeline)
18. [Model 1 - Linear Regression](#18-model-1---linear-regression)
19. [Model 2 - Support Vector Regression](#19-model-2---support-vector-regression)
20. [Model 3 - Modified Naive Bayes](#20-model-3---modified-naive-bayes)
21. [Kaplan-Meier Survival Prior](#21-kaplan-meier-survival-prior)
    - [21.1 Model 4 - Gradient Boosting Regressor](#211-model-4---gradient-boosting-regressor-l1-loss)
    - [21.2 Model 5 - Hybrid Survival-Boosting Ensemble](#212-model-5---hybrid-survival-boosting-ensemble)
22. [Evaluation Methodology](#22-evaluation-methodology)
23. [Evaluation Metrics](#23-evaluation-metrics)
24. [Experimental Results](#24-experimental-results)
25. [Understanding the Results](#25-understanding-the-results)
26. [Important Validation Limitation](#26-important-validation-limitation)
27. [Repository Structure](#27-repository-structure)
28. [Installation](#28-installation)
29. [Raw Dataset Setup](#29-raw-dataset-setup)
30. [Generating the Dataset](#30-generating-the-dataset)
31. [Dataset Validation](#31-dataset-validation)
32. [Running the Models](#32-running-the-models)
33. [Complete Reproduction](#33-complete-reproduction)
34. [Git Workflow](#34-git-workflow)
35. [Design Decisions](#35-design-decisions)
36. [Limitations](#36-limitations)
37. [Future Work](#37-future-work)
38. [Research Extensions](#38-research-extensions)
39. [Project Status](#39-project-status)
40. [Reference Work](#40-reference-work)
41. [Data Attribution](#41-data-attribution)
42. [Citation](#42-citation)
43. [Authors](#43-authors)
44. [License](#44-license)
45. [End-to-End Pipeline Summary](#45-end-to-end-pipeline-summary)
46. [Current Baseline at a Glance](#46-current-baseline-at-a-glance)
47. [Final Notes](#47-final-notes)

---

## 1. Project Overview

Satellite anomaly prediction is an important problem in spacecraft operations and satellite reliability.

Satellites operate in a complex environment influenced by orbital conditions, solar activity, radiation, thermal effects, and other environmental and operational factors.

A conventional anomaly prediction problem may be formulated as:

> **Will an anomaly occur?**

This project investigates a different formulation:

> **How much time will pass before the next recorded anomaly occurs?**

The problem is therefore formulated as a **Time-to-Event (TTE) prediction problem**.

The project combines historical satellite anomaly records with:

- Orbital characteristics
- Solar activity
- Temporal information

and evaluates several machine learning approaches for estimating the number of days until the next recorded anomaly.

The current implementation evaluates five approaches:

1. **Linear Regression** (Stanford Baseline)
2. **Support Vector Regression (SVR)** (Stanford Baseline)
3. **Modified Naive Bayes with a Kaplan-Meier survival prior** (Stanford Baseline)
4. **Gradient Boosting Regressor (L1 Loss)** (Proposed Extension)
5. **Hybrid Survival-Boosting Ensemble** (Proposed Extension)

---

## 2. Motivation

A binary anomaly classifier only provides an event indicator. For example:

```text
Anomaly: YES
```

This does not provide information about *when* the anomaly may occur.

A time-to-event model instead produces a value such as:

```text
Predicted TTE = 15 days
```

which provides temporal information.

The overall goal is therefore:

```text
Satellite + Environment + Time
              |
              v
      Machine Learning Model
              |
              v
     Predicted Time-to-Event
```

The project also explores whether concepts from survival analysis can be incorporated into a machine learning formulation for satellite anomaly prediction.

---

## 3. Problem Statement

For a satellite with consecutive anomaly dates:

```text
t₁, t₂, t₃, ..., tₙ
```

the time-to-event for anomaly *i* is defined as:

```text
TTEᵢ = tᵢ₊₁ - tᵢ
```

where:

- `tᵢ` is the current anomaly date
- `tᵢ₊₁` is the next anomaly date
- `TTEᵢ` is measured in days

The machine learning task is therefore:

```text
Input:
    Perigee
    Inclination
    Sunspot Number
    Month

Output:
    Time until next anomaly (days)
```

Mathematically:

$$X = [\text{Perigee}, \text{Inclination}, \text{SSN}, \text{Month}]$$

and the learning problem is:

$$f(X) \rightarrow \text{TTE}$$

---

## 4. Research Background

This project is based on the methodology presented in:

> Christopher Naughton, *"Satellite Anomaly Prediction using Survival Analysis and Machine Learning"*

The original project investigated satellite anomaly prediction using:

- Linear Regression
- Support Vector Regression
- Modified Naive Bayes
- Survival-analysis concepts

The original feature set contained:

- Starting Month
- Sunspot Number
- X-ray Flux
- Mass
- Perigee
- Inclination

The current project adapts the methodology to a smaller four-feature dataset because all original feature sources were not available under the constraints of this implementation.

The current feature set is:

- Perigee
- Inclination
- Sunspot Number
- Month

---

## 5. Project Objectives

### Primary Objective

Develop a reproducible machine learning pipeline for satellite anomaly time-to-event prediction.

### Secondary Objectives

1. Integrate multiple satellite and environmental datasets.
2. Normalize satellite identifiers across different sources.
3. Construct a time-to-event target from historical anomaly records.
4. Integrate orbital characteristics.
5. Integrate daily solar activity.
6. Compare multiple prediction approaches.
7. Evaluate models using error metrics measured in days.
8. Explore the relationship between survival-analysis concepts and machine learning.
9. Establish a reproducible experimental workflow.
10. Investigate model generalization across satellites.

---

## 6. System Architecture

The overall system is organized into five major stages:

```text
                 ┌─────────────────────────┐
                 │ NOAA Satellite Anomaly  │
                 │ Database                │
                 └────────────┬────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Data Cleaning   │
                    │ & Normalization │
                    └────────┬────────┘
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
      ┌────────────┐  ┌────────────┐  ┌────────────┐
      │ CelesTrak  │  │   SILSO    │  │ Temporal   │
      │ SATCAT     │  │ Sunspots   │  │ Features   │
      └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Feature Integration  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ TTE Construction     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Final ML Dataset     │
                 │                      │
                 │ 890 events           │
                 │ 11 satellites        │
                 │ 4 features           │
                 └──────────┬───────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
      ┌─────────────┐ ┌────────────┐ ┌──────────────┐
      │   Linear    │ │    SVR     │ │  Modified    │
      │ Regression  │ │            │ │ Naive Bayes  │
      └──────┬──────┘ └─────┬──────┘ └──────┬───────┘
             │              │               │
             └──────────────┼───────────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Model Evaluation   │
                  │ MAE / RMSE / MRE   │
                  └────────────────────┘
```

---

## 7. Data Sources

The project integrates three primary datasets.

### 7.1 NOAA Satellite Anomaly Database

The NOAA anomaly database contains historical anomaly records for satellites.

Expected file:

```text
raw/NOAA_Info/anom5j.xls
```

Relevant fields include `BIRD`, `ADATE`, `ADIAG`, `ACOMMENT`, `SVE`, and `SPIN`.

The current pipeline primarily uses `BIRD` and `ADATE` for anomaly timeline construction.

### 7.2 CelesTrak SATCAT

CelesTrak SATCAT provides satellite catalog information and orbital characteristics.

Expected file:

```text
raw/satcat.csv
```

The project uses `OBJECT_NAME`, `PERIGEE`, and `INCLINATION`, which are converted into:

- `Name`
- `Perigee (km)`
- `Inclination (deg)`

The orbital features are satellite-level attributes.

### 7.3 SILSO Sunspot Data

Daily sunspot observations are obtained from SILSO.

Expected file:

```text
raw/sunspot.csv
```

The project uses the daily sunspot number associated with the anomaly start date.

The relevant source columns are: Year, Month, Day, Decimal Year, Sunspot Number.

---

## 8. Raw Data Requirements

Before running the preprocessing pipeline, the following structure must exist:

```text
satellite-anomaly-survival-prediction/
│
└── raw/
    │
    ├── satcat.csv
    │
    ├── sunspot.csv
    │
    └── NOAA_Info/
        └── anom5j.xls
```

The raw datasets must be obtained separately from their respective providers. The repository does not generate these source files.

---

## 9. Data Preparation Pipeline

The complete preprocessing pipeline is implemented in `src/prepare_data.py`.

The script performs the following operations:

1. Load NOAA anomaly database
2. Normalize satellite identifiers
3. Parse anomaly dates
4. Load CelesTrak SATCAT
5. Normalize CelesTrak identifiers
6. Load SILSO daily sunspot data
7. Identify satellites with multiple anomalies
8. Match satellite records
9. Retain required GPS anomaly records
10. Remove duplicate anomaly records
11. Calculate TTE
12. Filter TTE
13. Filter satellites by event count
14. Merge sunspot data
15. Extract month
16. Construct satellite feature table
17. Write final datasets

---

## 10. Satellite Identifier Normalization

Different datasets may represent the same satellite using different naming conventions.

The preprocessing pipeline normalizes satellite identifiers by:

- converting to lowercase
- replacing `-` with spaces
- removing leading/trailing whitespace

For example, `GPS-5113` becomes `gps 5113`.

This improves consistency during dataset integration.

---

## 11. GPS Satellite Mapping

A major data-integration issue was identified for GPS satellites.

The NOAA anomaly database uses identifiers such as `gps 5113`, while CelesTrak uses names such as `NAVSTAR 3 (OPS 5113)`.

The preprocessing pipeline explicitly maps these identifiers:

| NOAA Identifier | CelesTrak Identifier |
|-----------------|----------------------|
| gps 5111 | NAVSTAR 1 (OPS 5111) |
| gps 5112 | NAVSTAR 2 (OPS 5112) |
| gps 5113 | NAVSTAR 3 (OPS 5113) |
| gps 5114 | NAVSTAR 4 (OPS 5114) |
| gps 5118 | NAVSTAR 6 (OPS 5118) |
| gps 9794 | NAVSTAR 8 (OPS 9794) |

This mapping ensures that orbital characteristics are available for the corresponding GPS satellites.

---

## 12. Time-to-Event Construction

The anomaly records for each satellite are sorted chronologically.

Consider Satellite A with anomalies on:

```text
01-Jan-2000
10-Jan-2000
25-Jan-2000
05-Feb-2000
```

The generated TTE values are:

```text
01-Jan → 10-Jan =  9 days
10-Jan → 25-Jan = 15 days
25-Jan → 05-Feb = 11 days
```

The last anomaly does not produce a TTE observation because there is no subsequent anomaly. Therefore, for each satellite:

```text
Number of TTE observations = Number of anomalies - 1
```

before additional filtering.

---

## 13. Filtering Strategy

The pipeline applies the following filters.

**Positive TTE** — only intervals satisfying `TTE > 0` are retained.

**Maximum TTE** — only intervals satisfying `TTE < 365` are retained. This removes intervals of one year or longer.

**Minimum number of events** — only satellites with more than 20 valid TTE observations are retained. This helps avoid training on satellites with extremely sparse anomaly histories.

**Explicit satellite exclusions** — the pipeline excludes `scatha` and `ecs 1` from the final dataset.

---

## 14. Feature Engineering

The final model uses four features.

| Feature | Column | Description | Unit / Range |
|---------|--------|-------------|--------------|
| Perigee | `perigee` | Satellite perigee altitude | kilometers |
| Inclination | `inclination` | Orbital inclination | degrees |
| Sunspot Number | `ssn` | Daily sunspot number associated with the anomaly interval start date | — |
| Month | `month` | Calendar month corresponding to the anomaly interval start date | 1–12 |

---

## 15. Final Dataset

After all preprocessing steps, the current dataset contains:

- **890** events
- **11** satellites
- **0** missing orbital-feature mappings

### `data.csv`

The event-level dataset contains `BIRD`, `TTE`, `T0`, `SSN`, and `Month`, where:

- `BIRD` = satellite identifier
- `TTE` = time until next anomaly in days
- `T0` = current anomaly date
- `SSN` = sunspot number
- `Month` = month of current anomaly

### `satfeat.csv`

The satellite-level dataset contains `Name`, `Perigee (km)`, and `Inclination (deg)`.

The model preprocessing joins `data.csv` with `satfeat.csv` using the satellite identifier.

---

## 16. Exploratory Dataset Statistics

Current TTE statistics:

| Statistic | Value |
|-----------|-------|
| Number of events | 890 |
| Number of satellites | 11 |
| Mean TTE | ~20.10 days |
| Standard deviation | ~39.45 days |
| Minimum TTE | 1 day |
| Maximum TTE | 346 days |

The target distribution is strongly concentrated toward smaller TTE values. The frequency of the first few TTE values is approximately:

| TTE (days) | Count |
|-----------:|------:|
| 1 | 124 |
| 2 | 88 |
| 3 | 63 |
| 4 | 57 |
| 5 | 54 |
| 6 | 45 |
| 7 | 39 |
| 8 | 31 |
| 9 | 22 |
| 10 | 23 |

This skew is important when interpreting percentage-based metrics.

---

## 17. Machine Learning Pipeline

All models use the same input features: `perigee`, `inclination`, `ssn`, `month`.

The general pipeline is:

```text
data.csv
    +
satfeat.csv
    |
    v
Feature Merge
    |
    v
Feature Selection
    |
    v
StandardScaler
    |
    +-------------------+
    |                   |
    v                   v
Model Training      Cross Validation
    |                   |
    +---------+---------+
              |
              v
        TTE Predictions
              |
              v
       Evaluation Metrics
```

---

## 18. Model 1 - Linear Regression

Implementation: `src/linear_regression.py`

The baseline model is `LinearRegression()`, with inputs standardized using `StandardScaler()`.

```text
Four Features
      |
      v
StandardScaler
      |
      v
LinearRegression
      |
      v
Predicted TTE
```

Linear Regression provides a simple baseline for determining whether a linear relationship exists between the selected features and TTE.

---

## 19. Model 2 - Support Vector Regression

Implementation: `src/svr.py`

The current configuration is:

```python
SVR(
    kernel="rbf",
    C=10.0,
    epsilon=0.1,
    gamma="auto"
)
```

The RBF kernel allows nonlinear relationships to be modeled.

```text
Four Features
      |
      v
StandardScaler
      |
      v
RBF SVR
      |
      v
Predicted TTE
```

---

## 20. Model 3 - Modified Naive Bayes

Implementation: `src/naive_bayes.py`

The Modified Naive Bayes approach combines:

- Gaussian feature likelihoods
- Event likelihood
- Survival likelihood
- Kaplan-Meier survival prior
- Posterior probability estimation

The conceptual pipeline is:

```text
Training TTE
     |
     v
Kaplan-Meier Prior
     |
     +
Feature Distributions
     |
     v
Posterior Probability
     |
     v
Predicted TTE
```

The approach adapts the standard Naive Bayes idea to a time-to-event setting.

---

## 21. Kaplan-Meier Survival Prior

The Kaplan-Meier estimator models the probability that the event has not yet occurred by time *t*. The survival function is:

$$S(t) = \prod_{t_i \leq t} \left( 1 - \frac{d_i}{n_i} \right)$$

where:

- $t_i$ is an observed event time
- $d_i$ is the number of events occurring at that time
- $n_i$ is the number of observations at risk

The resulting survival probability is incorporated into the modified Naive Bayes prediction process.

```text
TTE History
    |
    v
Kaplan-Meier Estimation
    |
    v
Survival Prior
    |
    +
Feature Likelihoods
    |
    v
Posterior Event Probability
    |
    v
Predicted Time
```

---

### 21.1 Model 4 - Gradient Boosting Regressor (L1 Loss)

Implementation: `src/gradient_boosting.py`

Gradient Boosting constructs an additive ensemble of shallow regression decision trees ($M=100$ estimators, learning rate $\eta=0.05$, max depth $=3$). Critically, it utilizes the **L1 (absolute error) loss function**:

$$\mathcal{L}(y, \hat{y}) = |y - \hat{y}|$$

Unlike standard Mean Squared Error loss which penalizes outliers quadratically, L1 loss trains trees to predict conditional medians, directly minimizing daily prediction error and providing strong robustness against extreme long-interval anomalies.

```text
Four Features
      │
      ▼
Gradient Boosting (100 Trees, L1 Loss)
      │
      ▼
Predicted TTE (Median-Optimal)
```

---

### 21.2 Model 5 - Hybrid Survival-Boosting Ensemble

Implementation: `src/hybrid_ensemble.py`

The Hybrid Ensemble combines non-parametric statistical survival analysis with gradient-boosted decision trees using a weighted convex combination:

$$\hat{y}_{\text{hybrid}} = 0.70 \cdot \hat{y}_{\text{GBR}} + 0.30 \cdot \hat{y}_{\text{NB-KM}}$$

This architecture leverages the high discriminative power of Gradient Boosting along with the global empirical risk distribution encoded by the Kaplan-Meier survival estimator, yielding the lowest Mean Relative Error (**165.0%**) and balanced day error (**MAE = 16.77 days**).

```text
               Four Input Features
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
Gradient Boosting (L1)       Modified Naive Bayes (KM)
   [Weight: 70%]                 [Weight: 30%]
         │                             │
         └──────────────┬──────────────┘
                        ▼
             Hybrid Ensembled TTE
```

---

## 22. Evaluation Methodology

All five models use 10-fold cross-validation:

```python
KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)
```

This ensures that the same cross-validation configuration is used for all three models.

For each fold:

```text
Training Fold
     |
     v
Fit StandardScaler
     |
     v
Transform Training Data
     |
     v
Train Model
     |
     v
Transform Test Data
     |
     v
Generate Predictions
     |
     v
Calculate Metrics
```

The test predictions from all folds are aggregated for the final metrics.

---

## 23. Evaluation Metrics

The project reports three primary metrics.

### Mean Relative Error (MRE)

$$MRE = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100$$

This measures the average error relative to the actual TTE. However, it is highly sensitive to small target values.

### Mean Absolute Error (MAE)

$$MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

MAE is expressed in days. For example, `MAE = 17 days` means the average absolute prediction error is approximately 17 days.

### Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

RMSE penalizes larger errors more strongly than MAE.

---

## 24. Experimental Results

All models were evaluated on 890 events, 11 satellites, 4 features, using 10-fold cross-validation with `random_state = 42`.

| Model | Train MRE | Test MRE | Test MAE | Test RMSE |
|-------|----------:|---------:|---------:|----------:|
| Linear Regression | 490.1% | 493.5% | 21.50 days | 39.05 days |
| SVR | 154.7% | 167.8% | 17.22 days | 41.34 days |
| Modified Naive Bayes | 160.1% | 166.2% | 17.19 days | 41.37 days |
| **Gradient Boosting (L1)** | **140.6%** | **166.5%** | **16.73 days** | **40.34 days** |
| **Hybrid Survival Ensemble** | **145.4%** | **165.0%** | **16.77 days** | **40.58 days** |

These are the current baseline experimental results.

---

## 25. Understanding the Results

### Comparing the models

The models exhibit different error characteristics across 10-fold cross-validation:

- Linear Regression: Test MAE = 21.50 days
- SVR: Test MAE = 17.22 days
- Modified Naive Bayes: Test MAE = 17.19 days
- **Gradient Boosting (L1)**: **Test MAE = 16.73 days** (Lowest absolute error)
- **Hybrid Survival Ensemble**: **Test MAE = 16.77 days** (Best balance, lowest Test MRE = 165.0%)

The RMSE values are:

- Linear Regression: 39.05 days
- SVR: 41.34 days
- Modified Naive Bayes: 41.37 days
- **Gradient Boosting (L1)**: **40.34 days**
- **Hybrid Survival Ensemble**: **40.58 days**

The difference between MAE and RMSE illustrates that the models may produce some relatively large errors even when their average absolute errors are smaller. The results should therefore be interpreted using multiple metrics.

### Why is Mean Relative Error so large?

The dataset contains many very short TTE values. Consider:

```text
Actual TTE    = 1 day
Predicted TTE = 5 days

Absolute error: 4 days
Relative error: 400%
```

Similarly:

```text
Actual TTE    = 2 days
Predicted TTE = 10 days

Absolute error: 8 days
Relative error: 400%
```

A high relative-error percentage therefore does not necessarily correspond to an equally large error in terms of days. For this dataset, MAE and RMSE provide important additional context.

---

## 26. Important Validation Limitation

The current experiments use random event-level K-fold cross-validation.

This creates a potential generalization issue because multiple observations belong to the same satellite. For example:

```text
Satellite A

Event 1 → Training
Event 2 → Training
Event 3 → Test
Event 4 → Training
```

The model can therefore encounter the same satellite in both training and testing. This is particularly relevant because **perigee** and **inclination** are satellite-level features and remain constant for the same satellite.

Random K-fold validation should therefore be treated as the current baseline rather than the final test of cross-satellite generalization.

### Planned group-aware validation

A future evaluation stage will use satellite-level grouping, with `BIRD` as the grouping variable. A `GroupKFold` strategy can ensure that all observations from one satellite remain in the same fold.

Group-aware folds (planned):

```text
Fold 1
 ├── Satellite A
 ├── Satellite B
 └── Satellite C

Fold 2
 ├── Satellite D
 ├── Satellite E
 └── Satellite F
```

Instead of the current event-level folds:

```text
Fold 1
 ├── Satellite A Event 1
 ├── Satellite B Event 2

Fold 2
 ├── Satellite A Event 3
 ├── Satellite B Event 4
```

This provides a stricter evaluation of whether learned relationships generalize across spacecraft.

---

## 27. Repository Structure

```text
satellite-anomaly-survival-prediction/
│
├── README.md                              # Comprehensive documentation and project guide
├── requirements.txt                       # Project Python dependencies
├── .gitignore                             # Git ignore specification
├── run_pipeline.py                        # Master pipeline: trains all 5 models, runs 10-fold CV, exports outputs
│
├── data.csv                               # Cleaned event-level dataset (890 events across 11 satellites)
├── satfeat.csv                            # Satellite orbital features (Perigee, Inclination)
│
├── raw/                                   # Source datasets
│   ├── satcat.csv                         # CelesTrak satellite catalog
│   ├── sunspot.csv                        # SILSO daily sunspot numbers
│   └── NOAA_Info/
│       └── anom5j.xls                     # NOAA satellite anomaly database
│
├── data/                                  # Canonical copy of prepared datasets
│   ├── data.csv
│   └── satfeat.csv
│
├── src/                                   # Source code implementations
│   ├── __init__.py
│   ├── prepare_data.py                    # Data preparation, normalization & TTE construction
│   ├── linear_regression.py               # Linear regression baseline
│   ├── svr.py                             # Support Vector Regression baseline (RBF kernel)
│   ├── naive_bayes.py                     # Modified Naive Bayes with Kaplan-Meier survival prior
│   ├── gradient_boosting.py               # Gradient Boosting Regressor (L1 absolute error loss)
│   └── hybrid_ensemble.py                 # Hybrid Survival-Boosting Ensemble (GBR + KM Naive Bayes)
│
├── models/                                # Serialized trained models & scalers (.joblib)
│   ├── linear_regression.joblib
│   ├── svr.joblib
│   ├── modified_naive_bayes.joblib
│   ├── gradient_boosting.joblib
│   ├── hybrid_ensemble.joblib
│   └── scaler.joblib
│
├── results/                               # Benchmark metrics and publication-quality plots
│   ├── metrics_summary.csv                # 10-fold CV performance across all 5 models
│   ├── predictions_comparison.csv         # Predicted vs actual TTE comparison table
│   ├── model_comparison.png               # MAE, RMSE, and MRE comparison bar charts
│   ├── predictions_scatter.png            # Actual vs Predicted scatter plots (Fig 2 reproduction)
│   ├── feature_importance.png             # Gradient Boosting feature importances
│   ├── kaplan_meier_survival_curve.png    # Non-parametric survival probability curve
│   ├── tte_distribution.png               # Target variable distribution & histogram
│   └── feature_correlations.png           # Feature correlation heatmap
│
├── notebooks/                             # Interactive presentation & demonstration
│   └── satellite_anomaly_prediction.ipynb # Fully pre-rendered Jupyter notebook
│
└── docs/                                  # Literature and reference documentation
    ├── research_paper.pdf                 # Stanford CS229 reference paper (Christopher Naughton)
    └── project_poster.pdf                 # Stanford CS229 reference poster
```

| Path | Purpose |
|------|---------|
| `raw/` | External raw datasets (NOAA anomaly records, CelesTrak SATCAT, SILSO sunspots) |
| `data.csv` | Prepared event-level dataset (890 events, 11 satellites) |
| `satfeat.csv` | Satellite-level orbital features (Perigee, Inclination) |
| `src/` | Modular data preparation, baselines, and advanced model implementations |
| `run_pipeline.py` | Master orchestrator running 10-fold CV, model serialization, and plot export |
| `models/` | Serialized scikit-learn models and feature scaler (`.joblib`) |
| `results/` | CSV summaries and high-resolution comparison plots |
| `notebooks/` | Interactive Jupyter presentation notebook |
| `docs/` | Reference Stanford research paper and project poster |
| `README.md` | Complete project documentation and guide |
| `requirements.txt` | Python dependencies |

---

## 28. Installation

### Prerequisites

Install Python 3.x and Git, then verify:

```bash
python --version
git --version
```

### Clone the repository

```bash
git clone https://github.com/rohanth3/satellite-anomaly-survival-prediction.git
cd satellite-anomaly-survival-prediction
```

### Create a virtual environment (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
```

After activation, the terminal should display `(.venv)`.

### Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements.txt
```

The main dependencies include:

- pandas
- numpy
- scikit-learn
- matplotlib
- openpyxl
- xlrd
- python-dateutil

---

## 29. Raw Dataset Setup

Place the raw data in `raw/` with the following structure:

```text
raw/
│
├── satcat.csv
├── sunspot.csv
│
└── NOAA_Info/
    └── anom5j.xls
```

---

## 30. Generating the Dataset

Run:

```bash
python src\prepare_data.py
```

Expected current output:

```text
[prepare] building dataset...
[prepare] wrote data.csv (890 events)
[prepare] wrote satfeat.csv (11 satellites)
```

---

## 31. Dataset Validation

Run:

```bash
python -c "import pandas as pd; d=pd.read_csv('data.csv'); s=pd.read_csv('satfeat.csv'); print('Events:',len(d)); print('Satellites in data:',d['BIRD'].nunique()); print('Satellites in features:',s['Name'].nunique()); print('Missing orbital features:',sorted(set(d['BIRD'])-set(s['Name'])))"
```

Expected:

```text
Events: 890
Satellites in data: 11
Satellites in features: 11
Missing orbital features: []
```

---

## 32. Running the Models

### 32.1 Baseline Models

**Linear Regression**

```bash
python src\linear_regression.py
```

**Support Vector Regression**

```bash
python src\svr.py
```

**Modified Naive Bayes**

```bash
python src\naive_bayes.py
```

### 32.2 Proposed Modern Extensions

**Gradient Boosting Regressor (L1 Loss)**

```bash
python src\gradient_boosting.py
```

**Hybrid Survival-Boosting Ensemble**

```bash
python src\hybrid_ensemble.py
```

### 32.3 Master Pipeline Runner

To run 10-fold cross-validation on all 5 models simultaneously, export serialized models to `models/`, and generate all benchmark plots in `results/`:

```bash
python run_pipeline.py
```

### 32.4 Interactive Jupyter Notebook

To launch the complete visual presentation notebook:

```bash
jupyter notebook notebooks\satellite_anomaly_prediction.ipynb
```

---

## 33. Complete Reproduction

A complete local reproduction can be performed using:

```bash
git clone https://github.com/rohanth3/satellite-anomaly-survival-prediction.git

cd satellite-anomaly-survival-prediction

python -m venv .venv

.venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

# Prepare datasets from raw data
python src\prepare_data.py

# Run complete pipeline (trains all 5 models, runs 10-fold CV, exports models & figures)
python run_pipeline.py
```

> Remember to place the raw datasets in `raw/` (see [Raw Dataset Setup](#29-raw-dataset-setup)) before running `prepare_data.py`.

---

## 34. Git Workflow

Check repository status:

```bash
git status
```

Add files:

```bash
git add .
```

Commit:

```bash
git commit -m "Update satellite anomaly prediction project"
```

Push:

```bash
git push origin main
```

---

## 35. Design Decisions

**Why four features instead of six?**
The original reference implementation used six features. The current implementation intentionally uses only features that could be reliably constructed from the available datasets: Perigee, Inclination, Sunspot Number, and Month.

**Why remove Mass?**
The CelesTrak SATCAT data used by this project does not provide the spacecraft mass required by the original pipeline. Rather than estimate or fabricate mass values, the feature was removed.

**Why remove X-Ray Flux?**
The historical GOES X-ray data required for the original implementation was not reliably integrated into the current data pipeline. X-ray flux was therefore excluded from the final feature set.

**Why use daily sunspot number?**
Daily sunspot number provides a historical solar-activity measurement that can be aligned directly with the anomaly start date. This makes it possible to construct the solar feature without introducing temporal aggregation assumptions.

**Why use TTE instead of binary anomaly classification?**
The objective of the project is to investigate temporal prediction. The target therefore represents *time until next recorded anomaly* rather than *anomaly / no anomaly*.

---

## 36. Limitations

### 36.1 Small satellite population

The final dataset contains only 11 satellites but 890 event intervals. The number of independent satellite entities is therefore much smaller than the number of observations.

### 36.2 Historical reporting bias

The anomaly database represents recorded events. It should not automatically be interpreted as a complete record of every physical anomaly experienced by every satellite.

### 36.3 Limited feature set

The current implementation does not include Mass or X-ray flux from the original formulation.

### 36.4 TTE distribution

The target distribution is highly skewed toward short intervals. This makes relative-error metrics difficult to interpret.

### 36.5 Satellite-level leakage risk

Random K-fold validation can place different observations from the same satellite into training and testing folds. Group-aware validation is therefore required for stronger claims about generalization.

### 36.6 Orbital features are static

Perigee and inclination are treated as fixed satellite-level characteristics. The project does not currently model their temporal evolution.

---

## 37. Future Work

The planned development roadmap is:

```text
Current Baseline
      |
      v
Exploratory Data Analysis
      |
      v
Group-Aware Validation
      |
      v
Residual Analysis
      |
      v
Feature Engineering
      |
      v
Additional ML Models
      |
      v
Hyperparameter Optimization
      |
      v
Survival Analysis Extensions
      |
      v
Final Evaluation
```

### Future EDA

- TTE distribution
- TTE histogram
- TTE boxplot
- Satellite-wise TTE distribution
- Event count per satellite
- SSN vs TTE
- Month vs TTE
- Perigee vs TTE
- Inclination vs TTE
- Outlier analysis

### Future models

- Random Forest Regression
- Gradient Boosting Regression
- XGBoost
- Random Survival Forest
- Gradient Boosted Survival Models
- Cox-based survival models
- Other time-to-event approaches

### Hyperparameter optimization

| Model | Candidate parameters |
|-------|----------------------|
| SVR | `C`, `epsilon`, `gamma`, `kernel` |
| Tree-based models | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `learning_rate` |

---

## 38. Research Extensions

The project can be extended beyond point TTE prediction. Instead of predicting only:

```text
Expected TTE = 20 days
```

a survival model could estimate:

```text
P(TTE <= 7 days)
P(TTE <= 30 days)
P(TTE <= 90 days)
```

or:

```text
S(7)
S(30)
S(90)
```

where *S(t)* is the probability that an anomaly has not yet occurred by time *t*. This would provide a richer representation of uncertainty.

### Potential experimental questions

1. How well can historical event-level information predict the next anomaly interval?
2. Does solar activity improve TTE prediction?
3. Do orbital characteristics provide useful predictive information?
4. How does nonlinear SVR compare with a linear baseline?
5. Does incorporating a survival prior improve probabilistic TTE prediction?
6. How well do models generalize to satellites not present in training?
7. Does a survival formulation provide better uncertainty information than direct regression?

---

## 39. Project Status

### Data Pipeline

- [x] Repository created
- [x] Virtual environment configured
- [x] Dependency management
- [x] NOAA anomaly data integration
- [x] CelesTrak integration
- [x] SILSO integration
- [x] Satellite identifier normalization
- [x] GPS satellite mapping
- [x] TTE construction
- [x] TTE filtering
- [x] Sunspot integration
- [x] Orbital feature integration
- [x] Dataset validation

### Machine Learning

- [x] Linear Regression (Baseline)
- [x] Support Vector Regression (Baseline)
- [x] Modified Naive Bayes with Kaplan-Meier Prior (Baseline)
- [x] Gradient Boosting Regressor with L1 Loss (Proposed Extension)
- [x] Hybrid Survival-Boosting Ensemble (Proposed Extension)
- [x] Feature standardization
- [x] 10-fold cross-validation
- [x] MAE, MedAE, RMSE, and Mean Relative Error evaluation

### Analysis & Artifacts

- [x] Full EDA and Target Distribution Analysis (`results/tte_distribution.png`)
- [x] Feature Correlation Analysis (`results/feature_correlations.png`)
- [x] Kaplan-Meier Survival Analysis (`results/kaplan_meier_survival_curve.png`)
- [x] Feature Importance Analysis (`results/feature_importance.png`)
- [x] 5-Model Benchmark Comparison Plots (`results/model_comparison.png`)
- [x] Predicted vs Actual Scatter Analysis (`results/predictions_scatter.png`)
- [x] Model Serialization (`models/*.joblib`)
- [x] Interactive Presentation Notebook (`notebooks/satellite_anomaly_prediction.ipynb`)
- [x] Master End-to-End Pipeline (`run_pipeline.py`)

---

## 40. Reference Work

The primary methodological reference for this implementation is:

> Christopher Naughton, *Satellite Anomaly Prediction using Survival Analysis and Machine Learning*.

The original work investigated satellite anomaly prediction using machine learning and survival analysis.

Original repository: <https://github.com/cwnaught/CS229-Final-Project>

This repository should be considered an adaptation rather than an exact reproduction, because the current implementation uses a different final feature set and dataset construction process.

---

## 41. Data Attribution

The project uses data derived from external scientific data providers:

- NOAA
- CelesTrak
- SILSO

Users should consult the respective providers' current terms, attribution requirements, and redistribution policies before redistributing source datasets. The repository's prepared datasets are derived from these external sources.

---

## 42. Citation

If this work or methodology is referenced, please cite the foundational research paper:

> Naughton, C. *Satellite Anomaly Prediction using Survival Analysis and Machine Learning.* CS229 Final Project, Stanford University.

---

## 43. Authors

**Rohanth Ganesha C**  
SRN: `PES1UG24AM227`  
Department of Computer Science and Engineering  
Specialization: Artificial Intelligence & Machine Learning  

**Poorvi Rangaswamy**  
SRN: `PES1UG24AM193`  
Department of Computer Science and Engineering  
Specialization: Artificial Intelligence & Machine Learning  

- GitHub: <https://github.com/rohanth3>
- Project Repository: <https://github.com/rohanth3/satellite-anomaly-survival-prediction>

---

## 44. License

This project is intended for educational and research purposes.

The licensing and redistribution requirements of externally sourced datasets remain subject to their respective data providers.

---

## 45. End-to-End Pipeline Summary

```text
                    RAW DATA SOURCES
                           |
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
        NOAA          CelesTrak           SILSO
      Anomalies       Orbital Data       Sunspots
          │                │                │
          └────────────────┼────────────────┘
                           |
                           ▼
                DATA NORMALIZATION
                           |
                           ▼
                 SATELLITE MATCHING
                           |
                           ▼
                  GPS NAME MAPPING
                           |
                           ▼
                CHRONOLOGICAL EVENTS
                           |
                           ▼
                TTE CALCULATION
                           |
                           ▼
                  TTE FILTERING
                           |
                           ▼
                FEATURE ENGINEERING
                           |
                           ▼
                ┌──────────────────┐
                │ FINAL DATASET    │
                │                  │
                │ 890 Events       │
                │ 11 Satellites    │
                │ 4 Features       │
                └────────┬─────────┘
                         |
              ┌───────────┬───────────┬───────────┬───────────┐
              │           │           │           │           │
              ▼           ▼           ▼           ▼           ▼
           Linear        SVR       Modified    Gradient    Hybrid
         Regression              Naive Bayes   Boosting   Ensemble
              │           │           │           │           │
              └───────────┴───────────┼───────────┴───────────┘
                         |
                         ▼
                 10-FOLD CV
                         |
                         ▼
               MODEL PREDICTIONS
                         |
                         ▼
              ┌────────────────────┐
              │ EVALUATION         │
              │                    │
              │ MAE                │
              │ RMSE               │
              │ Relative Error     │
              └─────────┬──────────┘
                        |
                        ▼
                 FUTURE WORK
                        |
             ┌──────────┼──────────┐
             │          │          │
             ▼          ▼          ▼
           EDA      GroupKFold   More Models
```

---

## 46. Current Baseline at a Glance

```text
Project:
Satellite Anomaly Time-to-Event Prediction

Dataset:
890 events
11 satellites

Features:
Perigee
Inclination
Sunspot Number
Month

Target:
Time to next recorded anomaly

Models:
Linear Regression (Baseline)
Support Vector Regression (Baseline)
Modified Naive Bayes (Baseline)
Gradient Boosting Regressor (Proposed Extension)
Hybrid Survival Ensemble (Proposed Extension)

Validation:
10-fold shuffled K-fold CV
random_state = 42

Current Results (10-Fold CV):

Linear Regression
    Test MRE : 493.5%
    Test MAE : 21.50 days
    Test RMSE: 39.05 days

SVR
    Test MRE : 167.8%
    Test MAE : 17.22 days
    Test RMSE: 41.34 days

Modified Naive Bayes
    Test MRE : 166.2%
    Test MAE : 17.19 days
    Test RMSE: 41.37 days

Gradient Boosting (L1 Loss)
    Test MRE : 166.5%
    Test MAE : 16.73 days (Lowest MAE)
    Test RMSE: 40.34 days

Hybrid Survival Ensemble
    Test MRE : 165.0% (Lowest MRE)
    Test MAE : 16.77 days
    Test RMSE: 40.58 days

Next Experimental Stage:
EDA
→ Group-aware validation
→ Residual analysis
→ Feature analysis
→ Model refinement
→ Survival-analysis extensions
```

---

## 47. Final Notes

This repository represents an evolving research and engineering project.

The current results should be considered a baseline experiment rather than a definitive evaluation of satellite anomaly predictability.

The next major methodological step is to move from event-level random cross-validation toward satellite-level grouped validation, and to analyze the distribution of prediction errors across individual satellites and TTE ranges.

The long-term goal is to develop a reproducible framework for investigating satellite anomaly prediction through the combination of:

```text
Satellite Orbital Characteristics
              +
Solar Activity
              +
Temporal Information
              +
Machine Learning
              +
Survival Analysis
              =
Time-to-Event Prediction
```
