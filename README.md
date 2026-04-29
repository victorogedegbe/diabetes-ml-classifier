# Diabetes Hospital Readmission Prediction

Predicting 30-day hospital readmission for diabetic patients on the **UCI Diabetes 130-US Hospitals** dataset (~100,000 encounters, 50 features).

A class-imbalance-focused study comparing logistic regression, random forest, and XGBoost variants — including statistical significance testing of AUC differences via DeLong's test.

---

## Problem

Readmission within 30 days of discharge is a costly outcome for hospitals (under the US Hospital Readmissions Reduction Program) and a clinically meaningful one for patients. The target class is heavily imbalanced — only **11.2%** of encounters are readmitted within 30 days — making this a non-trivial classification task where naive accuracy is misleading.

**Target:** `readmitted_bin` (1 = readmitted within 30 days, 0 = otherwise)

---

## Dataset

| | |
|---|---|
| Source | UCI ML Repository — Diabetes 130-US Hospitals (1999–2008) |
| Encounters | 101,766 → 100,244 after cleaning |
| Features | 50 raw → 45 modelling features after engineering |
| Class balance | 88.8% negative / 11.2% positive |

---

## Pipeline

**1. Cleaning & missing values**
- Replaced encoded `?` values with `NA`
- Dropped `weight` (97% missing), `payer_code` (40% missing), `examide`, `citoglipton` (zero variance), and identifiers
- Mapped `medical_specialty` and `race` missing values to `Unknown`
- Dropped rows missing all three diagnosis codes

**2. Feature engineering**
- **ICD-9 diagnosis grouping:** mapped raw ICD-9 codes (~700 unique values per diagnosis field) into 9 clinical chapters — Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other
- **Medication change count:** aggregated 21 individual diabetes drugs into a single `med_change_count` feature (count of drugs changed Up or Down during the encounter)
- **Medical specialty grouping:** collapsed ~70 specialty strings into 10 clinically-meaningful groups
- **Ordinal age encoding:** mapped age bands `[0-10), [10-20), …` to integer 1–10

**3. Exploratory data analysis**
Bivariate plots confirmed predictable structure before modelling:
- Prior inpatient visits showed a strong monotonic effect on readmission rate (8.5% at 0 visits → 39.6% at 6+ visits) — the single strongest predictor
- Discharge disposition mattered substantially (Home: 9.4% vs Other facility: 20.2%)
- Age-by-discharge interaction heatmap showed non-additive risk structure, motivating ensemble methods over linear models

**4. Train/test split**
80/20 stratified split using `caret::createDataPartition`, preserving the 11.2% positive rate in both folds.

---

## Models compared

Five model variants were compared, each addressing class imbalance differently:

| Model | Imbalance strategy |
|---|---|
| Logistic Regression (unweighted) | Threshold tuning |
| Logistic Regression (weighted) | Inverse-frequency observation weights |
| Random Forest (`classwt`) | Gini split weighting |
| Random Forest (balanced bootstrap) | Equal-class `sampsize` per tree (Chen, Liaw & Breiman 2004) |
| XGBoost | `scale_pos_weight` + early stopping on internal validation AUC |

XGBoost used a 64k/16k internal train/validation split for early stopping (97 rounds, `eta=0.05`, `max_depth=6`).

---

## Results

| Model | AUC | Precision | Recall | F1 | Imbalance Strategy |
|---|---|---|---|---|---|
| Logistic Regression (unweighted) | 0.632 | 0.557 | 0.017 | 0.034 | Threshold tuning |
| Logistic Regression (weighted) | 0.633 | 0.164 | 0.507 | 0.248 | Observation weights |
| Random Forest (classwt) | 0.586 | 0.208 | 0.017 | 0.031 | classwt (Gini only) |
| Random Forest (balanced bootstrap) | 0.658 | 0.217 | 0.328 | 0.261 | Balanced bootstrap |
| **XGBoost (scale_pos_weight)** | **0.666** | 0.178 | **0.573** | **0.272** | scale_pos_weight |

**Best by AUC: XGBoost — AUC 0.666, F1 0.272.**

This is consistent with the published academic benchmark on this dataset (Strack et al., 2014, AUC ≈ 0.65), confirming the result is realistic and well-calibrated rather than over-optimistic.

### DeLong's test — pairwise AUC significance

Statistical comparison of AUCs on aligned test rows:

| Comparison | D | p-value | Verdict |
|---|---|---|---|
| XGBoost vs LR (unweighted) | 3.91 | 9.4e-05 | XGBoost significantly better |
| XGBoost vs RF (classwt) | 9.16 | 5.5e-20 | XGBoost significantly better |
| XGBoost vs RF (balanced) | 0.89 | 0.375 | Statistically tied |
| RF (balanced) vs RF (classwt) | 10.58 | 3.9e-26 | Balanced bootstrap significantly better |
| LR (weighted) vs LR (unweighted) | 1.81 | 0.071 | No significant AUC difference |

### Key findings

- **The unweighted logistic regression's 89% accuracy is deceptive** — it predicts the positive class only 0.3% of the time. This illustrates why accuracy is the wrong metric for imbalanced problems.
- **Imbalance handling matters more than model choice for tree methods.** Random Forest's `classwt` only affects the Gini split criterion, not bootstrap sampling — switching to balanced bootstrap raised AUC from 0.586 to 0.658 (p < 1e-25).
- **For LR, the choice between weighting and threshold tuning is a precision/recall trade-off** — they yield equivalent AUC (DeLong p = 0.07).
- **XGBoost and balanced RF are statistically tied on AUC**, but XGBoost achieves substantially higher recall (0.57 vs 0.33) — the more useful operating point for a clinical screening tool.

---

## Tech stack

- **Language:** R
- **Modelling:** `glm`, `randomForest`, `xgboost`
- **Workflow:** `caret`, `dplyr`, `Matrix`
- **Evaluation:** `pROC` (ROC, AUC, DeLong's test)
- **Visualisation:** `ggplot2`, `corrgram`

---

## Repository contents

```
.
├── DS7003_Readmission_Analysis_R_SCRIPT.R   # full analysis pipeline
├── diabetes_model_comparison.csv            # final metrics table
├── eda_plots/                               # bivariate EDA figures
└── README.md
```

The dataset (`diabetic_data.csv`) is not included — download from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

## How to run

1. Clone the repo
2. Download `diabetic_data.csv` from UCI and place in the project root
3. Open the `.R` script in RStudio and run section by section, or `source()` the whole file
4. Required packages: `dplyr`, `ggplot2`, `caret`, `randomForest`, `xgboost`, `pROC`, `corrgram`, `Matrix`, `scales`

---

## Reference

Strack, B. *et al.* (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International.

---

*Project completed as part of MSc Data Science, University of East London (DS7003 Machine Learning module).*
