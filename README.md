#  SME Credit Risk Assessment Using Machine Learning

A comprehensive machine learning pipeline for predicting **credit risk** in Small and Medium Enterprises (SMEs) within supply chain finance. This project generates a realistic synthetic dataset, performs exploratory data analysis (EDA), engineers discriminative features, trains and compares multiple ML models, and provides explainable predictions using SHAP values.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

##  Overview

Credit risk assessment is a critical process in supply chain finance. This project develops a machine learning solution to classify SMEs as **High Risk (1)** or **Low Risk (0)** based on financial and operational features. The pipeline includes:

1. **Synthetic Data Generation** with realistic, non-linear risk heuristics
2. **Exploratory Data Analysis (EDA)** with professional-grade visualizations
3. **Feature Engineering & Selection** using variance thresholding
4. **Model Training & Comparison** across multiple algorithms
5. **Explainability** using SHAP (SHapley Additive exPlanations)

---

##  Features

-  **100,000-record dataset** with realistic financial distributions
-  **Non-linear risk heuristics** for authentic credit risk modeling
-  **Comprehensive EDA** with modern academic-style visualizations
-  **Feature Selection** with VarianceThreshold to remove noise features
-  **Multi-model comparison**: Random Forest, XGBoost, LightGBM
-  **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
-  **SHAP Explainability** for model interpretability
-  **Publication-quality plots** using seaborn "mako" palette

---

##  Dataset

The dataset is **synthetically generated** with 100,000 samples and 12 features, simulating realistic SME financial profiles:

| Feature | Description | Distribution |
|---------|-------------|--------------|
| `Annual_Income` | Annual income of the SME | Log-normal (μ=12.5, σ=0.8), clipped [50K–5M] |
| `Debt_Ratio` | Ratio of debt to assets | Beta (a=2, b=5) |
| `Loan_Amount` | Requested loan amount | Log-normal (μ=11.5, σ=0.7), clipped [10K–2M] |
| `Years_in_Business` | Years since the business was established | Uniform integer [1–25] |
| `Credit_History_Length` | Length of credit history (years) | Uniform integer [1–20] |
| `Previous_Defaults` | Number of previous loan defaults | Poisson (λ=0.3) |
| `Supplier_Payment_Delay` | Average delay in paying suppliers (days) | Normal (μ=15, σ=12), clipped [0–90] |
| `Inventory_Turnover` | Inventory turnover ratio | Uniform [1–20] |
| `Cash_Flow_Stability` | Cash flow stability index | Beta (a=5, b=2) |
| `Sector_Risk_Score` | Risk score of the business sector | Uniform [10–90] |
| `Random_Noise` | Noise variable (for feature selector testing) | Normal (μ=0, σ=1) |
| `Risk` | **Target Variable** — 0 (Low Risk) / 1 (High Risk) | Binary |

### Risk Label Generation

The target variable is computed using a non-linear heuristic combining multiple risk factors:

```python
risk_prob = (
    0.35 * (Debt_Ratio > 0.55) +
    0.25 * (Previous_Defaults >= 1) +
    0.20 * (Cash_Flow_Stability < 0.3) +
    0.10 * (Sector_Risk_Score > 70) +
    0.10 * (Supplier_Payment_Delay > 30)
)
```

Statistical noise is applied before thresholding at 0.45 to determine the final binary risk label, producing a **~5% high-risk rate** (imbalanced classes).

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data Processing** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn`, `xgboost`, `lightgbm` |
| **Evaluation** | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Explainability** | `shap` |
| **Feature Selection** | `VarianceThreshold` |

---

##  Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn shap
```

### Clone the Repository

```bash
git clone https://github.com/M-RahulReddy27/ML-project-for-SME-credit-risk-prediction.git
cd ML-project-for-SME-credit-risk-prediction
```

---

##  Usage

### Run the Notebook

Open and execute `SM_credit_Risk.ipynb` in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook SM_credit_Risk.ipynb
```

### Pipeline Steps

The notebook is organized into sequential cells:

1. **Cell 1** — Import libraries & generate the synthetic dataset
2. **Cell 2** — Exploratory Data Analysis (EDA) with visualizations
3. **Cell 3+** — Feature engineering, model training, evaluation, and SHAP analysis

---

##  Model Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA GENERATION                          │
│  Generate 100K records with non-linear risk heuristics      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               EXPLORATORY DATA ANALYSIS                     │
│  Distribution plots, correlation matrix, class balance      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            FEATURE ENGINEERING & SELECTION                   │
│  VarianceThreshold to remove low-variance / noise features  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 TRAIN / TEST SPLIT                          │
│  Stratified split to preserve class distribution            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              MODEL TRAINING & EVALUATION                    │
│  Random Forest · XGBoost · LightGBM                         │
│  Metrics: Accuracy, Precision, Recall, F1, ROC-AUC          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                SHAP EXPLAINABILITY                           │
│  Feature importance, SHAP summary & dependence plots        │
└─────────────────────────────────────────────────────────────┘
```

---

##  Results

### Models Compared

| Model | Description |
|-------|-------------|
| **Random Forest** | Ensemble of decision trees with bootstrap aggregation |
| **XGBoost** | Gradient-boosted decision trees with regularization |
| **LightGBM** | Light Gradient Boosting Machine for fast, efficient training |

### Key Metrics

Each model is evaluated on the following metrics after training on the 100K-sample dataset:

- **Accuracy** — Overall correctness of predictions
- **Precision** — How many predicted high-risk SMEs are actually high-risk
- **Recall** — How many actual high-risk SMEs are correctly identified
- **F1-Score** — Harmonic mean of Precision and Recall
- **ROC-AUC** — Area under the Receiver Operating Characteristic curve

> **Note**: Due to the ~5% class imbalance, Recall and ROC-AUC are particularly important metrics for evaluating high-risk detection capability.

---

##  Project Structure

```
ML-project-for-SME-credit-risk-prediction/
│
├── SM_credit_Risk.ipynb           # Main Jupyter Notebook (full pipeline)
├── kaggle_sme_credit_risk_100k.csv  # Generated dataset (100K records)
├── Finanical_data_sme.csv         # Additional financial data
├── financial_dataset_SME.csv      # Supplementary SME financial dataset
└── README.md                      # Project documentation
```

---

##  Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

##  License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

##  Contact

For questions, suggestions, or collaborations, please open an issue in the repository.

---

> **Disclaimer**: The dataset used in this project is synthetically generated for educational and research purposes. It does not represent real financial data from any organization.
