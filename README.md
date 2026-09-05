# Titanic Survival Prediction & Binary Classification Engine

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **Executive Summary:** A supervised Machine Learning classification project analyzing passenger demographic and logistical data from the RMS Titanic to predict survival. A Random Forest classifier is trained on an 80/20 train-test split after median/mode imputation and one-hot encoding, reaching **81.6% test accuracy** and a **ROC AUC of 0.90**.

---

## 📊 Exploratory Data Analysis & Key Insights

Exploratory Data Analysis (EDA) revealed significant survival correlations based on passenger class, age demographics, and family dynamics:

| Survival by Passenger Class | Survival Distribution by Age Group |
| :---: | :---: |
| ![Survival by Class](docs/images/survival_by_class.png) | ![Survival by Age Group](docs/images/survival_by_age_group.png) |
| *1st Class passengers exhibited over 60% survival rate vs. ~24% for 3rd Class.* | *Children (<18) had significantly higher survival rates due to "women & children first" protocol.* |

---

## 💻 Tech Stack & Tooling

*   **Language:** Python 3.11
*   **Data Processing & Analytics:** Pandas
*   **Machine Learning Framework:** Scikit-Learn
*   **Visualization:** Matplotlib
*   **Development Environment:** Spyder / Anaconda / Google Colab

---

## 🧠 Approach

`src/main.py` implements a straightforward, reproducible classification pipeline:

*   **Data Preprocessing:** Median imputation for missing `Age` values and mode imputation for missing `Embarked` values.
*   **Categorical Encoding:** One-hot encoding of `Sex` and `Embarked` via `pandas.get_dummies`.
*   **Model:** A single `RandomForestClassifier` (`n_estimators=100`, `min_samples_split=10`, `min_samples_leaf=2`, `max_features='sqrt'`, `random_state=42`).
*   **Evaluation:** An 80/20 train-test split, scored via a ROC curve and ROC AUC (no cross-validation or hyperparameter search is performed).

---

## 📈 Model Performance

Reproduced directly from `src/main.py` on the held-out 20% test split (`random_state=42`):

| Metric | Value |
| :--- | :---: |
| Accuracy | 81.6% |
| Precision | 80.6% |
| Recall | 73.0% |
| F1-Score | 76.6% |
| ROC AUC | 0.90 |

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.10+
*   pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ClxpI/ML-WhoSurvivedTheTitanic.git
   cd ML-WhoSurvivedTheTitanic
   ```
2. **Install dependencies and run:**
   ```bash
   pip install -r requirements.txt
   python src/main.py
   ```
