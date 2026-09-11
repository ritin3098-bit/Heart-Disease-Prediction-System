# ❤️ Heart Disease Prediction System

A **Django-based machine learning application** that predicts the likelihood of heart disease from patient clinical data, benchmarking multiple classification algorithms end-to-end — from EDA to a live prediction interface.

> ⚠️ **Medical Disclaimer:** Built for educational and portfolio purposes only. This is not a diagnostic tool and should never substitute professional medical advice.

**[▶ Demo Video](https://drive.google.com/file/d/1E6pjhdZ6qW8GyH4a0M08DxJ-I1TqZ7P-/view)**

---

## 🚀 Overview

Cardiovascular disease is one of the leading causes of death worldwide, and early risk detection from routine clinical data is a well-studied but genuinely hard classification problem — feature relationships are noisy, class boundaries overlap, and no single algorithm dominates across all datasets. This project treats that as an applied ML engineering challenge rather than a one-off notebook exercise: it builds a **full pipeline from raw clinical data to a deployed prediction interface**, and rigorously compares eight algorithms rather than assuming one "best" model up front.

The workflow breaks into three stages:

1. **Data science stage** — `heart_disease_analysis.py` loads the clinical dataset, performs exploratory analysis (distributions, correlations between features like cholesterol, blood pressure, and chest pain type), and saves all visualizations as reusable artifacts under `analysis_results/`.
2. **Modeling stage** — `ml_models/model_trainer.py` preprocesses the data, trains eight classifiers on an identical train/test split, runs 5-fold cross-validation on each to check generalization (not just test-set luck), and serializes every trained model with Pickle under `ml_models/trained_models/`.
3. **Application stage** — the Django app (`prediction/`) loads the selected model at request time, presents a clinical input form, runs inference, and returns a prediction with results persisted to a per-user history — all behind standard registration/login.

This structure mirrors how a real ML feature would ship in production: research and training are decoupled from serving, so the model can be retrained or swapped without touching the web layer.

## ✨ Key Features

- 📊 Exploratory data analysis with visualizations (distributions, correlation matrix)
- 🤖 Eight ML algorithms trained and compared with 5-fold cross-validation
- 🌐 Django web app with patient input form and real-time prediction
- 📈 Model evaluation via accuracy, AUC, cross-validation, and confusion matrices
- 👤 User registration/login and prediction history
- 💾 Trained models persisted with Pickle for fast inference

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python, Django |
| **ML / Data** | Scikit-learn, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Database** | SQLite |

## 🤖 Models Compared

Eight classification algorithms were trained and evaluated on identical train/test splits, with 5-fold cross-validation to assess generalization: Logistic Regression, Random Forest, SVM, Naive Bayes, KNN, Decision Tree, Gradient Boosting, and a stacked Ensemble.

## 📊 Results

| Model | Accuracy | AUC | CV Mean (±Std) |
|---|---|---|---|
| **Logistic Regression** | **85.2%** | **0.899** | 0.833 (±0.075) |
| Naive Bayes | 85.2% | 0.893 | **0.856** (±0.060) |
| Ensemble | 85.2% | 0.882 | 0.847 (±0.070) |
| Random Forest | 83.3% | 0.867 | 0.815 (±0.055) |
| SVM | 81.5% | 0.886 | 0.634 (±0.061) |
| Gradient Boosting | 81.5% | 0.886 | 0.797 (±0.079) |
| KNN | 79.6% | 0.875 | 0.620 (±0.056) |
| Decision Tree | 79.6% | 0.800 | 0.750 (±0.060) |

**Key finding:** Logistic Regression, Naive Bayes, and the Ensemble model all tied for the best test accuracy (85.2%), with Logistic Regression edging ahead on AUC (0.899) — the highest of any model. Naive Bayes showed the strongest and most stable cross-validation performance (0.856 mean, lowest relative variance), making it a strong candidate for production robustness on unseen data. SVM and KNN, despite reasonable AUC scores, showed a much larger accuracy/CV-mean gap — a sign of overfitting to the train/test split rather than genuine generalization. Full confusion matrices and comparison plots are in [`analysis_results/`](https://github.com/ritin3098-bit/Heart-Disease-Prediction-System/tree/main/analysis_results), and raw metrics in [`data/analysis/model_evaluation_report.csv`](https://github.com/ritin3098-bit/Heart-Disease-Prediction-System/blob/main/data/analysis/model_evaluation_report.csv).

## 📁 Project Structure

```
Heart-Disease-Prediction-System/
├── analysis_results/          # EDA plots, confusion matrices, model comparison
├── data/
│   ├── Heart_Disease_Prediction.csv
│   └── analysis/model_evaluation_report.csv
├── heart_disease_prediction/   # Django project settings
├── ml_models/
│   ├── model_trainer.py
│   └── trained_models/         # Saved .pkl models
├── prediction/                 # Django app: views, forms, ML integration
├── static/ & templates/        # Frontend assets
├── heart_disease_analysis.py
├── manage.py
└── requirements.txt
```

## 💻 Quick Start

```bash
# Clone & enter the repo
git clone https://github.com/ritin3098-bit/Heart-Disease-Prediction-System.git
cd Heart-Disease-Prediction-System

# Set up environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run migrations & start the server
python manage.py migrate
python manage.py runserver
```

Visit **http://127.0.0.1:8000/**

## 📋 Clinical Features Used

The model predicts on standard cardiology risk indicators: age, sex, resting blood pressure, cholesterol, chest pain type, maximum heart rate achieved, exercise-induced angina, ST depression, slope of the ST segment, number of major vessels, thallium stress test results, and EKG findings — the same feature set used in the widely-studied UCI Heart Disease dataset.

## 🔄 How It Works

```
Dataset → Cleaning & Preprocessing → EDA → Feature Prep → Train/Test Split
        → Model Training (×8) → 5-Fold Cross-Validation → Evaluation
        → Model Selection → Save Model (Pickle) → Django Integration
        → Patient Enters Data → Model Inference → Prediction + History
```

Walking through the key stages:

- **Preprocessing & EDA**: raw clinical records are cleaned and checked for missing/invalid values, then explored visually — feature distributions, class balance, and a correlation matrix — to understand which clinical indicators actually separate positive and negative cases before any modeling begins.
- **Training & cross-validation**: rather than reporting a single train/test accuracy (which can be misleadingly optimistic or pessimistic depending on the split), every model is also run through 5-fold cross-validation, giving a mean and standard deviation that reflect how consistently it performs across different subsets of the data.
- **Model selection**: accuracy, AUC, and CV stability are weighed together rather than picking on accuracy alone — a model that scores well on one test split but has high CV variance is flagged as likely overfit (see the Results section below).
- **Serving**: the selected model is deserialized once per request inside the Django `prediction` app, so inference is fast and the web layer stays decoupled from the training code entirely.

## 🔐 Security & Git Hygiene

`.gitignore` excludes `__pycache__/`, `venv/`, `.env`, `db.sqlite3`, logs, and IDE/checkpoint files — keeping the repo clean of environment-specific artifacts.

## 🔮 Roadmap

- Deploy to a cloud platform with PostgreSQL
- REST API via Django REST Framework
- Model explainability (SHAP/LIME) + hyperparameter tuning
- Automated retraining pipeline + CI/CD via GitHub Actions
- Expanded automated test coverage

## 👨‍💻 Author

**Ritin Setia** — Machine Learning / AI Enthusiast — [GitHub](https://github.com/ritin3098-bit)

---
*Educational/portfolio project. Not a substitute for professional medical advice.*
