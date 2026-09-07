❤️ Heart Disease Prediction System

A machine learning-based Heart Disease Prediction System developed with Python, Scikit-learn, and Django. The application analyzes patient health-related attributes and uses trained machine learning models to predict the likelihood of heart disease through a web-based interface.

Medical Disclaimer: This project is intended for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice.

📌 Project Overview

The goal of this project is to build an end-to-end machine learning application that:

Performs exploratory data analysis (EDA) on heart disease data.

Preprocesses patient health-related features.

Trains and evaluates multiple machine learning algorithms.

Saves trained models for later prediction.

Integrates the ML models into a Django web application.

Provides a user-friendly interface for entering patient information.

Generates a heart disease prediction and displays the result.

Stores prediction/history-related information through the Django application.

🚀 Key Features

🧠 Machine Learning-based heart disease prediction

📊 Exploratory Data Analysis and visualizations

🤖 Multiple ML algorithms for model comparison

🌐 Django web application

📝 Patient input form

🔮 Real-time prediction through the web interface

📈 Model evaluation and confusion matrices

📚 Prediction history

👤 User registration/login functionality

🎨 Responsive medical-themed frontend

💾 Saved trained ML models using Pickle

🛠️ Technologies Used

Programming & Frameworks

Python

Django

HTML5

CSS3

JavaScript

Machine Learning & Data Science

Scikit-learn

Pandas

NumPy

Matplotlib

Seaborn

Database

SQLite

Model Persistence

Pickle

🤖 Machine Learning Models

The project contains trained models for comparing different machine learning approaches, including:

Logistic Regression

Support Vector Machine (SVM)

Decision Tree

Random Forest / Ensemble approach

Gradient Boosting

The project includes saved model artifacts in:

ml_models/trained_models/

Model-related files include trained classifiers and supporting feature information.

📊 Dataset

The project uses a heart disease dataset containing patient-related clinical attributes.

The dataset is located at:

data/Heart_Disease_Prediction.csv

Example features used in the project include:

Age

Sex

Blood Pressure

Cholesterol

Chest Pain Type

Maximum Heart Rate

Exercise-Induced Angina

ST Depression

Slope of ST

Number of Vessels

Thallium

EKG-related information

The target variable represents the presence/absence of heart disease according to the dataset used for training.

🔬 Exploratory Data Analysis

The project includes several EDA visualizations and analysis results, stored in:

analysis_results/

Examples include:

Age distribution

Blood pressure distribution

Cholesterol distribution

Chest pain type distribution

EKG results distribution

Exercise-induced angina distribution

Maximum heart rate distribution

ST depression distribution

Thallium distribution

Target distribution

Correlation matrix

Model comparison

Confusion matrices

A model evaluation report is also available under:

data/analysis/model_evaluation_report.csv

📁 Project Structure

Heart-Disease-Prediction-System/
│
├── analysis_results/
│   ├── age_distribution.png
│   ├── bp_distribution.png
│   ├── cholesterol_distribution.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_svm.png
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   └── ...
│
├── data/
│   ├── analysis/
│   │   └── model_evaluation_report.csv
│   └── Heart_Disease_Prediction.csv
│
├── heart_disease_prediction/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── ml_models/
│   ├── model_trainer.py
│   └── trained_models/
│       ├── decision_tree.pkl
│       ├── ensemble.pkl
│       ├── ensemble_model.pkl
│       ├── feature_names.pkl
│       └── gradient_boosting.pkl
│
├── prediction/
│   ├── management/
│   ├── templatetags/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── context_processors.py
│   ├── forms.py
│   ├── ml_models.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── images/
│   └── js/
│       └── main.js
│
├── templates/
│   ├── pages/
│   ├── prediction/
│   ├── registration/
│   ├── base.html
│   └── sw.js
│
├── heart_disease_analysis.py
├── manage.py
├── requirements.txt
├── .gitignore
└── README.md

⚙️ Installation

1. Clone the repository

git clone https://github.com/ritin3098-bit/Heart-Disease-Prediction-System.git

2. Navigate into the project

cd Heart-Disease-Prediction-System

3. Create a virtual environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate

4. Install dependencies

pip install -r requirements.txt

▶️ Run the Django Application

Apply migrations:

python manage.py migrate

Start the development server:

python manage.py runserver

Open the application in your browser:

http://127.0.0.1:8000/

🧪 Machine Learning Workflow

The project follows an end-to-end ML workflow:

Dataset
   ↓
Data Cleaning & Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Preparation
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
Save Trained Model
   ↓
Django Integration
   ↓
Patient Input
   ↓
Heart Disease Prediction

📈 Model Evaluation

The project evaluates the trained models using standard classification metrics such as:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Model comparison results and visualizations are included in the analysis_results/ directory.

🌐 Django Application

The Django application provides a web interface where users can:

Open the prediction system.

Enter required health/clinical information.

Submit the information.

Pass the input to the trained ML model.

Receive a prediction result.

View prediction-related information/history.

The application also includes authentication-related pages such as registration, login, password reset/change, and profile pages.

🔐 Security & Git

The repository includes a .gitignore file to avoid committing unnecessary or environment-specific files such as:

__pycache__/
*.pyc
venv/
env/
.env
db.sqlite3
*.log
.idea/
.vscode/
.ipynb_checkpoints/

🎯 Future Improvements

Possible future enhancements include:

Deploying the Django application to a cloud platform.

Adding a REST API using Django REST Framework.

Improving model explainability using SHAP/LIME.

Adding additional ML algorithms.

Hyperparameter optimization.

Adding automated ML model retraining.

Improving UI/UX.

Adding comprehensive automated tests.

Adding CI/CD with GitHub Actions.

Using PostgreSQL for production deployment.

👨‍💻 Author

Ritin Setia

Machine Learning / AI Enthusiast

GitHub:
https://github.com/ritin3098-bit

⭐ Project

If you find this project useful for learning or demonstration purposes, consider giving the repository a ⭐ on GitHub.

⚠️ Disclaimer

This project is created for educational and portfolio purposes. Predictions generated by the system should not be interpreted as medical diagnoses. Always consult a qualified healthcare professional for medical decisions.
