# 🏥 HealthGuard AI

## Intelligent Health Risk Prediction System using Machine Learning

HealthGuard AI is a Machine Learning-based healthcare analytics application that predicts whether an individual is **Healthy** or **Unhealthy** using physiological measurements, lifestyle habits, and medical history indicators.

The project was developed using multiple supervised learning algorithms and deployed through an interactive Streamlit dashboard for real-time health status prediction.

---

## 🎯 Problem Statement

A biomedical research institute conducts large-scale population health studies to understand how underlying health conditions influence disease risk and long-term health outcomes.

Researchers require a reliable system to distinguish between individuals with healthy profiles and those who may be at higher health risk.

HealthGuard AI addresses this challenge by leveraging Machine Learning to classify individuals based on healthcare and lifestyle data.

---

## 📊 Dataset Overview

- **Total Records:** 9,800
- **Features:** 22
- **Problem Type:** Binary Classification

### Target Variable

| Value | Class |
|---------|---------|
| 0 | Healthy |
| 1 | Unhealthy |

---

## 📋 Features Used

- Age
- BMI
- Blood Pressure
- Cholesterol
- Glucose Level
- Heart Rate
- Sleep Hours
- Exercise Hours
- Water Intake
- Stress Level
- Smoking
- Alcohol
- Diet
- Mental Health
- Physical Activity
- Medical History
- Allergies
- Diet Type (Vegan)
- Diet Type (Vegetarian)
- Blood Group AB
- Blood Group B
- Blood Group O

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib
- Joblib

---

## 🔄 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. One-Hot Encoding
5. Feature Scaling using StandardScaler
6. Model Training
7. Model Evaluation
8. Model Selection
9. Streamlit Deployment

---

## 🤖 Models Evaluated

The following machine learning algorithms were trained and evaluated:

| Model |
|---------|
| Logistic Regression |
| K-Nearest Neighbors (KNN) |
| Random Forest Classifier |
| Gradient Boosting Classifier |
| Voting Classifier |

---

## 🏆 Final Model Selection

### Random Forest Classifier

The final model was selected based on **Recall Score**, which is particularly important in healthcare applications.

### Why Recall?

In healthcare prediction systems, a False Negative can be costly because an unhealthy individual may be incorrectly classified as healthy.

A higher Recall score helps reduce missed high-risk cases.

After comparing all models, **Random Forest achieved the highest Recall score** and was therefore selected as the final deployment model.

---

## 📈 Model Performance

| Model                | Recall |
|----------------------|:------:|
| Logistic Regression  | 82.5%  |
| KNN                  | 87.5%  |
| Random Forest        | 93.7%  |
| Gradient Boosting    | 86.9%  |
| Voting Classifier    | 91.8% |
### Best Classifier that we should use for NovaGen(based on Recall) - Random Forest with accuracy of 93.7%

## 🖥️ Streamlit Dashboard Features

### 🏠 Project Overview
- Problem Statement
- Project Objectives
- Dataset Summary

### 📊 Dataset Information
- Feature Details
- Health Indicators Overview

### 🩺 Health Predictor
- Interactive Input Form
- 22 Healthcare Features
- Real-Time Prediction

### 📈 Model Information
- Final Model Details
- Health Status Prediction

---



## 📂 Project Structure

```text
HealthGuard-AI/
│
├── app.py
├── health_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
│
├── dataset/
│   └── health_data.csv
│
├── notebooks/
│   └── HealthGuard_AI.ipynb
│

```

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/HealthGuard-AI.git
cd HealthGuard-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🔮 Future Improvements

- Hyperparameter Tuning
- Feature Importance Analysis
- Explainable AI (XAI)
- XGBoost Integration
- Risk Probability Visualization
- Cloud Deployment
- User Authentication
- Health Analytics Dashboard

---

## 🎓 Academic Project Details

**Project Name:** HealthGuard AI  
**Domain:** Healthcare Analytics  
**Category:** Supervised Machine Learning  
**Dataset Size:** 9,800 Records  
**Features:** 22  
**Selected Model:** Random Forest Classifier  

---


Developed as part of a Supervised Machine Learning project focused on healthcare analytics and predictive modeling.

⭐ If you found this project useful, consider giving it a star.
