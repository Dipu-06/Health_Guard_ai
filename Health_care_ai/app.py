import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:
    model = joblib.load("health_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("Model or scaler file not found.")
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<h1 style='text-align:center; color:#2E86C1;'>
🏥 HealthGuard AI
</h1>

<h3 style='text-align:center; color:gray;'>
Intelligent Health Risk Prediction System
</h3>

<hr>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Project Overview",
        "📊 Dataset Information",
        "🩺 Health Predictor"
    ]
)

# ==================================================
# PAGE 1 : OVERVIEW
# ==================================================

if page == "🏠 Project Overview":

    st.header("🎯 Project Objective")

    st.write("""
    HealthGuard AI is a machine learning-based healthcare
    decision support system developed to classify individuals
    as Healthy or Unhealthy using physiological measurements,
    lifestyle indicators, and medical history attributes.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Records", "9,800")
    col2.metric("Features", "22")
    col3.metric("Target Classes", "2")
    col4.metric("Model", "Logistic Regression")

    st.markdown("---")

    st.subheader("📌 Applications")

    st.write("""
    - Clinical Trial Participant Selection
    - Population Health Studies
    - Health Risk Assessment
    - Preventive Healthcare Analytics
    - Medical Research Support
    """)

    st.subheader("🔄 Machine Learning Workflow")

    st.write("""
    1. Data Collection
    2. Data Cleaning
    3. Feature Engineering
    4. Data Scaling
    5. Model Training
    6. Model Evaluation
    7. Health Risk Prediction
    """)

# ==================================================
# PAGE 2 : DATASET INFO
# ==================================================

elif page == "📊 Dataset Information":

    st.header("📋 Dataset Features")

    feature_df = pd.DataFrame({
        "Feature": [
            "Age",
            "BMI",
            "Blood Pressure",
            "Cholesterol",
            "Glucose Level",
            "Heart Rate",
            "Sleep Hours",
            "Exercise Hours",
            "Water Intake",
            "Stress Level",
            "Smoking",
            "Alcohol",
            "Diet",
            "Mental Health",
            "Physical Activity",
            "Medical History",
            "Allergies",
            "Diet_Type__Vegan",
            "Diet_Type__Vegetarian",
            "Blood_Group_AB",
            "Blood_Group_B",
            "Blood_Group_O"
        ]
    })

    st.dataframe(feature_df, use_container_width=True)

    st.info("""
    Target Variable:

    0 = Healthy

    1 = Unhealthy
    """)

# ==================================================
# PAGE 3 : PREDICTOR
# ==================================================

elif page == "🩺 Health Predictor":

    st.header("🩺 Health Status Prediction")

    with st.form("prediction_form"):

        st.subheader("👤 Basic Health Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", 18, 100, 25)
            bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
            bp = st.number_input("Blood Pressure", 80, 250, 120)

        with c2:
            cholesterol = st.number_input("Cholesterol", 100, 400, 180)
            glucose = st.number_input("Glucose Level", 50, 300, 100)
            heart_rate = st.number_input("Heart Rate", 40, 180, 72)

        with c3:
            sleep = st.slider("Sleep Hours", 0.0, 12.0, 8.0)
            exercise = st.slider("Exercise Hours", 0.0, 8.0, 1.0)
            water = st.slider("Water Intake", 0.0, 8.0, 2.0)

        st.subheader("🧠 Lifestyle Information")

        c4, c5, c6 = st.columns(3)

        with c4:
            stress = st.slider("Stress Level", 1, 10, 5)
            smoking = st.selectbox("Smoking", [0, 1])
            alcohol = st.selectbox("Alcohol", [0, 1])

        with c5:
            diet = st.selectbox("Diet", [0, 1, 2, 3])
            mental = st.slider("Mental Health", 1, 10, 5)
            physical = st.slider("Physical Activity", 1, 10, 5)

        with c6:
            history = st.selectbox("Medical History", [0, 1])
            allergies = st.selectbox("Allergies", [0, 1])

        st.subheader("🥗 Diet Type")

        c7, c8 = st.columns(2)

        with c7:
            vegan = st.selectbox("Diet_Type__Vegan", [0, 1])

        with c8:
            vegetarian = st.selectbox(
                "Diet_Type__Vegetarian",
                [0, 1]
            )

        st.subheader("🩸 Blood Group")

        c9, c10, c11 = st.columns(3)

        with c9:
            blood_ab = st.selectbox(
                "Blood_Group_AB",
                [0, 1]
            )

        with c10:
            blood_b = st.selectbox(
                "Blood_Group_B",
                [0, 1]
            )

        with c11:
            blood_o = st.selectbox(
                "Blood_Group_O",
                [0, 1]
            )

        predict_btn = st.form_submit_button(
            "🔍 Predict Health Status"
        )

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if predict_btn:

        sample = np.array([[
            age,
            bmi,
            bp,
            cholesterol,
            glucose,
            heart_rate,
            sleep,
            exercise,
            water,
            stress,
            smoking,
            alcohol,
            diet,
            mental,
            physical,
            history,
            allergies,
            vegan,
            vegetarian,
            blood_ab,
            blood_b,
            blood_o
        ]])

        try:

            sample_scaled = scaler.transform(sample)

            prediction = model.predict(sample_scaled)

            if prediction[0] == 1:

                st.error("""
                ⚠️ HEALTH STATUS: UNHEALTHY

                The model indicates elevated health risk.
                """)

            else:

                st.success("""
                ✅ HEALTH STATUS: HEALTHY

                The model indicates a healthy profile.
                """)

        except Exception as e:
            st.error(f"Prediction Error: {e}")