import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "randomforest_tuned.pkl"
model = joblib.load(MODEL_PATH)

st.title("❤️ Heart Disease Risk Prediction App ❤️")

age = st.number_input("Age (years)", 20, 100, 40)
sex = st.selectbox("Sex", ("Male", "Female"))
cp = st.selectbox(
    "Chest Pain Type",
    [
        "0 - Typical angina",
        "1 - Atypical angina",
        "2 - Non-anginal pain",
        "3 - Asymptomatic"
    ]
)
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", 0.0, 10.0, 1.0)
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-3)", 0, 3, 0)
exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
restecg = st.selectbox(
    "Resting Electrocardiographic Results",
    ["Normal (0)", "ST-T Wave Abnormality (1)", "Left Ventricular Hypertrophy (2)"]
)
slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping (0)", "Flat (1)", "Downsloping (2)"]
)
thal = st.selectbox(
    "Thalassemia",
    ["Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"]
)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No (0)", "Yes (1)"])

sex_num = 1 if sex == "Male" else 0

cp_num = int(cp.split(" - ")[0])
exang_num = int(exang.split(" ")[-1])
restecg_num = int(restecg.split("(")[-1].replace(")", ""))
slope_num = int(slope.split("(")[-1].replace(")", ""))
thal_num = int(thal.split("(")[-1].replace(")", ""))
fbs_num = int(fbs.split("(")[-1].replace(")", ""))

input_data = pd.DataFrame([[
    age, sex_num, cp_num, trestbps, chol, thalach,
    oldpeak, ca, exang_num, restecg_num, slope_num, thal_num, fbs_num
]], columns=[
    "age", "sex", "cp", "trestbps", "chol", "thalach",
    "oldpeak", "ca", "exang", "restecg", "slope", "thal", "fbs"
])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

