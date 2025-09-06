
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = [
    PROJECT_ROOT / "models" / "final_model.pkl",
    PROJECT_ROOT / "models" / "svm_tuned.pkl",
    PROJECT_ROOT / "models" / "randomforest_tuned.pkl",
    PROJECT_ROOT / "models" / "logisticregression_model.pkl",
    PROJECT_ROOT / "models" / "randomforest_model.pkl",
    PROJECT_ROOT / "models" / "svm_model.pkl",
    PROJECT_ROOT / "models" / "decisiontree_model.pkl",
]

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

st.title("❤️ Heart Disease Risk Predictor")
st.write("Provide your health metrics and get a model prediction.")

def try_load_model():
    for p in MODEL_PATHS:
        if p.exists():
            return joblib.load(p)
    st.warning("No trained model found in 'models/'. Please run training first (see README).")
    return None

model = try_load_model()

with st.form("input_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex (1=male, 0=female)", [0,1], index=1)
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3], index=0)
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Cholesterol (chol)", min_value=80, max_value=700, value=230)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0,1], index=0)
    restecg = st.selectbox("Resting ECG (0-2)", [0,1,2], index=1)
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise induced angina (exang)", [0,1], index=0)
    oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of peak exercise ST segment (0-2)", [0,1,2], index=1)
    ca = st.selectbox("Number of major vessels colored by fluoroscopy (ca)", [0,1,2,3], index=0)
    thal = st.selectbox("Thalassemia (0-3 typical coding)", [0,1,2,3], index=2)

    submitted = st.form_submit_button("Predict")

if submitted:
    if model is None:
        st.stop()

    X = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])
    pred = model.predict(X)[0]
    proba = None
    try:
        proba = model.predict_proba(X)[0,1]
    except Exception:
        proba = None

    st.subheader("Prediction")
    st.write("**Risk (binary):**", int(pred))
    if proba is not None:
        st.write("**Estimated probability:**", float(proba))
    st.info("This tool is for educational purposes only and not a medical device.")
