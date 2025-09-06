import streamlit as st
import pandas as pd
import joblib

model = joblib.load(r"D:\sprints\Heart_Disease_Project\models\randomforest_tuned.pkl")

st.title("Heart Disease Prediction App ❤️")

age = st.number_input("Age", 20, 100, 40)
sex = st.selectbox("Sex", ("Male", "Female"))
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
ca = st.number_input("Number of major vessels (ca)", 0, 3, 0)
exang = st.selectbox("Exercise induced angina (exang)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
slope = st.selectbox("Slope of peak exercise ST segment (slope)", [0, 1, 2])
thal = st.selectbox("Thalassemia (thal)", [1, 2, 3])
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1])

sex_num = 1 if sex == "Male" else 0
input_data = pd.DataFrame([[
    age, sex_num, cp, trestbps, chol, thalach, oldpeak, ca, exang, restecg, slope, thal, fbs
]], columns=["age", "sex", "cp", "trestbps", "chol", "thalach", "oldpeak", "ca", "exang", "restecg", "slope", "thal", "fbs"])


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High risk of Heart Disease")
    else:
        st.success("✅ Low risk of Heart Disease")
