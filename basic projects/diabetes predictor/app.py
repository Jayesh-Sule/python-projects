import streamlit as st
import numpy as np
import pickle
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide"
)
st.markdown("""
<style>
.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #ffffff;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
.predict-btn {
    background-color: #8A2BE2;
    color: white;
    font-size: 20px;
    padding: 12px;
    width: 100%;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
model =  pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
st.title(" 🩺 Diabetes Prediction System")
st.write("### Enter patient details to check diabetes risk.")
st.markdown("---")
st.sidebar.header("📌 About App")
st.sidebar.info(
    """
    This machine learning model predicts whether a person 
    has diabetes based on health details.
    """
)

st.sidebar.write("Built with ❤️ using Streamlit & ML")
col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    Glucose = st.number_input("Glucose Level", 0, 300, 120)

with col2:
    BloodPressure = st.number_input("Blood Pressure", 0, 200, 70)
    SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)

with col3:
    Insulin = st.number_input("Insulin Level", 0, 900, 80)
    BMI = st.number_input("BMI", 0.0, 70.0, 25.0)

col4, col5 = st.columns(2)
with col4:
    DiabetesPedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

with col5:
    Age = st.number_input("Age", 1, 120, 30)
if st.button("🔮 Predict", use_container_width=True):

    st.write("### ⏳ Analyzing your health data...")
    progress = st.progress(0)
    status_text = st.empty()
    for i in range(101):
        progress.progress(i)
        status_text.text(f"Processing... {i}%")
        time.sleep(0.02)  
    status_text.text("Completed ✔")
    time.sleep(0.3)
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigree, Age]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    if prediction == 1:
        st.markdown("""
        <div class='result-box'>
        <h2 style='color: red;'>⚠️ High Risk of Diabetes</h2>
        <p>Please consult a doctor for further medical tests.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='result-box'>
        <h2 style='color: green;'>✅ No Diabetes Detected</h2>
        <p>Your health indicators look normal.</p>
        </div>
        """, unsafe_allow_html=True)
