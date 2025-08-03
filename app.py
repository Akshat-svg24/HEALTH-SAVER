import streamlit as st
import pickle
import numpy as np

# Load model and scaler (file names must match exactly)
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Example simple input UI
st.title("Heart Attack Risk Predictor")

age = st.number_input("Age", 1, 120, 45)
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
chol = st.number_input("Cholesterol", 100, 600, 200)

if st.button("Predict"):
    features = np.array([[age, cp, chol]])  # Replace with full feature list
    features_scaled = scaler.transform(features)
    result = model.predict(features_scaled)
    st.success(f"Prediction: {'High Risk' if result[0] == 1 else 'Low Risk'}")
