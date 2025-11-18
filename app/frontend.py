import streamlit as st
import numpy as np
import onnxruntime as ort
import requests

# Load ONNX model
session = ort.InferenceSession("artifacts/student_model.onnx")

st.title("🌾 Irrigation Scheduling Predictor")
st.write("Enter sensor readings below to predict irrigation class.")

# Define inputs (you can adjust based on your feature names)
feature_names = ["temperature", "humidity", "soil_moisture", "altitude", "rainfall", "wind_speed"]

inputs = []
for name in feature_names:
    value = st.number_input(f"{name}", value=0.0)
    inputs.append(value)

if st.button("Predict"):
    payload = {"values": inputs}
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    prediction = response.json()
    
    st.write("Raw response:", prediction)  # Debug line (optional)

    st.success(f"Predicted Irrigation Class: {prediction['predicted_class']}")

