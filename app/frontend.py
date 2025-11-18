import streamlit as st
import requests
import onnxruntime as ort  # still loaded if you want to use locally later

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Smart Irrigation – Scheduling Predictor",
    page_icon="💧",
    layout="wide",
)

# Load ONNX model (optional, currently not used – prediction goes via API)
session = ort.InferenceSession("artifacts/student_model.onnx")

# -----------------------------
# Feature details & ranges
# -----------------------------
feature_details = {
    "Temperature (F)": "Temperature around your field, in Fahrenheit. Example: between 60 and 110°F in hot weather.",
    "Humidity": "How much moisture is in the air, in percent (%). Example: usually between 30% and 90%.",
    "Soil Moisture": "How wet the soil is, in percent (%). Example: dry soil ~10–20%, wet soil ~60–80%.",
    "Altitude": "Height of your field above sea level, in meters. If you don't know, you can approximate or keep it as default.",
    "Rainfall": "Rain received in the last 24 hours, in millimeters (mm). If no rain, keep it 0.",
    "Wind Speed": "Speed of the wind around your field, in meters per second (m/s). Light wind is around 1–5 m/s.",
}

# min, max, default for each feature – adjust as per your domain
feature_ranges = {
    "Temperature (F)": {"min": 0.0, "max": 150.0, "default": 85.0},
    "Humidity": {"min": 0.0, "max": 100.0, "default": 60.0},
    "Soil Moisture": {"min": 0.0, "max": 100.0, "default": 30.0},
    "Altitude": {"min": 0.0, "max": 4000.0, "default": 300.0},
    "Rainfall": {"min": 0.0, "max": 500.0, "default": 0.0},
    "Wind Speed": {"min": 0.0, "max": 50.0, "default": 2.0},
}

feature_names = list(feature_details.keys())

# -----------------------------
# Header
# -----------------------------
st.markdown("## 🌾 Smart Irrigation – Scheduling Predictor")
st.write(
    "Fill in the readings from your sensors. "
    "We’ll suggest how much irrigation is needed for your field."
)

# -----------------------------
# Layout – two columns for inputs
# -----------------------------
with st.form("irrigation_form"):
    left_col, right_col = st.columns(2)
    inputs = []

    for idx, name in enumerate(feature_names):
        conf = feature_ranges[name]
        col = left_col if idx % 2 == 0 else right_col

        with col:
            value = st.number_input(
                label=name,
                min_value=conf["min"],
                max_value=conf["max"],
                value=conf["default"],
            )
            # Always-visible explanation under the input
            st.caption(feature_details[name])

        inputs.append(float(value))

    st.markdown("---")

    # Crop type selection
    crop_type = st.selectbox(
        "Crop Type",
        [
            "Wheat",
            "Rice (Paddy)",
            "Maize",
            "Cotton",
            "Sugarcane",
            "Pulses",
            "Vegetables",
            "Fruits",
            "Other",
        ],
    )
    st.caption(
        "Select the crop grown in this part of your field. "
        "This helps the system understand water needs better (if the backend uses it)."
    )

    st.markdown("---")
    submit = st.form_submit_button("🔍 Predict Irrigation Class")

# -----------------------------
# Prediction logic
# -----------------------------
if submit:
    # Keep old backend format, but also send crop_type separately
    payload = {
        "values": inputs,       # original list the API already expects
        "crop_type": crop_type  # NEW field – backend can start using this later
    }

    try:
        with st.spinner("Contacting irrigation model..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            prediction = response.json()

        st.markdown("### ✅ Prediction Result")

        # Safely get predicted class
        predicted_class = prediction.get("predicted_class", "Unknown")
        st.success(f"Predicted Irrigation Class: **{predicted_class}**")

        # Show raw response for debugging (optional)
        with st.expander("See full model response (for debugging)"):
            st.json(prediction)

    except Exception as e:
        st.error(
            "Could not get a prediction from the server. "
            "Please check if the FastAPI backend is running."
        )
        st.code(str(e))
