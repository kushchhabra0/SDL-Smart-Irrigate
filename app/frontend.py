# frontend.py

import requests
import streamlit as st
import pandas as pd
from datetime import datetime

# ========= CONFIG =========
BACKEND_URL = "http://127.0.0.1:8000"  # change if FastAPI runs elsewhere

st.set_page_config(
    page_title="Smart Irrigation – Scheduling Predictor",
    layout="wide",
)

# ========= CUSTOM CSS =========
st.markdown(
    """
    <style>
    /* Main background */
    body {
        background-color: #e5e7eb;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #e5e7eb;
    }

    /* Top header */
    [data-testid="stHeader"] {
        background-color: #0f172a;
        border-bottom: 1px solid #1f2937;
    }
    [data-testid="stToolbar"] {
        background-color: transparent;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #020617;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* ===============================
       FIXED SIDEBAR BUTTON COLORS
       =============================== */
    
    /* We need to target the Streamlit button container which is a div
       within the div we create with st.markdown. Streamlit components
       often sit inside a div with a fixed style class. 
       This structure uses the surrounding div for context. 
    */

    /* Normal sidebar buttons */
    .sidebar-button > div > button {
        width: 100% !important;
        background-color: #111827 !important;   /* dark grey/blue */
        color: #e5e7eb !important;
        border-radius: 6px;
        border: 1px solid #1f2937;
        padding: 0.4rem 0.75rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }

    /* Hover effect */
    .sidebar-button > div > button:hover {
        background-color: #2563eb !important;   /* blue on hover */
        color: #ffffff !important;
    }

    /* Active page button (This is the one that forces the bright blue color) */
    .active-button > div > button {
        background-color: #2563eb !important;
        color: white !important;
        border: 1px solid #1e40af !important;
        font-weight: 600;
    }

    /* Main buttons (non-sidebar) */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        color: white;
    }

    .metric-card {
        padding: 0;
        border-radius: 0;
        border: none;
        background-color: transparent;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .muted-text {
        color: #6b7280;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ========= SESSION STATE =========
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

if "page" not in st.session_state:
    st.session_state["page"] = "Home"


# ========= UTILITY FUNCTIONS =========

def custom_sidebar_button(label: str, page_name: str, key: str):
    """
    Renders a Streamlit button wrapped in a custom div to apply
    active or normal styling based on the current session state page.
    """
    is_active = st.session_state["page"] == page_name
    class_name = "active-button" if is_active else "sidebar-button"
    
    # 1. Start the custom wrapper div
    st.sidebar.markdown(f'<div class="{class_name}">', unsafe_allow_html=True)
    
    # 2. Render the actual Streamlit button inside the wrapper
    if st.sidebar.button(label, key=key, use_container_width=True):
        st.session_state["page"] = page_name
    
    # 3. Close the custom wrapper div
    st.sidebar.markdown('</div>', unsafe_allow_html=True)


def call_backend_predict(features):
    """
    Calls FastAPI backend /predict endpoint.

    Backend expects:
        { "values": [float, float, ...] }

    and returns:
        { "predicted_class": int }
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"values": features},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("predicted_class", None), data
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")
        return None, None


def map_class_to_recommendation(predicted_class: int):
    """
    Map numeric class to irrigation recommendation text.
    """
    mapping = {
        0: "Class 0 – No irrigation needed: soil moisture is sufficient.",
        1: "Class 1 – Light irrigation recommended within the next few hours.",
        2: "Class 2 – Moderate irrigation needed; schedule irrigation today.",
        3: "Class 3 – Heavy irrigation required immediately; soil is very dry.",
    }
    return mapping.get(
        predicted_class,
        "Unknown class – please check your model documentation.",
    )


def estimate_irrigation_duration(predicted_class: int, field_area_ha: float):
    """
    Very simple rule-of-thumb duration suggestion based on class and field area.
    You can tune these numbers later using agronomy guidance.
    """
    base_minutes_per_ha = {
        0: 0,   # no irrigation
        1: 20,  # light
        2: 40,  # moderate
        3: 60,  # heavy
    }.get(predicted_class, 30)

    total_minutes = base_minutes_per_ha * field_area_ha
    return base_minutes_per_ha, total_minutes


# ========= CROP DATA (Numerical ranges retained) =========

CROP_CONDITIONS = {
    "Rice": {
        "Temperature": "21–35 °C",
        "Humidity": "60–80 %",
        "Soil Moisture": "70–100 %", # Flooded / very wet
        "Seasonal Rainfall": "1000–2000 mm per season",
        "Wind Speed": "0–5 m/s",    # Low wind preferred
        "Notes": "Needs standing water for most of the growth period. Sensitive to water stress at flowering.",
    },
    "Wheat": {
        "Temperature": "10–24 °C",
        "Humidity": "40–60 %",
        "Soil Moisture": "35–65 %", # Moderately moist, avoid waterlogging
        "Seasonal Rainfall": "400–750 mm per season",
        "Wind Speed": "0–10 m/s",   # Low to moderate wind
        "Notes": "Sensitive to water stress at grain filling. Prefers cool, dry climate during maturity.",
    },
    "Maize": {
        "Temperature": "18–27 °C",
        "Humidity": "50–70 %",
        "Soil Moisture": "40–70 %", # Moist but well-drained
        "Seasonal Rainfall": "500–800 mm per season",
        "Wind Speed": "0–8 m/s",    # Avoid high winds during tasseling
        "Notes": "Critical stages are tasseling, silking, and grain filling.",
    },
    "Cotton": {
        "Temperature": "21–30 °C",
        "Humidity": "40–60 %",
        "Soil Moisture": "30–60 %", # Moderately moist; sensitive to waterlogging
        "Seasonal Rainfall": "600–800 mm per season",
        "Wind Speed": "0–15 m/s",   # Moderate breeze is acceptable
        "Notes": "Requires less water at maturity to avoid vegetative growth and promote boll opening.",
    },
    "Vegetables (general)": {
        "Temperature": "18–30 °C (varies by crop)",
        "Humidity": "50–80 %",
        "Soil Moisture": "50–75 %", # Uniformly moist soil; avoid drying out
        "Seasonal Rainfall": "Depends on crop; supplementary irrigation usually required",
        "Wind Speed": "0–5 m/s",    # Low wind preferred
        "Notes": "Most vegetables are very sensitive to irregular watering. Use frequent, light irrigation.",
    },
}


# ========= PAGE FUNCTIONS =========

def page_home():
    st.title("Smart Irrigation – Scheduling Assistant")

    st.write(
        """
        This application helps you decide *when* and *how much* to irrigate your field
        using sensor readings and a machine learning model.
        Use the navigation on the left to get predictions and crop-wise guidance.
        """
    )

    st.markdown(
        """
        ### What you can do here

        - *Irrigation Predictor* – Enter live sensor data and get an irrigation class (0–3)  
        - *Crop Guide* – See best environmental conditions for common crops  
        - *History* – See your recent prediction inputs and outputs on the predictor page
        """
    )


def page_predictor():
    st.title("Irrigation Scheduling Predictor")

    st.markdown(
        """
        Enter the latest values from your sensors.
        The model will predict an *irrigation class (0–3)* and give a recommendation.
        """
    )

    st.markdown(
        """
        *Class meanings*

        - *Class 0* – No irrigation needed (soil moisture sufficient)  
        - *Class 1* – Light irrigation  
        - *Class 2* – Moderate irrigation  
        - *Class 3* – Heavy irrigation (field very dry)
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input(
            "Temperature (°C)", min_value=-10.0, max_value=60.0, value=30.0
        )
        humidity = st.number_input(
            "Humidity (%)", min_value=0.0, max_value=100.0, value=60.0
        )

    with col2:
        soil_moisture = st.number_input(
            "Soil Moisture (%)", min_value=0.0, max_value=100.0, value=30.0
        )
        altitude = st.number_input(
            "Altitude (m)", min_value=-100.0, max_value=9000.0, value=300.0
        )

    with col3:
        rainfall = st.number_input(
            "Rainfall last 24h (mm)", min_value=0.0, max_value=500.0, value=0.0
        )
        wind_speed = st.number_input(
            "Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=2.0
        )

    st.markdown("---")

    col_area, col_crop = st.columns([1, 1.2])
    with col_area:
        field_area = st.number_input(
            "Field area (hectares)",
            min_value=0.1,
            max_value=1000.0,
            value=1.0,
        )
    with col_crop:
        crop_type = st.selectbox(
            "Crop Type (for your own reference)",
            ["Rice", "Wheat", "Maize", "Cotton", "Vegetables (general)", "Other"],
        )

    st.markdown("---")

    if st.button("Predict Irrigation Class"):
        features = [
            float(temperature),
            float(humidity),
            float(soil_moisture),
            float(altitude),
            float(rainfall),
            float(wind_speed),
        ]

        predicted_class, raw_response = call_backend_predict(features)

        if predicted_class is not None:
            # Show raw class
            st.success(f"Model output: *Class {predicted_class}*")

            # Human-readable explanation
            recommendation = map_class_to_recommendation(predicted_class)
            st.write(recommendation)

            # Automatic duration suggestion
            base_per_ha, total_minutes = estimate_irrigation_duration(
                predicted_class, field_area
            )
            if predicted_class == 0:
                st.info(
                    "Since this is *Class 0 (no irrigation needed)*, "
                    "no irrigation duration is suggested."
                )
            else:
                st.info(
                    f"Suggested irrigation duration: *~{total_minutes:.0f} minutes* "
                    f"for your field area ({field_area} ha**, "
                    f"≈ {base_per_ha:.0f} minutes/ha for class {predicted_class})."
                )

            # Save to history
            st.session_state["prediction_history"].append(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "crop_type": crop_type,
                    "field_area_ha": field_area,
                    "temperature": temperature,
                    "humidity": humidity,
                    "soil_moisture": soil_moisture,
                    "altitude": altitude,
                    "rainfall": rainfall,
                    "wind_speed": wind_speed,
                    "class": predicted_class,
                }
            )

            with st.expander("View raw response (debug)"):
                st.json(raw_response)

    # Prediction history
    if st.session_state["prediction_history"]:
        st.markdown(
            '<div class="section-title">Recent predictions</div>',
            unsafe_allow_html=True,
        )
        df = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(df, use_container_width=True)


def page_crop_guide():
    st.title("Crop-wise Best Conditions")

    st.write(
        """
        Select a crop to view its recommended environmental conditions.
        You can adjust these values according to your local agronomy notes.
        """
    )

    crop = st.selectbox("Choose a crop", list(CROP_CONDITIONS.keys()))
    data = CROP_CONDITIONS[crop]

    st.markdown(f"### Recommended conditions for {crop}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Temperature")
        st.write(data["Temperature"])
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Humidity")
        st.write(data["Humidity"])
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Soil Moisture")
        # REMOVED BOLDING: f'**{data["Soil Moisture"]}**' -> data["Soil Moisture"]
        st.write(data["Soil Moisture"]) 
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Seasonal Rainfall")
        st.write(data["Seasonal Rainfall"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Wind Speed")
    # REMOVED BOLDING: f'**{data["Wind Speed"]}**' -> data["Wind Speed"]
    st.write(data["Wind Speed"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Notes</div>', unsafe_allow_html=True)
    st.write(data["Notes"])

    # Optional: comparison table of all crops
    st.markdown('<div class="section-title">Compare crops</div>', unsafe_allow_html=True)
    table_data = []
    for name, vals in CROP_CONDITIONS.items():
        row = {
            "Crop": name,
            "Temperature": vals["Temperature"],
            "Humidity": vals["Humidity"],
            "Soil Moisture": vals["Soil Moisture"],
            "Seasonal Rainfall": vals["Seasonal Rainfall"],
            "Wind Speed": vals["Wind Speed"],
        }
        table_data.append(row)

    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True)


# ========= SIDEBAR NAVIGATION (buttons, no radio) =========

with st.sidebar:
    st.title("Smart Irrigation")
    st.markdown("---")

    # Use the helper function to render all buttons correctly
    # and automatically apply 'active-button' or 'sidebar-button' class.
    
    custom_sidebar_button("Home", "Home", key="btn_home")
    custom_sidebar_button("Irrigation Predictor", "Irrigation Predictor", key="btn_predict")
    custom_sidebar_button("Crop Guide", "Crop Guide", key="btn_crop_guide")

    st.markdown("---")
    st.markdown(f"*Current page:* {st.session_state['page']}")

# ========= ROUTER =========

page = st.session_state["page"]

if page == "Home":
    page_home()
elif page == "Irrigation Predictor":
    page_predictor()
elif page == "Crop Guide":
    page_crop_guide()