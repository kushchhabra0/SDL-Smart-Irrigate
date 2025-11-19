# Smart-Irrigate – AI-Powered Irrigation Scheduling System

Smart-Irrigate is an AI-driven irrigation decision-support system designed to help farmers make informed watering decisions using real-time environmental sensor data.  
This project leverages machine learning, PySpark-based teacher models, ONNX inference, and a modern Streamlit-powered interface.

## 🌱 Overview

Smart-Irrigate provides **accurate irrigation class predictions (0–3)** based on:

- Temperature  
- Humidity  
- Soil moisture  
- Altitude  
- Rainfall  
- Wind speed  

The system analyzes sensor data and generates:

- 🌿 **Irrigation Class (0–3)**
- 💧 **Watering recommendations**
- ⏱️ **Estimated irrigation duration based on field size**
- 📘 **Crop-specific climate requirements** via Crop Guide

## 🧠 Key Features

- **Machine Learning Pipeline**: PyTorch student model, PySpark teacher model, ONNX export  
- **FastAPI Backend**: Lightweight prediction endpoint  
- **Streamlit Frontend**: Modern multi-page UI with custom styling  
- **Crop Knowledge Base**  
- **Comprehensive Dataset & EDA Notebook**  
- **Clean project structure with modular code**

## 📁 Project Structure

```
SDL-Smart-Irrigate/
├── api/                       # FastAPI backend
├── app/                       # Frontend & ML utilities
├── artifacts/                 # Trained models (ONNX/PT)
├── dataset/                   # Irrigation dataset
├── tests/                     # Unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kushchhabra0/SDL-Smart-Irrigate.git
cd SDL-Smart-Irrigate
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate    # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Backend (FastAPI)
```bash
uvicorn api.app:app --reload
```

Backend will start at:

```
http://127.0.0.1:8000
```

---

## 🖼️ Running the Frontend (Streamlit)
```bash
streamlit run app/frontend.py
```

Includes:

- Real-time sensor input  
- Recommended value ranges  
- Crop selection  
- Class prediction  
- Automatic irrigation duration  
- Prediction history  
- Crop Guide page  

---

## 🧠 Training & Exporting Models

### Train student model & export to ONNX:
```bash
python app/model_train_and_export.py
```

### Run teacher model via PySpark:
```bash
spark-submit app/pyspark_teacher.py
```

---

## 🧪 API Example Test

```bash
curl -X POST "http://127.0.0.1:8000/predict"  -H "Content-Type: application/json"  -d "{"values": [30, 60, 40, 300, 0, 2]}"
```

---

## 📊 Data Analysis

The notebook `eda-aihtproject.ipynb` contains:

- Dataset exploration  
- Distribution analysis  
- Correlation study  
- Preprocessing workflow  

---

## 🙌 Contributors

- **Kushal Chhabra** — ML, UI, Backend, DevOps  
- **Jay Talwar** — Frontend + ML Support  
- **Riya Shekawat** — Data Analysis + Documentation  

---

## 📜 License

This project is developed under SDL Lab guidelines for academic and research purposes.

