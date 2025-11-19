# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

app = FastAPI()

# CORS so frontend can call backend on Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX model once at startup
session = ort.InferenceSession("artifacts/student_model.onnx")


class InputData(BaseModel):
    values: list[float]  # {"values": [temp, humidity, ...]}


@app.post("/predict")
def predict(data: InputData):
    """
    Input:
    {
        "values": [temperature, humidity, soil_moisture,
                   altitude, rainfall, wind_speed]
    }
    """
    features = np.array([data.values], dtype=np.float32)

    outputs = session.run(["output"], {"input": features})
    predicted_class = int(np.argmax(outputs[0]))

    return {"predicted_class": predicted_class}
