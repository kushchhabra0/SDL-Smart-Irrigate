# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

app = FastAPI()

# Load ONNX model once at startup
session = ort.InferenceSession("artifacts/student_model.onnx")


class InputData(BaseModel):
    # The frontend will send: {"values": [ ... floats ... ]}
    values: list[float]


@app.post("/predict")
def predict(data: InputData):
    """
    Accepts JSON:
    {
        "values": [temperature, humidity, soil_moisture,
                   altitude, rainfall, wind_speed]
    }
    """
    features = np.array([data.values], dtype=np.float32)

    # Adjust "input" and "output" names if your ONNX model uses different ones
    outputs = session.run(["output"], {"input": features})
    predicted_class = int(np.argmax(outputs[0]))

    return {"predicted_class": predicted_class}
