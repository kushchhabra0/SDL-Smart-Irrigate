from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

app = FastAPI()
session = ort.InferenceSession("artifacts/student_model.onnx")

class InputData(BaseModel):
    values: list[float]

@app.post("/predict")
def predict(data: InputData):
    features = np.array([data.values], dtype=np.float32)
    outputs = session.run(["output"], {"input": features})
    predicted_class = int(np.argmax(outputs[0]))
    return {"predicted_class": predicted_class}
