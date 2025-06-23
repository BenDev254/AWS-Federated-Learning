import os
import json
from datetime import datetime
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# === Setup FastAPI ===
app = FastAPI(title="Risk Level Inference API")

# === Define Model (same structure as in train.py) ===
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
    nn.Softmax(dim=1)
)

# === Load Trained Model and Scaler ===
model.load_state_dict(torch.load("results/model.pt"))
model.eval()

scaler = joblib.load("results/scaler.pkl")  # This must be saved during training!

# === Pydantic Input ===
class InferenceInput(BaseModel):
    diag_count: int

# === Create logs directory if not exists ===
os.makedirs("logs", exist_ok=True)
log_file = "logs/inference_logs.txt"

def log_prediction(input_count, predicted_class, class_probs):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_diag_count": input_count,
        "predicted_class": predicted_class,
        "class_probabilities": class_probs
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# === Inference Route ===
@app.post("/predict")
def predict(input_data: InferenceInput):
    try:
        X = np.array([[input_data.diag_count]])
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled).float()

        with torch.no_grad():
            probs = model(X_tensor)
            predicted_class_index = torch.argmax(probs, dim=1).item()

        risk_levels = ["Low", "Medium", "High"]
        predicted_class = risk_levels[predicted_class_index]
        class_probs = {risk_levels[i]: float(p) for i, p in enumerate(probs[0])}

        # Log the prediction
        log_prediction(input_data.diag_count, predicted_class, class_probs)

        return {
            "risk_level": predicted_class,
            "probabilities": class_probs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
