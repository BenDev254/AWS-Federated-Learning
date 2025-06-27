import os
import json
import gzip
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import boto3
from datetime import datetime

# ───────────────────────────────────────
# Load Config
# ───────────────────────────────────────
with open("config.json", "r") as f:
    config = json.load(f)

s3_path = config["dataset_s3"]
dataset_path = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")

# ───────────────────────────────────────
# Download & Load Diagnoses Data
# ───────────────────────────────────────
diagnoses_file = "diagnoses_icd.csv.gz"
diagnoses_path = os.path.join(dataset_path, diagnoses_file)

df = pd.read_csv(diagnoses_path, compression="gzip")
print(f"[INFO] Loaded diagnoses shape: {df.shape}")

# ───────────────────────────────────────
# Create Dataset: Diagnosis Count → Risk Level
# ───────────────────────────────────────
count_per_patient = df.groupby("subject_id").size().reset_index(name="diagnosis_count")

def map_to_risk(count):
    if count < 5:
        return 0  # Low
    elif count < 15:
        return 1  # Medium
    else:
        return 2  # High

count_per_patient["risk_level"] = count_per_patient["diagnosis_count"].apply(map_to_risk)

X = count_per_patient[["diagnosis_count"]].values
y = count_per_patient["risk_level"].values

# ───────────────────────────────────────
# Load Global Scaler (if available)
# ───────────────────────────────────────
global_dir = os.environ.get("SM_CHANNEL_GLOBAL", "/opt/ml/input/data/global")
global_scaler_path = os.path.join(global_dir, "scaler.pkl")

if os.path.exists(global_scaler_path):
    print("[INFO] Found global scaler — loading")
    with open(global_scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)
else:
    print("[INFO] No global scaler found — fitting new")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ───────────────────────────────────────
# Train-Test Split
# ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ───────────────────────────────────────
# Define Model
# ───────────────────────────────────────
class RiskClassifier(nn.Module):
    def __init__(self):
        super(RiskClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.model(x)

model = RiskClassifier()

# ───────────────────────────────────────
# Load Global Model (if available)
# ───────────────────────────────────────
global_model_path = os.path.join(global_dir, "model.pt")
if os.path.exists(global_model_path):
    print("[INFO] Found global model — loading weights")
    model.load_state_dict(torch.load(global_model_path))
else:
    print("[INFO] No global model found — training from scratch")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ───────────────────────────────────────
# Training Loop
# ───────────────────────────────────────
epochs = 10
log_lines = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    log_line = f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}"
    print(log_line)
    log_lines.append(log_line)

# ───────────────────────────────────────
# Save Model
# ───────────────────────────────────────
torch.save(model.state_dict(), "model.pt")

# Save logs
with open("train.log", "w") as f:
    f.write("\n".join(log_lines))

print("[INFO] Training & upload complete.")

# ───────────────────────────────────────
# Upload to S3
# ───────────────────────────────────────
try:
    s3 = boto3.client("s3", region_name=config["region"])
    s3_bucket = config["s3_bucket"]
    s3_prefix = config["s3_prefix"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def upload_to_s3(local_file):
        key = f"{s3_prefix}/{timestamp}/{os.path.basename(local_file)}"
        s3.upload_file(local_file, s3_bucket, key)
        print(f"[S3 UPLOAD] {local_file} → s3://{s3_bucket}/{key}")

    upload_to_s3("model.pt")
    upload_to_s3("scaler.pkl")
    upload_to_s3("train.log")

    print("[INFO] Training & upload complete (S3 only).")

except Exception as e:
    print(f"[ERROR] Failed to upload to S3: {e}")
