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
import requests
import pickle
import boto3
from io import StringIO

# ───────────────────────────────────────
# Load Config
# ───────────────────────────────────────
with open("config.json", "r") as f:
    config = json.load(f)

s3_path = config["dataset_s3"]
region = config.get("region", "eu-north-1")
s3_output_prefix = config["s3_output_prefix"]
use_mimic = config.get("use_mimic", True)

dataset_path = "/opt/ml/input/data/training"
diagnoses_file = "diagnoses_icd.csv.gz"
diagnoses_path = os.path.join(dataset_path, "hosp", diagnoses_file)

# ───────────────────────────────────────
# Load MIMIC Data
# ───────────────────────────────────────
def load_mimic_data():
    df = pd.read_csv(diagnoses_path, compression="gzip")
    count_per_patient = df.groupby("subject_id").size().reset_index(name="diagnosis_count")
    count_per_patient["risk_level"] = count_per_patient["diagnosis_count"].apply(map_to_risk)
    return count_per_patient[["diagnosis_count", "risk_level"]]

# ───────────────────────────────────────
# Load Envoy Data from S3
# ───────────────────────────────────────
def load_envoy_data():
    s3 = boto3.client("s3", region_name=region)
    bucket, key = parse_s3_path(f"{s3_output_prefix}/envoy_data.csv")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj["Body"])
    return df

def parse_s3_path(s3_uri):
    s3_uri = s3_uri.replace("s3://", "")
    parts = s3_uri.split("/", 1)
    return parts[0], parts[1]

# ───────────────────────────────────────
# Risk Mapping
# ───────────────────────────────────────
def map_to_risk(count):
    if count < 5:
        return 0
    elif count < 15:
        return 1
    else:
        return 2

# ───────────────────────────────────────
# Combine Data
# ───────────────────────────────────────
frames = []

if use_mimic:
    mimic_df = load_mimic_data()
    frames.append(mimic_df)

try:
    envoy_df = load_envoy_data()
    frames.append(envoy_df)
except Exception as e:
    print(f"[WARN] No envoy data loaded: {e}")

if not frames:
    raise RuntimeError("No data available for training.")

combined_df = pd.concat(frames, ignore_index=True)
print(f"[INFO] Combined dataset shape: {combined_df.shape}")

X = combined_df[["diagnosis_count"]].values
y = combined_df["risk_level"].values

# ───────────────────────────────────────
# Scale Input
# ───────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ───────────────────────────────────────
# Split & Prepare Torch Data
# ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ───────────────────────────────────────
# Model & Train
# ───────────────────────────────────────
class RiskClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
    def forward(self, x):
        return self.model(x)

model = RiskClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

log_lines = []
for epoch in range(10):
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
    log_line = f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}"
    print(log_line)
    log_lines.append(log_line)

# ───────────────────────────────────────
# Save Artifacts
# ───────────────────────────────────────
torch.save(model.state_dict(), "model.pt")
with open("train.log", "w") as f:
    f.write("\n".join(log_lines))

# ───────────────────────────────────────
# Upload to S3
# ───────────────────────────────────────
def upload_to_s3(filename):
    s3 = boto3.client("s3", region_name=region)
    bucket, key_prefix = parse_s3_path(s3_output_prefix)
    key = f"{key_prefix}/{filename}"
    with open(filename, "rb") as f:
        s3.upload_fileobj(f, Bucket=bucket, Key=key)
    print(f"[UPLOAD] {filename} → s3://{bucket}/{key}")

for file in ["model.pt", "scaler.pkl", "train.log"]:
    upload_to_s3(file)

print("[INFO] Training & upload complete.")
