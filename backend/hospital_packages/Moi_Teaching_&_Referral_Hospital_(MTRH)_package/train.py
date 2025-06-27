import os
import json
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import gzip
import shutil
from botocore import UNSIGNED
from botocore.config import Config
import boto3
from datetime import datetime
import joblib




# Time stamp log
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")



def categorize_risk(count):
    if count <= 5:
        return 0  # Low
    elif 6 <= count <= 15:
        return 1  # Medium
    else:
        return 2  # High



# Load Config
with open("config.json", "r") as f:
    config = json.load(f)

HOSPITAL_NAME = os.environ.get("HOSPITAL_NAME", "Unnamed_Hospital")
DATASET_URL = config["dataset_s3"]
UPLOADS_URLS = config.get("upload_urls") or [config["upload_url"]]

# Step 1: Download dataset
diagnoses_file = "diagnoses_icd.csv"
compressed_file = "diagnoses_icd.csv.gz"

if not os.path.exists(diagnoses_file):
    print("Downloading compressed dataset...")
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3.download_file("physionet-open", "mimic-iv-demo/hosp/diagnoses_icd.csv.gz", compressed_file)

    print("Decompressing dataset...")
    with gzip.open(compressed_file, "rb") as f_in, open(diagnoses_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# Step 2: Load and preprocess
df = pd.read_csv(diagnoses_file)
df = df[['subject_id', 'icd_code']].dropna()

# Step 3: Count diagnoses per subject
diag_counts = df.groupby('subject_id').count().rename(columns={"icd_code": "diag_count"})

# Step 4: Categorize using bins
diag_counts['risk_level'], bins = pd.qcut(
    diag_counts['diag_count'],
    q=3,
    labels=[0, 1, 2],
    retbins=True,
    duplicates='drop'
)
diag_counts['risk_level'] = diag_counts['risk_level'].astype(int)

print(f"Binning thresholds used: {bins}")

# Step 5: Prepare data
X = diag_counts[['diag_count']].values
y = diag_counts['risk_level'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Show class distribution
print("Train class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Test class distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

# Convert to torch tensors
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

# Step 6: Define model
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 3)  # 3 classes
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 7: Train model
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# Step 8: Evaluate
model.eval()
with torch.no_grad():
    output = model(X_test)
    preds = torch.argmax(output, dim=1)

report = classification_report(
    y_test.numpy(),
    preds.numpy(),
    target_names=["Low", "Medium", "High"]
)

print(report)

# After fitting the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
os.makedirs("results", exist_ok=True)
joblib.dump(scaler, "results/scaler.pkl")

# Step 9: Save results
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/model.pt")
with open("results/log.txt", "a") as f:
    f.write(f"=== Training Session @ {timestamp} for {HOSPITAL_NAME} ===\n")
    f.write("Training complete\n")
    f.write(report)
    f.write("\n\n")


# Step 10: Upload results
for url in UPLOADS_URLS:
    try:
        with open("results/model.pt", "rb") as m, open("results/log.txt", "rb") as l:
            files = {"model": m, "log": l}
            data = {"envoy_name": HOSPITAL_NAME}
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            print(f"Upload to {url} successful.")
            break
    except Exception as e:
        print(f"Upload to {url} failed: {e}")
