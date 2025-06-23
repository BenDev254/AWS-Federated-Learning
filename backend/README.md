# 🏥 Federated Learning Director – MIMIC-IV Clinical Risk Prediction

A FastAPI-based platform to coordinate federated machine learning across multiple hospitals, enabling privacy-preserving clinical model training using local instances of the MIMIC-IV demo dataset.

## ✅ Project Overview

This project enables multiple hospitals (envoys) to:
- Train risk prediction models **locally** using their private data.
- Automatically **upload model weights** and logs to a central director server.
- The director can **aggregate multiple hospital models** into a global model.
- Users can run **risk inference** using either:
  - The hospital-specific model.
  - The aggregated global model.

---

## 🚀 Core Features

### 1. Hospital Model Training
- Local training script using PyTorch and MIMIC-IV `diagnoses_icd.csv`.
- Diagnoses per patient are mapped to **risk levels**: Low, Medium, High.
- Models are trained as multi-class classifiers with:
  - Input: diagnosis count
  - Output: one of three risk levels
- Results (`model.pt`, `scaler.pkl`, and training logs) are automatically uploaded to the director.

### 2. Director Backend (FastAPI)
- Endpoints for:
  - Uploading model results per hospital.
  - Running inference using a specific hospital’s model.
  - Running inference using a **global model** aggregated from multiple hospitals.
- Models are stored in:
  ```
  ./received_results/{envoy_name}/model.pt
  ```
- Aggregated models are saved in:
  ```
  ./aggregated/model.pt
  ```

### 3. Model Aggregation
- Multiple trained models can be **averaged (FedAvg-style)** into a single global model.
- Aggregation script ensures compatible model architectures.
- A global `scaler.pkl` is reused or selected from a reliable hospital for preprocessing.

### 4. Inference API
- `/infer`: Predict using a specific hospital’s uploaded model.
- `/infer_global`: Predict using the latest aggregated global model.
- Results include risk level and optionally class probabilities.

---

## 🏗️ Achievements So Far

- ✅ Successfully trained local models for:
  - KUTRRH (Kenyatta University Teaching, Referral & Research Hospital)
  - Moi Teaching and Referral Hospital (MTRH)
- ✅ FastAPI backend live for model uploads and inference.
- ✅ Working aggregation of models from multiple hospitals.
- ✅ Global model deployed and inference-ready.
- ✅ Basic logging and model versioning support.

---

## 🔧 Tech Stack

| Component | Tool/Framework |
|----------|----------------|
| Backend API | FastAPI |
| ML Framework | PyTorch |
| Data Source | MIMIC-IV Clinical Demo (S3) |
| Storage | Local File System |
| Serialization | Torch `state_dict` + `joblib` |
| Logging | Text file logs (`logs/`) |

---

## 📁 Directory Structure

```
.
├── Backend/
│   ├── main.py                  # FastAPI server
│   ├── inference.py             # Inference test script
│   ├── train.py                 # Hospital training script
│   ├── aggregate_models.py      # Aggregates model.pt files
│   ├── received_results/        # Uploaded models per envoy
│   ├── aggregated/              # Aggregated global model
│   └── logs/                    # Inference & training logs
```

---

## 📌 Next Steps

- [ ] Automate model validation before aggregation.
- [ ] Build a simple React-based admin dashboard.
- [ ] Add authentication for model uploads and inference.
- [ ] Expand training to more hospitals (KNH, Embu, Mbagathi).
- [ ] Integrate experiment tracking (e.g., MLflow or Weights & Biases).

---

## 💡 Acknowledgements

- Built using [MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimiciv-demo/).
- Inspired by privacy-preserving ML and federated healthcare applications.
- Powered by FastAPI, PyTorch, and AWS/Azure compute.