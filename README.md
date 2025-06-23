# 🏥 Federated Learning Medical Dashboard

A secure, privacy-preserving web application for hospitals to collaborate in training machine learning models using **Federated Learning**. Designed for both **Doctors** and **Data Scientists**, the platform allows hospitals (envoys) to manage patient diagnoses, train local models, and contribute to global model improvement — without sharing raw data.

---

## ⚙️ 2. Backend Setup (FastAPI)

### 🐍 Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 📄 Create `.env` File

```ini
DATABASE_URL=postgresql://fastapi_user:pass@localhost/federated_db
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=eu-north-1
S3_BUCKET_NAME=fl-training-results-techlife
SECRET_KEY=your-secret
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

---

### 🗃️ Initialize DB

Make sure PostgreSQL is running and then run:

```bash
alembic upgrade head
```

---

### 🚀 Run the FastAPI Server

```bash
uvicorn main:app --reload
```

Runs on: [http://localhost:8000](http://localhost:8000)

---

## 🌐 3. Frontend Setup (React)

Open a new terminal tab and run:

```bash
cd frontend
npm install
npm start
```

Runs on: [http://localhost:3000](http://localhost:3000)

---

## 🔐 Authentication

- JWT-based login system
- Users are either:
  - `doctor`
  - `data_scientist`
- Tokens include role-based routing in the frontend

---

## 🩺 Core Features

### 👨‍⚕️ Doctor Role

- Register patients (name → reversible subject ID)
- Record diagnoses (ICD-10, risk level)
- View patient records
- Run local/global inference on hospital data
- Upload local diagnoses CSV to S3

---

### 🧪 Data Scientist Role

- Manage hospitals (envoys)
- Package and deploy training code
- Launch local training
- Aggregate trained models into a global model
- Distribute global models to envoys
- Run global inference

---

## 📦 API Overview

### 🚑 Envoy Management

| Method | Endpoint                          | Description                  |
|--------|-----------------------------------|------------------------------|
| GET    | `/list_envoys`                    | List all envoys              |
| POST   | `/create_envoy`                   | Register a new envoy         |
| POST   | `/envoy/{id}/launch_training`     | Launch training on envoy     |
| POST   | `/aggregate`                      | Aggregate models             |
| POST   | `/distribute-global-model`        | Push global model to envoys  |

### 📋 Diagnosis Management

| Method | Endpoint                              | Description                          |
|--------|---------------------------------------|--------------------------------------|
| POST   | `/envoy/{id}/add-diagnoses`           | Add patient diagnoses                |
| POST   | `/envoy/{id}/export_diagnoses`        | Upload local diagnoses to S3         |
| GET    | `/envoy/{id}/inference/global`        | Run inference using global model     |
| GET    | `/envoy/{id}/inference/envoy-model`   | Run inference using hospital model   |

---

## 🧪 Sample Diagnosis Payload

```json
[
  {
    "subject_id": null,
    "hadm_id": "H123456",
    "icd_code": "I10",
    "icd_version": 10,
    "seq_num": 1,
    "diagnosis_count": 2,
    "risk_level": 2
  }
]
```

---

## 📁 S3 Folder Structure

```
fl-training-results-techlife/
├── envoys/
│   ├── kutrrh/
│   │   ├── diagnoses.csv
│   │   └── trained_model/{timestamp}/
│   └── ...
├── global_model/
│   └── model.pt
```

---

## 📝 License

Licensed under the MIT License.

---

## 🙏 Acknowledgements

- FastAPI, SQLAlchemy, Alembic
- React, React Router, Axios
- AWS S3, PostgreSQL
- ICD-10 coding system