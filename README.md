
# 🏥 TechLife FL: Federated Learning for Universal Healthcare

> *Secure AI collaboration for hospitals—without compromising patient privacy.*

**TechLife FL** is a full-stack Federated Learning platform that allows hospitals to collaboratively train machine learning models on sensitive patient data—without sharing the data itself. Built for both **Doctors** and **Data Scientists**, the platform promotes data sovereignty while enabling smarter healthcare across Africa and beyond.

---

# Live Version

You can access the live test application here = https://demo-fl-frontend-eqapesdxgmaganhz.canadacentral-01.azurewebsites.net/ 

- Doctor Credentials - auth = doc; pass = doc
- Data Scientist Credentials - auth = pass; pass = pass

- Note -- Some of the data in this demo environment is test data and there not complete. Kindly generate fresh new envoys for your test. 


---

### ⚠️ Important Notice

Liaise with the resident doctor before any training. Share the training materials only first, and wait for the hospital to treat a number of patients.  
Only then should you launch training.  

Run **Global Inference** only after training models on a number of envoys successfully and the doctor exporting the materials to their individual S3 buckets, to avoid runtime errors.  

📧 **For any assistance, contact the lead Data Scientist Benard via [benard@techlife.africa](mailto:benard@techlife.africa)**

---


## ⚙️ Setup Instructions

### 1️⃣ Clone and Navigate
```bash
git clone https://github.com/BenDev254/AWS-Federated-Learning.git

```
The are two directories; frontend and backend, run these two directories in two instances 

---

### 2️⃣ Backend (FastAPI)

#### 📦 Set up Virtual Environment and Install Requirements
```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

---

#### 📄 Create a `.env` File in the Root Directory
```ini
DATABASE_URL=your-db-url # i.e postgresql://fastapi_user:pass@localhost:5432/federated_db

AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=eu-north-1

DIRECTOR_S3_BUCKET_NAME=your-s3-bucket # i.e fl-training-results-techlife
ENVOY_S3_BUCKET_NAME=your-s3-bucket # i.e. fl-training-envoy-a # You can use the same bucket for testing different folders

SECRET_KEY=your-secret

```


---

#### 🗃️ Create and Configure PostgreSQL Locally (Linux)


```bash -- Linux Installation
# Install postgres if you haven't done so yet
sudo apt install postgresql postgresql-contrib -y
```


```bash -- Linux start service in Linux 
# Start the postgresql server
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

```bash
# Switch to postgres system user
sudo -i -u postgres
```

Then run:

```bash
# Create database
createdb federated_db

# Access PostgreSQL shell
psql
```

Inside the PostgreSQL shell:

```sql
-- Create user with password 'pass'
CREATE USER fastapi_user WITH PASSWORD 'pass';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE federated_db TO fastapi_user;
\q
```

Exit the postgres user:
```bash
exit
```

---

#### ⚙️ Apply DB Migrations
```bash
alembic upgrade head
```

---

#### 🚀 Run FastAPI Server
```bash
uvicorn main:app --reload
```

Access at: [http://localhost:8000](http://localhost:8000)

---

### 3️⃣ Frontend (React)

```bash
cd frontend
npm install
npm start
```

Frontend runs at: [http://localhost:3000](http://localhost:3000)

---

## 🔐 Authentication System

- JWT-based authentication
- Role-based access:
  - `doctor`: patient records, diagnosis, inference
  - `data_scientist`: training, model aggregation, deployment

---

## 🩺 Platform Roles & Features

### 👨‍⚕️ Doctor Dashboard

- Register patients (name → secure ID)
- Enter ICD-10 diagnoses + risk levels
- View and manage local patient data
- Upload to S3 for training
- Run local or global inference

### 🧪 Data Scientist Dashboard

- Create and manage hospital envoys
- Package training jobs
- Launch distributed training
- Aggregate models globally
- Distribute models to envoys
- Trigger inference at scale




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

## 🧠 About the Machine Learning Model

The platform includes a machine learning model that **classifies patient risk levels** using structured diagnosis data. It supports **decentralized training** using **Federated Learning** and is based on the **MIMIC-IV dataset**, a real-world critical care database.

---

### 📊 Source Dataset: **MIMIC-IV**
- Developed by MIT
- 40,000+ anonymized ICU patients
- Includes ICD-coded diagnoses
- Used here for demonstration purposes

---

### 🔍 Model Objective: **Patient Risk Stratification**

Predicts whether a patient’s condition is **Low**, **Medium**, or **High** risk, based on structured diagnosis history. Ideal for triage, early warning, and health system optimization.

---

### 🧬 Input Format

Doctors input structured diagnosis info:

```json
{
  "hadm_id": "H00123",
  "icd_code": "I10",
  "seq_num": 1,
  "diagnosis_count": 3
}
```

---

### 📥 Model Features

| Feature              | Description                             |
|----------------------|-----------------------------------------|
| `diagnosis_count`    | Number of ICD entries per patient       |
| `icd_code`           | ICD-10 diagnosis code (encoded)         |
| `seq_num`            | Order of diagnosis during the visit     |
| `icd_version`        | ICD version (9 or 10)                   |

---

### 🎯 Output (Target Variable)

- `Low` (0–1 diagnoses)
- `Medium` (2–4 diagnoses)
- `High` (5+ diagnoses)

Labels are synthetic by default but can be improved with real annotations.

---

### 🧠 Model Architecture

- **Type:** Multi-class classifier
- **Tech:** PyTorch (local training)
- **Options:** Logistic Regression / Shallow Neural Net
- **Training Mode:** Federated (local envoy → global aggregation)
- **Loss Function:** CrossEntropyLoss

---

### 🔁 Federated Learning Workflow

1. Envoy trains a local model on its dataset
2. Trained weights (not data) are sent to the Director
3. Director aggregates into a global model
4. Global model is pushed back to envoys

---

### ⚠️ Limitations

- Current data is synthetic/test
- Labels are heuristically derived
- No personal or demographic data yet used
- ICD encoding is basic; embeddings can be improved

---

### 🧪 Future Roadmap

- Add real hospital triage criteria
- Incorporate age, sex, vitals, medications
- Enhance ICD code embeddings
- Track per-envoy performance and model history
- Introduce secure aggregation and privacy tooling

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

## 📜 License

Licensed under the MIT License.

---

## 🙏 Acknowledgements

- MIMIC-IV Dataset (MIT Lab for Computational Physiology)
- FastAPI, SQLAlchemy, Alembic
- React, React Router, Axios
- PyTorch, AWS S3
- ICD-10 and standard healthcare terminologies



## 🛡️ Ownership & Demo Use Only

This codebase is the intellectual property of **Techlife Collective** and is intended solely for **demo and evaluation purposes** during public showcases such as hackathons.

Unauthorized commercial use or redistribution is prohibited.

For licensing inquiries, partnership discussions, or feedback, please contact:

- 🌐 Website: [www.techlife.africa](https://www.techlife.africa)
- 📧 Lead Developer: [benard@techlife.africa](mailto:benard@techlife.africa)