from fastapi import Body

@app.post("/envoy/{envoy_id}/add_diagnosis")
def add_local_diagnosis(
    envoy_id: int,
    diagnosis_count: int = Body(...),
    risk_level: int = Body(...),  # 0=Low, 1=Medium, 2=High
    db: Session = Depends(get_db)
):
    envoy = db.query(Envoy).filter(Envoy.id == envoy_id).first()
    if not envoy:
        raise HTTPException(status_code=404, detail="Envoy not found")

    entry = LocalDiagnosis(
        envoy_id=envoy_id,
        diagnosis_count=diagnosis_count,
        risk_level=risk_level
    )
    db.add(entry)
    db.commit()
    return {"message": "Diagnosis added"}


import boto3
import pandas as pd
from io import StringIO

@app.post("/envoy/{envoy_id}/export_csv")
def export_envoy_data_to_csv(envoy_id: int, db: Session = Depends(get_db)):
    envoy = db.query(Envoy).filter(Envoy.id == envoy_id).first()
    if not envoy:
        raise HTTPException(status_code=404, detail="Envoy not found")

    # Fetch local data
    data = db.query(LocalDiagnosis).filter(LocalDiagnosis.envoy_id == envoy_id).all()
    if not data:
        raise HTTPException(status_code=404, detail="No local data found")

    # Convert to DataFrame
    df = pd.DataFrame([{
        "diagnosis_count": d.diagnosis_count,
        "risk_level": d.risk_level
    } for d in data])

    # Convert to CSV in memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Upload to S3
    s3 = boto3.client("s3", region_name=envoy.region)
    s3_key = f"{envoy.s3_prefix}/envoy_data.csv"
    s3.put_object(Bucket=envoy.s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())

    return {"message": f"Exported data to s3://{envoy.s3_bucket}/{s3_key}"}


from fastapi import APIRouter, HTTPException
import json, os
import sagemaker
from sagemaker.pytorch import PyTorch

router = APIRouter()

@router.post("/launch_training/{envoy_name}")
def launch_training(envoy_name: str):
    envoy_dir = os.path.join("envoys", envoy_name)
    config_path = os.path.join(envoy_dir, "config.json")
    train_script = os.path.join(envoy_dir, "train.py")

    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Envoy config not found")
    
    with open(config_path) as f:
        config = json.load(f)

    dataset_s3 = config["dataset_s3"]
    output_s3 = config["s3_output_prefix"]
    region = config.get("region", "eu-north-1")

    # Optional: support per-envoy credentials in future
    session = sagemaker.Session()
    role = "arn:aws:iam::853869586998:role/service-role/AmazonSageMaker-ExecutionRole-20250618T101261"

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=envoy_dir,
        role=role,
        framework_version="1.13.1",
        py_version="py39",
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=output_s3,
        base_job_name=f"{envoy_name.lower()}-fl-training"
    )

    estimator.fit({
        "training": dataset_s3
    })

    return {"message": f"Training launched for {envoy_name}"}
