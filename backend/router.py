from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from db.database import SessionLocal
from models import Envoy, LocalDiagnosis
import nbformat
import shutil
import zipfile
import io, botocore
import os, boto3, uuid
from botocore.config import Config
import torch
import torch.nn as nn
import numpy as np
from glob import glob
import json
from datetime import datetime
from schemas import DiagnosisBatchInput, DiagnosisIn
from typing import Optional, List
import tempfile
import csv
from fastapi import UploadFile
from collections import defaultdict
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError
import sagemaker
from sagemaker.pytorch import PyTorch
from pathlib import Path
import tarfile
import re
from schemas import PatientCreate
from models import Patient 
from fastapi import Body



app = FastAPI()

BASE_DIR = "hospital_packages"
NOTEBOOKS_DIR = "./envoy_notebooks"
TEMPLATE_NOTEBOOK = "./templates/default_template.ipynb"
EXPERIMENT_BASE_PATH = "./experiments"
DEFAULT_S3_BUCKET = "fl-training-results-techlife"
DEFAULT_S3_PREFIX = "Misplaced/"
DIRECTOR_S3_BUCKET = "fl-training-results-techlife"
DIRECTOR_S3_PREFIX = "envoys"
s3 = boto3.client("s3", region_name="eu-north-1")



TEMPLATE_DIR = "templates"
TRAIN_TEMPLATE_PATH = os.path.join( "fl-training/train.py")
REQUIREMENTS_TEMPLATE_PATH = os.path.join("fl-training/requirements.txt")
SAGEMAKER_ROLE = "arn:aws:iam::853869586998:role/service-role/AmazonSageMaker-ExecutionRole-20250618T101261"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



@app.post("/create_envoy")
def create_envoy(
    name: str = Body(...),
    s3_bucket: str = Body(...),
    s3_prefix: str = Body(...),
    region: str = Body("eu-north-1"),
    db: Session = Depends(get_db)
):
    # 0. Validate inputs
    if not all([name.strip(), s3_bucket.strip(), s3_prefix.strip(), region.strip()]):
        raise HTTPException(status_code=400, detail="All fields are required and must be non-empty.")

    if db.query(Envoy).filter(Envoy.name == name).first():
        raise HTTPException(status_code=400, detail="Envoy already exists")

    # Prep values
    serial_id = str(uuid.uuid4())
    envoy_dir = os.path.join(BASE_DIR, name)
    director_key_prefix = f"{DIRECTOR_S3_PREFIX}/{name}"

    # Ensure S3 client works before writing to DB or disk
    try:
        s3 = boto3.client("s3", region_name=region)
        # Validate that bucket exists (throws error if not)
        s3.head_bucket(Bucket=s3_bucket)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or inaccessible S3 bucket: {e}")

    try:
        # 1. Create local dir
        os.makedirs(envoy_dir, exist_ok=False)

        # 2. Save config.json
        config = {
            "dataset_s3": "s3://physionet-open/mimic-iv-demo/2.2/hosp/diagnoses_icd.csv.gz",
            "director_bucket": DIRECTOR_S3_BUCKET,
            "director_prefix": director_key_prefix,
            "s3_bucket": s3_bucket,
            "s3_prefix": s3_prefix,
            "s3_output_prefix": f"s3://{s3_bucket}/{s3_prefix}",
            "envoy_dataset_s3": f"s3://{s3_bucket}/{s3_prefix}/{name}_diagnoses.csv",
            "upload_urls": [f"s3://{s3_bucket}/{s3_prefix}/uploads"],
            "region": region,
        }

        with open(os.path.join(envoy_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # 3. Serial file
        with open(os.path.join(envoy_dir, "serial.txt"), "w") as f:
            f.write(f"{serial_id}\nCreated: {datetime.utcnow().isoformat()}Z\n")

        # 4. Training files
        shutil.copyfile(TRAIN_TEMPLATE_PATH, os.path.join(envoy_dir, "train.py"))
        shutil.copyfile(REQUIREMENTS_TEMPLATE_PATH, os.path.join(envoy_dir, "requirements.txt"))

        # 5. Upload to Director S3
        for fname in ["config.json", "serial.txt", "train.py", "requirements.txt"]:
            s3.upload_file(os.path.join(envoy_dir, fname), DIRECTOR_S3_BUCKET, f"{director_key_prefix}/{fname}")

        # 6. Create S3 folders in envoy bucket
        for subdir in ["uploads/.keep", "logs/.keep", "data/.keep"]:
            s3.put_object(Bucket=s3_bucket, Key=f"{s3_prefix.rstrip('/')}/{subdir}", Body=b"")

        # 7. Only now write to DB
        envoy = Envoy(
            name=name,
            serial_id=serial_id,
            s3_output_prefix=f"s3://{s3_bucket}/{s3_prefix}",
            training_host=None,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            region=region
        )
        db.add(envoy)
        db.commit()

    except Exception as e:
        # Cleanup partial local files
        if os.path.exists(envoy_dir):
            shutil.rmtree(envoy_dir)
        raise HTTPException(status_code=500, detail=f"Envoy creation failed: {e}")

    return {
        "message": f"✅ Envoy '{name}' created successfully and stored in director S3.",
        "envoy_dir": envoy_dir,
        "serial_id": serial_id,
        "director_s3_path": f"s3://{DIRECTOR_S3_BUCKET}/{director_key_prefix}/",
        "envoy_s3_prefix": f"s3://{s3_bucket}/{s3_prefix}/"
    }

@app.delete("/delete_envoy/{envoy_id}")
def delete_envoy(envoy_id: int, db: Session = Depends(get_db)):
    try:
        # 1. Fetch envoy record
        envoy = db.query(Envoy).filter(Envoy.id == envoy_id).first()
        if not envoy:
            raise HTTPException(status_code=404, detail="Envoy not found")

        envoy_name = envoy.name
        envoy_bucket = envoy.s3_bucket
        envoy_prefix = envoy.s3_prefix
        envoy_region = envoy.region

        # 2. Delete local files
        local_envoy_path = os.path.join(BASE_DIR, envoy_name)
        try:
            if os.path.exists(local_envoy_path):
                shutil.rmtree(local_envoy_path)
        except Exception as e:
            logging.warning(f"⚠️ Failed to delete local files for {envoy_name}: {e}")

        # 3. Delete objects in envoy's S3 bucket
        try:
            delete_s3_objects(envoy_bucket, envoy_prefix)
        except ClientError as e:
            logging.warning(f"⚠️ Could not delete S3 objects in envoy bucket {envoy_bucket}: {e}")

        # 4. Delete objects in Director S3 bucket
        director_prefix = f"{DIRECTOR_S3_PREFIX}/{envoy_name}/"
        try:
            delete_s3_objects(DIRECTOR_S3_BUCKET, director_prefix)
        except ClientError as e:
            logging.warning(f"⚠️ Could not delete S3 objects in director bucket: {e}")

        # 5. Delete envoy from database
        db.delete(envoy)
        db.commit()

        return {
            "message": f"✅ Envoy '{envoy_name}' deleted from DB, local disk, and S3 (if found).",
            "envoy_deleted": envoy_name
        }

    except Exception as e:
        logging.exception("❌ Unhandled error during envoy deletion")
        raise HTTPException(status_code=500, detail="Internal Server Error")

def delete_s3_objects(bucket, prefix):
    """
    Delete all S3 objects under a given prefix.
    """
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    keys_to_delete = []
    for page in pages:
        contents = page.get("Contents", [])
        keys_to_delete.extend([{"Key": obj["Key"]} for obj in contents])

    if keys_to_delete:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": keys_to_delete})


@app.delete("/reset_database")
def reset_database(db: Session = Depends(get_db)):
    try:
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        # Recreate all tables
        Base.metadata.create_all(bind=engine)

        return {
            "message": "✅ All database tables dropped and recreated successfully."
        }

    except Exception as e:
        logging.exception("❌ Failed to reset database")
        raise HTTPException(status_code=500, detail="Internal Server Error")

        
@app.get("/list_envoys")
def list_envoys(db: Session = Depends(get_db)):
    envoys = db.query(Envoy).all()
    return [
        {
            "id": e.id,
            "name": e.name,
            "serial_id": e.serial_id,
            "region": e.region,
            "s3_bucket": e.s3_bucket,
            "s3_prefix": e.s3_prefix,
            "s3_output_prefix": e.s3_output_prefix,
            "training_host": e.training_host,
        }
        for e in envoys
    ]



@app.post("/package-training-code/{envoy_id}")
def package_training_code(envoy_id: int, db: Session = Depends(get_db)):
    try:
        # Step 1: Fetch envoy from DB (if exists)
        envoy: Envoy = db.query(Envoy).filter(Envoy.id == envoy_id).first()

        if envoy:
            source_prefix = f"{DIRECTOR_S3_PREFIX}/{envoy.s3_prefix}"
            dest_bucket = envoy.s3_bucket
            dest_prefix = envoy.s3_prefix
        else:
            source_prefix = f"{DIRECTOR_S3_PREFIX}/{envoy_id}"
            dest_bucket = DIRECTOR_S3_BUCKET
            dest_prefix = f"{FALLBACK_S3_PREFIX}/{envoy_id}"

        # Step 2: Prepare local temp directory
        tmp_dir = Path(f"/tmp/envoy_{envoy_id}")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        files_to_package = ["train.py", "requirements.txt", "config.json"]

        # Step 3: Download files from S3
        for file_name in files_to_package:
            s3_key = f"{source_prefix}/{file_name}"
            local_path = tmp_dir / file_name

            print(f"Downloading: s3://{DIRECTOR_S3_BUCKET}/{s3_key}")
            try:
                s3.download_file(DIRECTOR_S3_BUCKET, s3_key, str(local_path))
            except botocore.exceptions.ClientError as e:
                raise HTTPException(status_code=404, detail=f"File not found: {s3_key}")

        # Step 4: Create .tar.gz archive
        tar_path = tmp_dir.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for file_name in files_to_package:
                tar.add(tmp_dir / file_name, arcname=file_name)

        # Step 5: Upload to destination S3
        dest_key = f"{dest_prefix}/training_package.tar.gz"
        print(f"Uploading to: s3://{dest_bucket}/{dest_key}")
        s3.upload_file(str(tar_path), dest_bucket, dest_key)

        # Step 6: Cleanup
        shutil.rmtree(tmp_dir)
        tar_path.unlink(missing_ok=True)

        return {
            "message": "Training package archived and uploaded successfully",
            "s3_uri": f"s3://{dest_bucket}/{dest_key}"
        }

    except Exception as e:
        tb = traceback.format_exc()
        print("Unhandled Exception:\n", tb)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


from sqlalchemy import func
from typing import Union
from sqlalchemy import Integer
from sqlalchemy import func, cast, Integer

@app.post("/envoy/{envoy_id}/add-diagnoses")
def add_local_diagnoses(
    envoy_id: Union[int, str],
    diagnoses: List[DiagnosisIn],
    db: Session = Depends(get_db)
):
    # Fetch envoy
    if isinstance(envoy_id, int) or envoy_id.isdigit():
        envoy = db.query(Envoy).filter(Envoy.id == int(envoy_id)).first()
    else:
        envoy = db.query(Envoy).filter(func.lower(Envoy.name) == envoy_id.lower()).first()

    if not envoy:
        raise HTTPException(status_code=404, detail="Envoy not found")

    entries = []
    for d in diagnoses:
        # Generate subject_id if missing
        subject_id = d.subject_id
        if not subject_id:
            latest = (
                db.query(func.max(cast(LocalDiagnosis.subject_id, Integer)))
                .filter(LocalDiagnosis.envoy_id == envoy.id)
                .scalar()
            )
            next_id = (latest or 0) + 1
            subject_id = str(next_id).zfill(4)  # Format: '0001', '0002', ...

        # Save diagnosis
        entry = LocalDiagnosis(
            envoy_id=envoy.id,
            subject_id=subject_id,
            hadm_id=d.hadm_id,
            icd_code=d.icd_code,
            icd_version=d.icd_version,
            seq_num=d.seq_num,
            diagnosis_count=d.diagnosis_count,
            risk_level=d.risk_level,
            created_at=d.created_at or datetime.utcnow()
        )
        entries.append(entry)

    db.add_all(entries)
    db.commit()

    return {
        "message": f"{len(entries)} diagnoses saved successfully for envoy '{envoy.name}'.",
        "subject_ids": [e.subject_id for e in entries]
    }

@app.get("/envoy/{envoy_id}/diagnoses")
def get_envoy_diagnoses(
    envoy_id: int,
    subject_id: Optional[int] = None,
    hadm_id: Optional[int] = None,
    icd_code: Optional[str] = None,
    db: Session = Depends(get_db)
):
    # Validate envoy
    envoy = db.query(Envoy).filter(Envoy.id == envoy_id).first()
    if not envoy:
        raise HTTPException(status_code=404, detail="Envoy not found")

    # Build query with optional filters
    query = db.query(LocalDiagnosis).filter(LocalDiagnosis.envoy_id == envoy_id)

    if subject_id is not None:
        query = query.filter(LocalDiagnosis.subject_id == subject_id)
    if hadm_id is not None:
        query = query.filter(LocalDiagnosis.hadm_id == hadm_id)
    if icd_code is not None:
        query = query.filter(LocalDiagnosis.icd_code == icd_code)

    results = query.order_by(LocalDiagnosis.created_at.desc()).all()

    # Return formatted result
    return [
        {
            "id": d.id,
            "subject_id": d.subject_id,
            "hadm_id": d.hadm_id,
            "icd_code": d.icd_code,
            "icd_version": d.icd_version,
            "seq_num": d.seq_num,
            "created_at": d.created_at.isoformat() if d.created_at else None
        }
        for d in results
    ]



def get_risk_level(count: int) -> int:
    if count <= 2:
        return 0  # Low
    elif count <= 5:
        return 1  # Medium
    else:
        return 2  # High



@app.post("/envoy/{envoy_id}/export_diagnoses")
def export_envoy_diagnoses_to_s3(envoy_id: int, db: Session = Depends(get_db)):
    envoy = db.query(Envoy).filter(Envoy.id == envoy_id).first()
    if not envoy:
        raise HTTPException(status_code=404, detail="Envoy not found")

    s3_bucket = envoy.s3_bucket or DEFAULT_S3_BUCKET
    s3_prefix = envoy.s3_prefix or DEFAULT_S3_PREFIX
    fallback_used = not envoy.s3_bucket or not envoy.s3_prefix

    diagnoses = db.query(LocalDiagnosis).filter(LocalDiagnosis.envoy_id == envoy_id).all()
    if not diagnoses:
        raise HTTPException(status_code=404, detail="No diagnoses to export")

    count_map = defaultdict(int)
    for d in diagnoses:
        key = d.hadm_id or d.subject_id or d.id
        count_map[key] += 1

    updated = False
    rows = []
    for d in diagnoses:
        key = d.hadm_id or d.subject_id or d.id
        diagnosis_count = d.diagnosis_count or count_map[key]
        risk_level = d.risk_level if d.risk_level is not None else get_risk_level(diagnosis_count)

        if d.diagnosis_count != diagnosis_count or d.risk_level != risk_level:
            d.diagnosis_count = diagnosis_count
            d.risk_level = risk_level
            updated = True

        rows.append([
            d.subject_id,
            d.hadm_id,
            d.icd_code,
            d.icd_version,
            d.seq_num,
            diagnosis_count,
            risk_level,
            d.created_at.isoformat() if d.created_at else ""
        ])

    if updated:
        db.commit()

    s3 = boto3.client("s3", region_name=envoy.region or "eu-north-1")
    filename = f"{envoy.name.replace(' ', '_')}_diagnoses.csv"
    s3_key = f"{s3_prefix.rstrip('/')}/{filename}"

    try:
        existing_obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        existing_data = existing_obj["Body"].read().decode("utf-8")
        csv_buffer = io.StringIO(existing_data)
        reader = list(csv.reader(csv_buffer))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            reader = [["subject_id", "hadm_id", "icd_code", "icd_version", "seq_num", "diagnosis_count", "risk_level", "created_at"]]
        else:
            raise

    reader.extend(rows)

    out_buffer = io.StringIO()
    writer = csv.writer(out_buffer)
    writer.writerows(reader)

    s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=out_buffer.getvalue())

    # Log fallback usage
    if fallback_used:
        try:
            log_key = f"{DEFAULT_S3_PREFIX.rstrip('/')}/Misplaced.txt"
            existing_log = s3.get_object(Bucket=DEFAULT_S3_BUCKET, Key=log_key)
            log_content = existing_log["Body"].read().decode("utf-8")
        except botocore.exceptions.ClientError as e:
            log_content = "" if e.response["Error"]["Code"] == "NoSuchKey" else None
            if log_content is None:
                raise

        new_entry = f"{datetime.utcnow().isoformat()} - {envoy.name}\n"
        updated_log = log_content + new_entry
        s3.put_object(Bucket=DEFAULT_S3_BUCKET, Key=log_key, Body=updated_log)

    return {
        "message": f"Diagnosis data exported for envoy: {envoy.name}",
        "csv_s3_key": f"s3://{s3_bucket}/{s3_key}",
        "db_updated": updated,
        "fallback_used": fallback_used
    }




def check_if_exists(s3_client, bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        return e.response["Error"]["Code"] != "404"

def copy_dataset_to_envoy(bucket: str, prefix: str, region: str) -> str:
    public_bucket = "physionet-open"
    public_key = "mimic-iv-demo/2.2/hosp/diagnoses_icd.csv.gz"
    local_file = "diagnoses_icd.csv.gz"
    dest_key = f"{prefix.rstrip('/')}/diagnoses_icd.csv.gz"

    s3 = boto3.client("s3", region_name=region)
    if check_if_exists(s3, bucket, dest_key):
        return f"s3://{bucket}/{dest_key}"

    public_s3 = boto3.client("s3", config=Config(signature_version=botocore.UNSIGNED))
    public_s3.download_file(public_bucket, public_key, local_file)
    s3.upload_file(local_file, bucket, dest_key)
    os.remove(local_file)

    return f"s3://{bucket}/{dest_key}"

from urllib.parse import urlparse



@router.post("/envoy/{envoy_id}/launch_training")
def launch_sagemaker_training(envoy_id: int, db: Session = Depends(get_db)):
    try:
        envoy = db.query(Envoy).filter(Envoy.id == envoy_id).first()
        if not envoy:
            raise HTTPException(status_code=404, detail="Envoy not found")

        if not envoy.s3_bucket or not envoy.s3_prefix:
            raise HTTPException(status_code=400, detail="Envoy is missing S3 bucket or prefix")

        region = envoy.region or "eu-north-1"

        # ✅ Copy dataset and confirm the URI
        dataset_uri = copy_dataset_to_envoy(envoy.s3_bucket, envoy.s3_prefix, region)

        sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=region))

        bucket = envoy.s3_bucket
        prefix = envoy.s3_prefix.rstrip("/")
        code_archive_uri = f"s3://{bucket}/{prefix}/training_package.tar.gz"
        output_path = f"s3://{bucket}/{prefix}/uploads/"

        estimator = PyTorch(
            entry_point="train.py",
            source_dir=code_archive_uri,
            role=SAGEMAKER_ROLE,
            framework_version="1.13.1",
            py_version="py39",
            instance_count=1,
            instance_type="ml.m5.large",
            output_path=output_path,
            base_job_name=f"fl-training-{envoy.name.lower().replace(' ', '-')}",
            sagemaker_session=sagemaker_session
        )

        estimator.fit({"training": dataset_uri})

        return {
            "message": f"SageMaker training launched for envoy {envoy.name}.",
            "s3_code": code_archive_uri,
            "s3_output": output_path,
            "dataset": dataset_uri,
            "region": region
        }

    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"SageMaker error: {e}")

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected error occurred: {str(e)}"},
            headers={"Access-Control-Allow-Origin": "*"}  # ✅ Ensure CORS on error
        )



@app.post("/copy-trained-artifacts/{envoy_id}")
def copy_trained_artifacts(envoy_id: int, db: Session = Depends(get_db)):
    envoy = db.query(Envoy).filter(Envoy.id == envoy_id).first()
    if not envoy:
        raise HTTPException(status_code=404, detail="Envoy not found")

    source_bucket = envoy.s3_bucket
    base_prefix = envoy.s3_prefix.rstrip("/")
    region = envoy.region or "eu-north-1"

    s3 = boto3.client("s3", region_name=region)

    response = s3.list_objects_v2(Bucket=source_bucket, Prefix=f"{base_prefix}/")
    if "Contents" not in response:
        raise HTTPException(status_code=404, detail="No training output found")

    # Extract all timestamps using regex
    folders = set()
    for obj in response["Contents"]:
        key = obj["Key"]
        match = re.match(rf"{re.escape(base_prefix)}/(\d{{8}}_\d{{6}})/", key)
        if match:
            folders.add(match.group(1))

    if not folders:
        raise HTTPException(status_code=404, detail="No timestamped folders found")

    latest_folder = sorted(folders)[-1]
    source_prefix = f"{base_prefix}/{latest_folder}"
    dest_bucket = "fl-training-results-techlife"
    dest_prefix = f"envoys/{base_prefix}/trained_model/{latest_folder}"

    # Copy all objects under source_prefix
    copied = []
    response = s3.list_objects_v2(Bucket=source_bucket, Prefix=source_prefix)
    for obj in response.get("Contents", []):
        file_key = obj["Key"]
        file_name = file_key.split("/")[-1]
        if not file_name:
            continue  # Skip folder placeholder keys
        dest_key = f"{dest_prefix}/{file_name}"

        s3.copy_object(
            Bucket=dest_bucket,
            CopySource={"Bucket": source_bucket, "Key": file_key},
            Key=dest_key
        )
        copied.append(dest_key)

    return {
        "message": f"Copied {len(copied)} files from {source_prefix} to {dest_prefix}",
        "files_copied": copied
    }



def average_model_weights(model_paths):
    """Load and average model weights"""
    state_dicts = [torch.load(p, map_location='cpu') for p in model_paths]
    avg_state_dict = {}

    for key in state_dicts[0]:
        avg_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)

    return avg_state_dict

def average_scalers(scaler_paths):
    """Load and average StandardScaler parameters"""
    scalers = [pickle.load(open(p, 'rb')) for p in scaler_paths]

    mean_avg = sum(s.mean_ for s in scalers) / len(scalers)
    scale_avg = sum(s.scale_ for s in scalers) / len(scalers)

    from sklearn.preprocessing import StandardScaler
    avg_scaler = StandardScaler()
    avg_scaler.mean_ = mean_avg
    avg_scaler.scale_ = scale_avg
    avg_scaler.var_ = scale_avg ** 2  # optional
    avg_scaler.n_features_in_ = len(mean_avg)

    return avg_scaler

REGION = "eu-north-1"

AGGREGATOR_BUCKET = "fl-training-results-techlife"
import pickle

@app.post("/aggregate")
def aggregate_models(envoy_ids: List[int], db: Session = Depends(get_db)):
    s3 = boto3.client("s3")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"aggregator/{timestamp}"

    aggregated_models = []
    scalers = []

    for envoy_id in envoy_ids:
        envoy = db.query(Envoy).filter(Envoy.id == envoy_id).first()
        if not envoy:
            raise HTTPException(status_code=404, detail=f"Envoy ID {envoy_id} not found")

        envoy_name = envoy.name
        prefix = f"envoys/{envoy_name}/trained_model/"

        print(f"[DEBUG] Checking S3: Bucket={AGGREGATOR_BUCKET}, Prefix={prefix}")
        response = s3.list_objects_v2(Bucket=AGGREGATOR_BUCKET, Prefix=prefix)
        folders = {
            os.path.dirname(obj['Key']).split("/")[-1]
            for obj in response.get("Contents", [])
            if obj['Key'].endswith("model.pt")
        }

        if not folders:
            raise HTTPException(status_code=404, detail=f"No trained models found for envoy {envoy_name}")

        latest_folder = sorted(folders)[-1]
        base_path = f"{prefix}{latest_folder}/"

        # Download model.pt and scaler.pkl
        with tempfile.TemporaryDirectory() as tmpdir:
            local_model = os.path.join(tmpdir, "model.pt")
            local_scaler = os.path.join(tmpdir, "scaler.pkl")

            s3.download_file(AGGREGATOR_BUCKET, f"{base_path}model.pt", local_model)
            s3.download_file(AGGREGATOR_BUCKET, f"{base_path}scaler.pkl", local_scaler)

            model_state = torch.load(local_model, map_location="cpu")
            aggregated_models.append(model_state)

            with open(local_scaler, "rb") as f:
                scaler = pickle.load(f)
                scalers.append(scaler)

    # Average models (simple FedAvg)
    averaged_model = aggregated_models[0]
    for key in averaged_model:
        for other_model in aggregated_models[1:]:
            averaged_model[key] += other_model[key]
        averaged_model[key] /= len(aggregated_models)

    # Save aggregated model and scaler
    with tempfile.TemporaryDirectory() as outdir:
        agg_model_path = os.path.join(outdir, "model.pt")
        torch.save(averaged_model, agg_model_path)

        # Just pick first scaler for now (you could average scalers if needed)
        agg_scaler_path = os.path.join(outdir, "scaler.pkl")
        with open(agg_scaler_path, "wb") as f:
            pickle.dump(scalers[0], f)

        s3.upload_file(agg_model_path, AGGREGATOR_BUCKET, f"{output_prefix}/model.pt")
        s3.upload_file(agg_scaler_path, AGGREGATOR_BUCKET, f"{output_prefix}/scaler.pkl")

    return {
        "message": f"Aggregated model saved to s3://{AGGREGATOR_BUCKET}/{output_prefix}/",
        "timestamp": timestamp
    }


def average_models(models):
    """Simple averaging of PyTorch models"""
    avg_model = models[0]
    state_dicts = [model.state_dict() for model in models]

    avg_state_dict = {}
    for key in state_dicts[0]:
        avg_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)

    avg_model.load_state_dict(avg_state_dict)
    return avg_model

    s3 = boto3.client("s3", region_name=REGION)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_s3_prefix = f"aggregator/{timestamp}/"

    model_paths = []
    scaler_paths = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for envoy_id in envoy_ids:
            envoy = db.query(Envoy).filter(Envoy.id == envoy_id).first()
            if not envoy:
                raise HTTPException(status_code=404, detail=f"Envoy {envoy_id} not found")

            # Find latest timestamped folder in trained_model/
            trained_model_prefix = f"{envoy.s3_prefix.rstrip('/')}/trained_model/"
            result = s3.list_objects_v2(Bucket=envoy.s3_bucket, Prefix=trained_model_prefix)
            if 'Contents' not in result:
                raise HTTPException(status_code=404, detail=f"No trained models found for envoy {envoy.name}")

            folders = sorted({
                key['Key'].split("/")[2]
                for key in result['Contents']
                if key['Key'].endswith("model.pt")
            }, reverse=True)

            if not folders:
                raise HTTPException(status_code=404, detail=f"No valid folders found for envoy {envoy.name}")

            latest_folder = folders[0]

            for filename in ["model.pt", "scaler.pkl"]:
                s3_key = f"{trained_model_prefix}{latest_folder}/{filename}"
                local_path = os.path.join(tmpdir, f"{envoy_id}_{filename}")
                s3.download_file(envoy.s3_bucket, s3_key, local_path)
                if filename == "model.pt":
                    model_paths.append(local_path)
                else:
                    scaler_paths.append(local_path)

        # Aggregate models and scalers
        avg_weights = average_model_weights(model_paths)
        avg_scaler = average_scalers(scaler_paths)

        # Save artifacts
        model_out = os.path.join(tmpdir, "model.pt")
        scaler_out = os.path.join(tmpdir, "scaler.pkl")

        torch.save(avg_weights, model_out)
        with open(scaler_out, "wb") as f:
            pickle.dump(avg_scaler, f)

        # Upload to Director bucket
        for file_path in [model_out, scaler_out]:
            s3_key = f"{global_s3_prefix}{os.path.basename(file_path)}"
            s3.upload_file(file_path, DIRECTOR_BUCKET, s3_key)

    return {
        "message": f"Aggregation complete. Global model stored in s3://{DIRECTOR_BUCKET}/{global_s3_prefix}",
        "model_uri": f"s3://{DIRECTOR_BUCKET}/{global_s3_prefix}model.pt",
        "scaler_uri": f"s3://{DIRECTOR_BUCKET}/{global_s3_prefix}scaler.pkl"
    }


DIRECTOR_BUCKET = "fl-training-results-techlife"


@app.post("/distribute-global-model")
def distribute_global_model_to_envoys(db: Session = Depends(get_db)):
    s3 = boto3.client("s3")
    base_prefix = "aggregator/"
    files_to_copy = ["model.pt", "scaler.pkl"]

    # List timestamped folders
    print(f"[DEBUG] Listing folders under: s3://{DIRECTOR_BUCKET}/{base_prefix}")
    response = s3.list_objects_v2(Bucket=DIRECTOR_BUCKET, Prefix=base_prefix)
    if "Contents" not in response:
        raise HTTPException(status_code=404, detail="No aggregator content found")

    folders = set()
    for obj in response["Contents"]:
        parts = obj["Key"].split("/")
        if len(parts) >= 2 and re.match(r"\d{8}_\d{6}", parts[1]):
            folders.add(parts[1])

    if not folders:
        raise HTTPException(status_code=404, detail="No timestamped folders found under aggregator/")

    latest_timestamp = sorted(folders)[-1]
    latest_prefix = f"{base_prefix}{latest_timestamp}/"
    print(f"[DEBUG] Latest folder selected: {latest_prefix}")

    for file in files_to_copy:
        source_key = f"{latest_prefix}{file}"
        print(f"[DEBUG] Validating existence: s3://{DIRECTOR_BUCKET}/{source_key}")
        try:
            s3.head_object(Bucket=DIRECTOR_BUCKET, Key=source_key)
        except Exception as e:
            print(f"[ERROR] Missing file: {source_key} - {e}")
            raise HTTPException(status_code=404, detail=f"{file} not found in {latest_prefix}")

    envoys = db.query(Envoy).all()
    if not envoys:
        raise HTTPException(status_code=404, detail="No envoys found in the database")

    for envoy in envoys:
        print(f"[INFO] Distributing to Envoy: {envoy.name} (Bucket: {envoy.s3_bucket}, Prefix: {envoy.s3_prefix})")
        
        if not envoy.s3_bucket or not envoy.s3_prefix:
            print(f"[WARNING] Skipping envoy {envoy.name}: Missing S3 bucket or prefix")
            continue  # Skip misconfigured envoy

        for file in files_to_copy:
            dest_key = f"{envoy.s3_prefix.rstrip('/')}/global_model/{file}"
            source_key = f"{latest_prefix}{file}"
            try:
                print(f"[DEBUG] Copying s3://{DIRECTOR_BUCKET}/{source_key} → s3://{envoy.s3_bucket}/{dest_key}")
                source_bucket = DIRECTOR_BUCKET.replace("s3://", "")
                destination_bucket = envoy.s3_bucket.replace("s3://", "")

                s3.copy_object(
                    Bucket=destination_bucket,
                    CopySource={'Bucket': source_bucket, 'Key': source_key},
                    Key=dest_key
                )
            except Exception as e:
                print(f"[ERROR] Failed to copy {file} for {envoy.name}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to copy to {envoy.name}: {e}")

    return {
        "message": f"Global model from {latest_timestamp} distributed to {len(envoys)} envoy buckets (excluding invalid configs)",
        "timestamp": latest_timestamp
    }

import pandas as pd
import torch.nn as nn


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

@app.get("/envoy/{envoy_id}/inference/global")
def run_global_model_inference(envoy_id: int, db: Session = Depends(get_db)):
    # Fetch envoy
    envoy = db.query(Envoy).filter(Envoy.id == envoy_id).first()
    if not envoy:
        raise HTTPException(status_code=404, detail="Envoy not found")

    # Determine S3 bucket & prefix from envoy
    bucket = envoy.s3_bucket
    prefix = envoy.s3_prefix.rstrip("/")
    filename = f"{envoy.name.replace(' ', '_')}_diagnoses.csv"
    csv_key = f"{prefix}/{filename}"
    model_key = f"{prefix}/global_model/model.pt"
    scaler_key = f"{prefix}/global_model/scaler.pkl"

    s3 = boto3.client("s3", region_name=envoy.region or "eu-north-1")

    try:
        # Load CSV
        csv_obj = s3.get_object(Bucket=bucket, Key=csv_key)
        df = pd.read_csv(io.BytesIO(csv_obj['Body'].read()))
        print(f"[INFO] Diagnoses CSV shape: {df.shape}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load diagnoses CSV: {e}")

    try:
        # Load scaler
        scaler_obj = s3.get_object(Bucket=bucket, Key=scaler_key)
        scaler = pickle.load(io.BytesIO(scaler_obj['Body'].read()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load scaler: {e}")

    try:
        # Load model
        model_obj = s3.get_object(Bucket=bucket, Key=model_key)
        buffer = io.BytesIO(model_obj["Body"].read())
        model = RiskClassifier()
        model.load_state_dict(torch.load(buffer, map_location=torch.device("cpu")))
        model.eval()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load global model: {e}")

    # Risk label mapping
    label_map = {0: "Low", 1: "Medium", 2: "High"}

    try:
        # Prepare input data
        X = df[["diagnosis_count"]].values
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = model(X_tensor)
            preds = torch.argmax(logits, dim=1).numpy()

        results = []
        for i, row in df.iterrows():
            results.append({
                "subject_id": row.get("subject_id"),
                "diagnosis_count": row.get("diagnosis_count"),
                "predicted_class": int(preds[i]),
                "predicted_risk": label_map[int(preds[i])]
            })

        return {
            "envoy": envoy.name,
            "source_csv": f"s3://{bucket}/{csv_key}",
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")




import pandas as pd
import torch.nn as nn


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



@app.get("/envoy/{envoy_id}/inference/envoy-model")
def run_envoy_model_inference(envoy_id: int, db: Session = Depends(get_db)):
    # Lookup envoy
    envoy = db.query(Envoy).filter(Envoy.id == envoy_id).first()
    if not envoy:
        raise HTTPException(status_code=404, detail="Envoy not found")

    bucket = envoy.s3_bucket
    prefix = envoy.s3_prefix.rstrip("/")
    region = envoy.region or "eu-north-1"
    filename = f"{envoy.name.replace(' ', '_')}_diagnoses.csv"
    csv_key = f"{prefix}/{filename}"
    s3 = boto3.client("s3", region_name=region)

    # Step 1: Locate latest trained model timestamp folder
    try:
        
        response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/")
        contents = response.get("Contents", [])
        

        # Collect only valid timestamp folder names
        timestamps = set()
        for obj in contents:
            key = obj["Key"]
            parts = key.split("/")
            if len(parts) >= 2 and re.match(r"^\d{8}_\d{6}$", parts[1]):
                timestamps.add(parts[1])
        if not timestamps:
            raise HTTPException(status_code=404, detail="No trained model folders found")
        latest_timestamp = sorted(timestamps)[-1]
        print(f"[DEBUG] Using latest timestamp: {latest_timestamp}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find trained model timestamp: {e}")

    # Step 2: Load CSV from S3
    try:
        csv_obj = s3.get_object(Bucket=bucket, Key=csv_key)
        df = pd.read_csv(io.BytesIO(csv_obj['Body'].read()))
        print(f"[INFO] Loaded CSV shape: {df.shape}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load diagnoses CSV: {e}")

    # Step 3: Load model & scaler
    model_key = f"{prefix}/{latest_timestamp}/model.pt"
    scaler_key = f"{prefix}/{latest_timestamp}/scaler.pkl"

    try:
        model_obj = s3.get_object(Bucket=bucket, Key=model_key)
        model_buf = io.BytesIO(model_obj["Body"].read())
        model = RiskClassifier()
        model.load_state_dict(torch.load(model_buf, map_location=torch.device("cpu")))
        model.eval()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load envoy model: {e}")

    try:
        scaler_obj = s3.get_object(Bucket=bucket, Key=scaler_key)
        scaler = pickle.load(io.BytesIO(scaler_obj["Body"].read()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load envoy scaler: {e}")

    # Step 4: Run Inference
    try:
        label_map = {0: "Low", 1: "Medium", 2: "High"}
        X = df[["diagnosis_count"]].values
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = model(X_tensor)
            preds = torch.argmax(logits, dim=1).numpy()

        results = []
        for i, row in df.iterrows():
            results.append({
                "subject_id": row.get("subject_id"),
                "diagnosis_count": row.get("diagnosis_count"),
                "predicted_class": int(preds[i]),
                "predicted_risk": label_map[int(preds[i])]
            })

        return {
            "envoy": envoy.name,
            "source_csv": f"s3://{bucket}/{csv_key}",
            "model_used": f"s3://{bucket}/{model_key}",
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


import hashlib



@app.post("/api/patients/register")
def register_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    latest_hadm_id = db.query(func.max(Patient.hadm_id)).scalar() or 0
    next_hadm_id = latest_hadm_id + 1


    new_patient = Patient(
        name=patient.name,
        gender=patient.gender,
        phone=patient.phone,
        hadm_id=next_hadm_id,
        
        
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)

    return {"message": "Patient registered", "hadm_id": new_patient.hadm_id}


# In FastAPI route
@app.get("/api/patients")
def get_patients_with_diagnoses(db: Session = Depends(get_db)):
    patients = db.query(Patient).all()

    result = []
    for p in patients:
        diagnosis = (
            db.query(LocalDiagnosis)
            .filter(LocalDiagnosis.hadm_id == p.hadm_id)
            .order_by(LocalDiagnosis.created_at.desc())
            .first()
        )

        result.append({
            "id": p.id,
            "name": p.name,
            "gender": p.gender,
            "phone": p.phone,
            "hadm_id": p.hadm_id,
            "subject_id": diagnosis.subject_id if diagnosis else None
        })

    return result


from schemas import RegisterRequest, LoginRequest, TokenResponse
from models import User
from auth import hash_password, verify_password, create_access_token

@app.post("/api/register")
def register_user(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    user = User(
        username=req.username,
        password=hash_password(req.password),
        email=req.email,
        phone=req.phone,
        role=req.role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "Registration successful"}

@app.post("/api/login", response_model=TokenResponse)
def login_user(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.username, "role": user.role})
    return TokenResponse(token=token, role=user.role)