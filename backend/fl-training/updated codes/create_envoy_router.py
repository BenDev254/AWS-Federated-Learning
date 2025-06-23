from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
import os
import json
import shutil
from models import Envoy, get_db  # Ensure models.py defines Envoy and get_db

app = FastAPI()

CONFIGS_DIR = "configs"
TEMPLATE_DIR = "templates"
TRAIN_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "train.py")
REQUIREMENTS_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "requirements.txt")


@app.post("/create_envoy")
def create_envoy(name: str, s3_bucket: str, s3_prefix: str, region: str = "eu-north-1", db: Session = Depends(get_db)):
    # Check for existing envoy
    if db.query(Envoy).filter(Envoy.name == name).first():
        raise HTTPException(status_code=400, detail="Envoy already exists")

    # Save new envoy to DB
    envoy = Envoy(name=name, s3_bucket=s3_bucket, s3_prefix=s3_prefix, region=region)
    db.add(envoy)
    db.commit()

    # Create envoy-specific config
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    config = {
        "dataset_s3": "s3://physionet-open/mimic-iv-demo/2.2/hosp/diagnoses_icd.csv.gz",
        "region": region,
        "s3_output_prefix": f"s3://{s3_bucket}/{s3_prefix}",
        "use_mimic": True
    }
    config_path = os.path.join(CONFIGS_DIR, f"{name}_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Create envoy-specific directory and package training artifacts
    envoy_dir = os.path.join("envoys", name)
    os.makedirs(envoy_dir, exist_ok=True)
    shutil.copyfile(TRAIN_TEMPLATE_PATH, os.path.join(envoy_dir, "train.py"))
    shutil.copyfile(REQUIREMENTS_TEMPLATE_PATH, os.path.join(envoy_dir, "requirements.txt"))
    shutil.copyfile(config_path, os.path.join(envoy_dir, "config.json"))

    return {"message": f"Envoy '{name}' created and training package prepared."}
