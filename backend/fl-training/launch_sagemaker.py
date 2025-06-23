import sagemaker
from sagemaker.pytorch import PyTorch

# Setup
session = sagemaker.Session()
role = "arn:aws:iam::853869586998:role/service-role/AmazonSageMaker-ExecutionRole-20250618T101261"
bucket = session.default_bucket()

# Launch training
estimator = PyTorch(
    entry_point="train.py",
    source_dir=".",  # current folder
    role=role,
    framework_version="1.13.1",
    py_version="py39",
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{bucket}/fl-training/output",
    base_job_name="fl-training-job"
)

estimator.fit({
    "training": "s3://physionet-open/mimic-iv-demo/2.2/"  # This includes hosp/diagnoses_icd.csv.gz
})

