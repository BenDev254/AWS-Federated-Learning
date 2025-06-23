from pydantic import BaseModel, Field, conint
from typing import List
from typing import Optional
from datetime import datetime
from pydantic import validator

class DiagnosisEntry(BaseModel):
    diagnosis_count: conint(ge=0)  # must be 0 or more
    risk_level: conint(ge=0, le=2)  # must be 0, 1, or 2

class DiagnosisBatchInput(BaseModel):
    diagnoses: List[DiagnosisEntry]



class DiagnosisIn(BaseModel):
    subject_id: Optional[str] = None
    hadm_id: Optional[str]
    icd_code: Optional[str]
    icd_version: Optional[int] = 10
    seq_num: Optional[int] = 1
    diagnosis_count: Optional[int] = 0
    risk_level: Optional[int] = 0
    created_at: Optional[datetime]

    @validator("subject_id", pre=True, always=True)
    def empty_str_to_none(cls, v):
        return v or None



class PatientCreate(BaseModel):
    name: str
    gender: str
    phone: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str
    phone: str
    role: str  

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    token: str
    role: str
