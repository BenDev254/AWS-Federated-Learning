from sqlalchemy import Column, Integer, String,  ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Envoy(Base):
    __tablename__ = "envoys"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    training_host = Column(String, nullable=True) 
    s3_output_prefix = Column(String)
    s3_bucket = Column(String)
    s3_prefix = Column(String)
    region = Column(String)
    serial_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))

    # Relationships
    diagnoses = relationship("LocalDiagnosis", back_populates="envoy", cascade="all, delete")

class LocalDiagnosis(Base):
    __tablename__ = "local_diagnoses"

    id = Column(Integer, primary_key=True)
    envoy_id = Column(Integer, ForeignKey("envoys.id"))
    subject_id = Column(Integer, nullable=True) 
    hadm_id = Column(Integer, nullable=True)    
    icd_code = Column(String, nullable=True)
    icd_version = Column(Integer, default=10)
    seq_num = Column(Integer, default=1)
    diagnosis_count = Column(Integer)  
    risk_level = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    #Relationships 
    envoy = relationship("Envoy", back_populates="diagnoses")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    gender = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    hadm_id = Column(Integer, unique=True) 



class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String) 
    email = Column(String, unique=True)
    phone = Column(String)
    role = Column(String) 



