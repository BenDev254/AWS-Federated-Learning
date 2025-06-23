# models.py
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class LocalDiagnosis(Base):
    __tablename__ = "local_diagnoses"

    id = Column(Integer, primary_key=True, index=True)
    envoy_id = Column(Integer, ForeignKey("envoys.id"))
    diagnosis_count = Column(Integer)
    risk_level = Column(Integer)  # 0=Low, 1=Medium, 2=High

    envoy = relationship("Envoy", back_populates="diagnoses")

# Add to Envoy model:
Envoy.diagnoses = relationship("LocalDiagnosis", back_populates="envoy", cascade="all, delete")
