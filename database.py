from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./doctor_portal.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    role = Column(String, default="doctor")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    lab_reports = relationship("LabReport", back_populates="user")


class LabReport(Base):
    __tablename__ = "lab_reports"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    patient_id = Column(String, nullable=True, index=True)  # Optional: link to patient if available
    type = Column(String, default="lab")  # lab, imaging, document
    title = Column(String, nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="normal")  # normal, abnormal, critical
    
    # OCR Results
    raw_text = Column(Text, nullable=True)  # Full extracted text
    structured_data = Column(JSON, nullable=True)  # Parsed structured data
    confidence = Column(Float, default=0.0)
    
    # File storage
    image_path = Column(String, nullable=True)  # Path to uploaded image
    attachment_url = Column(String, nullable=True)
    
    # Metadata
    language_detected = Column(String, nullable=True)  # ar, en, ar+en
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="lab_reports")


class LabValue(Base):
    __tablename__ = "lab_values"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    lab_report_id = Column(String, ForeignKey("lab_reports.id"), nullable=False, index=True)
    patient_id = Column(String, nullable=True, index=True)
    metric = Column(String, nullable=False)  # e.g., "HbA1c", "Glucose"
    value = Column(Float, nullable=False)
    unit = Column(String, nullable=True)  # e.g., "mg/dL", "%"
    date = Column(DateTime, default=datetime.utcnow)
    
    lab_report = relationship("LabReport")


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
