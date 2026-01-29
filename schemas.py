"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class OCRRequest(BaseModel):
    """Request schema for OCR processing"""
    user_id: str = Field(..., description="User ID who uploaded the image")
    patient_id: Optional[str] = Field(None, description="Optional patient ID if linking to a patient")
    prompt: Optional[str] = Field(
        "<image>\n<|grounding|>Convert the document to markdown.",
        description="OCR prompt (default converts to markdown)"
    )


class OCRResponse(BaseModel):
    """Response schema for OCR processing"""
    success: bool
    lab_report_id: str
    raw_text: str
    structured_data: Dict[str, Any]
    confidence: float
    languages: List[str]
    message: Optional[str] = None


class LabReportResponse(BaseModel):
    """Response schema for lab report"""
    id: str
    user_id: str
    patient_id: Optional[str]
    type: str
    title: str
    date: datetime
    status: str
    raw_text: Optional[str]
    structured_data: Optional[Dict[str, Any]]
    confidence: float
    language_detected: Optional[str]
    attachment_url: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class LabReportListResponse(BaseModel):
    """Response schema for list of lab reports"""
    reports: List[LabReportResponse]
    total: int


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str
    detail: Optional[str] = None
