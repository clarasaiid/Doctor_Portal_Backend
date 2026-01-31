"""
FastAPI Backend for Doctor Portal OCR Service
Integrates DeepSeek-OCR-2 for Arabic and English lab test extraction
"""
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
from contextlib import asynccontextmanager
import os
import uuid
import aiofiles
from datetime import datetime
import logging
from dotenv import load_dotenv

from database import get_db, init_db, LabReport, User
from schemas import OCRRequest, OCRResponse, LabReportResponse, LabReportListResponse, ErrorResponse
from models.ocr_model import get_ocr_model

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("Initializing database...")
    init_db()
    
    logger.info("Loading OCR model...")
    try:
        # Pre-load the model
        get_ocr_model()
        logger.info("OCR model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load OCR model: {str(e)}")
        logger.warning("API will start but OCR endpoints may fail until model is available")
    
    yield
    
    # Shutdown (if needed)
    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Doctor Portal OCR API",
    description="OCR service for extracting Arabic and English text from lab tests using DeepSeek-OCR-2",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration - allow all origins for flexibility with tunnels
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for localtunnel/ngrok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Doctor Portal OCR API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    model_loaded = False
    try:
        model = get_ocr_model()
        model_loaded = model.model is not None
    except:
        pass
    
    return {
        "status": "ok",
        "database": "connected",
        "ocr_model": "loaded" if model_loaded else "not_loaded"
    }


@app.post("/api/v1/ocr/process", response_model=OCRResponse, tags=["OCR"])
async def process_ocr(
    file: UploadFile = File(..., description="Image file to process"),
    user_id: str = None,
    patient_id: Optional[str] = None,
    prompt: Optional[str] = "<image>\n<|grounding|>Convert the document to markdown.",
    db: Session = Depends(get_db)
):
    """
    Process an image file with DeepSeek-OCR-2 and store the results
    
    - **file**: Image file (JPG, PNG, etc.)
    - **user_id**: User ID who uploaded the image
    - **patient_id**: Optional patient ID to link the report
    - **prompt**: OCR prompt (default converts document to markdown)
    """
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Check file size
        file_size = 0
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"
            )
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1] or ".jpg"
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
        
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        
        # Ensure user exists
        if user_id:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                # Create user if doesn't exist
                user = User(id=user_id, name=f"User {user_id}", email=f"user{user_id}@example.com")
                db.add(user)
                db.commit()
        
        # Process with OCR
        ocr_model = get_ocr_model()
        ocr_result = ocr_model.process_image(
            image_path=file_path,
            prompt=prompt,
            base_size=1024,
            image_size=768,
            crop_mode=True
        )
        
        # Extract structured data
        structured_data = ocr_model.extract_lab_values(ocr_result["text"])
        
        # Determine status based on extracted values
        status_value = "normal"
        if structured_data.get("structured"):
            # Simple logic: check for high values
            if structured_data["structured"].get("HbA1c", 0) > 7.0:
                status_value = "abnormal"
            if structured_data["structured"].get("HbA1c", 0) > 9.0:
                status_value = "critical"
        
        # Create lab report record
        lab_report = LabReport(
            id=file_id,
            user_id=user_id or "anonymous",
            patient_id=patient_id,
            type="lab",
            title=f"Lab Report - {datetime.now().strftime('%Y-%m-%d')}",
            date=datetime.now(),
            status=status_value,
            raw_text=ocr_result["text"],
            structured_data=structured_data.get("structured", {}),
            confidence=ocr_result["confidence"],
            language_detected="+".join(ocr_result["languages"]),
            image_path=file_path,
            attachment_url=f"/api/v1/files/{file_id}{file_ext}"
        )
        
        db.add(lab_report)
        db.commit()
        db.refresh(lab_report)
        
        logger.info(f"Created lab report: {lab_report.id}")
        
        return OCRResponse(
            success=True,
            lab_report_id=lab_report.id,
            raw_text=ocr_result["text"],
            structured_data=structured_data.get("structured", {}),
            confidence=ocr_result["confidence"],
            languages=ocr_result["languages"],
            message="OCR processing completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing OCR: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )


@app.get("/api/v1/lab-reports", response_model=LabReportListResponse, tags=["Lab Reports"])
async def get_lab_reports(
    user_id: Optional[str] = None,
    patient_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get lab reports with optional filtering"""
    query = db.query(LabReport)
    
    if user_id:
        query = query.filter(LabReport.user_id == user_id)
    if patient_id:
        query = query.filter(LabReport.patient_id == patient_id)
    
    total = query.count()
    reports = query.order_by(LabReport.date.desc()).offset(offset).limit(limit).all()
    
    return LabReportListResponse(
        reports=[LabReportResponse.from_orm(r) for r in reports],
        total=total
    )


@app.get("/api/v1/lab-reports/{report_id}", response_model=LabReportResponse, tags=["Lab Reports"])
async def get_lab_report(
    report_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific lab report by ID"""
    report = db.query(LabReport).filter(LabReport.id == report_id).first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lab report not found"
        )
    
    return LabReportResponse.from_orm(report)


@app.get("/api/v1/files/{file_id}", tags=["Files"])
async def get_file(file_id: str):
    """Serve uploaded image files"""
    from fastapi.responses import FileResponse
    
    # Find file in uploads directory
    for ext in [".jpg", ".jpeg", ".png", ".pdf"]:
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
        if os.path.exists(file_path):
            return FileResponse(file_path)
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="File not found"
    )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
