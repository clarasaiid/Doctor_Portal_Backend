# Doctor Portal OCR Backend

Backend service for processing lab test images using DeepSeek-OCR-2 for Arabic and English text extraction.

## Features

- ✅ DeepSeek-OCR-2 integration for Arabic and English OCR
- ✅ FastAPI REST API
- ✅ SQLite database for storing lab reports
- ✅ User and patient management
- ✅ Structured data extraction from lab reports
- ✅ CORS enabled for frontend integration

## Setup

### 1. Prerequisites

- Python 3.12.9+
- CUDA 11.8+ (for GPU acceleration, optional)
- PyTorch 2.6.0+

### 2. Install Dependencies

```bash
# Create virtual environment
conda create -n deepseek-ocr2 python=3.12.9 -y
conda activate deepseek-ocr2

# Install PyTorch with CUDA support (if you have GPU)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install flash-attn (for GPU acceleration)
pip install flash-attn==2.7.3 --no-build-isolation
```

### 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
# Important: Set CUDA_VISIBLE_DEVICES if using GPU
```

### 4. Initialize Database

The database will be automatically created on first run. You can also initialize it manually:

```python
from database import init_db
init_db()
```

### 5. Run the Server

```bash
# Development mode
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health check with model status

### OCR Processing
- `POST /api/v1/ocr/process` - Process an image file
  - Body: multipart/form-data with `file` (image), `user_id`, `patient_id` (optional), `prompt` (optional)
  - Returns: OCR results with extracted text and structured data

### Lab Reports
- `GET /api/v1/lab-reports` - List lab reports (supports `user_id`, `patient_id`, `limit`, `offset` query params)
- `GET /api/v1/lab-reports/{report_id}` - Get specific lab report

### Files
- `GET /api/v1/files/{file_id}` - Serve uploaded image files

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage Example

```bash
# Process an image
curl -X POST "http://localhost:8000/api/v1/ocr/process" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@lab_test.jpg" \
  -F "user_id=d1" \
  -F "patient_id=p1"

# Get lab reports for a user
curl "http://localhost:8000/api/v1/lab-reports?user_id=d1"
```

## Notes

- The DeepSeek-OCR-2 model will be downloaded from Hugging Face on first run (can be several GB)
- GPU is recommended for faster processing but CPU will work
- Arabic and English text extraction is supported
- The model supports dynamic resolution and various document formats

## Troubleshooting

1. **Model download fails**: Check internet connection and Hugging Face access
2. **CUDA errors**: Make sure CUDA 11.8+ is installed and compatible with your GPU
3. **Out of memory**: Reduce `image_size` parameter or use CPU mode
4. **Import errors**: Make sure all dependencies are installed, especially `flash-attn` for GPU
