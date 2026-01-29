# Quick Start Guide

## Backend Quick Start

```bash
# 1. Navigate to backend
cd backend

# 2. Create and activate conda environment
conda create -n deepseek-ocr2 python=3.12.9 -y
conda activate deepseek-ocr2

# 3. Install PyTorch (GPU)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install flash-attn (GPU only, optional)
pip install flash-attn==2.7.3 --no-build-isolation

# 6. Setup environment
cp env.example .env
# Edit .env if needed

# 7. Start server
python main.py
```

Backend will be at: http://localhost:8000

## Frontend Quick Start

```bash
# 1. Navigate to project
cd project

# 2. Install dependencies
npm install

# 3. Start Expo
npm start
```

Then press `w` for web or use Expo Go app.

## Test It Works

1. Backend health: `curl http://localhost:8000/health`
2. Open frontend → AI Assistant tab → Upload lab test image → Run Extraction

## Important Notes

- First backend run downloads DeepSeek-OCR-2 model (~10GB) - takes 10-30 minutes
- Backend must be running before using OCR in frontend
- Model supports Arabic and English text extraction
- Lab reports are stored in SQLite database (`doctor_portal.db`)
