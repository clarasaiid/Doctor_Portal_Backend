#!/bin/bash

# Start script for Doctor Portal OCR Backend

echo "Starting Doctor Portal OCR Backend..."

# Activate conda environment if it exists
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate deepseek-ocr2 2>/dev/null || echo "Conda environment 'deepseek-ocr2' not found. Using system Python."
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Copying from env.example..."
    cp env.example .env
    echo "Please edit .env with your configuration before running again."
    exit 1
fi

# Create uploads directory
mkdir -p uploads

# Run the server
echo "Starting FastAPI server on http://localhost:8000"
python main.py
