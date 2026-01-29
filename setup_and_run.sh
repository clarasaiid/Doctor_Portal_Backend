#!/bin/bash

# Setup and run script for Doctor Portal OCR Backend

set -e

echo "🚀 Setting up Doctor Portal OCR Backend..."
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from env.example..."
    cp env.example .env
    echo "⚠️  Please edit .env if you need to change settings (defaults should work)"
fi

# Create uploads directory
mkdir -p uploads

# Install dependencies
echo "📥 Installing dependencies..."
echo "   This may take a few minutes..."

# Install PyTorch first (CPU version for simplicity)
echo "   Installing PyTorch..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Install other requirements
echo "   Installing other packages..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "⚠️  IMPORTANT: On first run, DeepSeek-OCR-2 model will be downloaded (~10GB)"
echo "   This may take 10-30 minutes depending on your internet speed."
echo ""
echo "🚀 Starting backend server..."
echo "   Backend will be available at: http://localhost:8000"
echo "   API docs will be at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server
python main.py
