#!/bin/bash

echo "🚀 Starting Live Detection Web App..."
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt -q
    pip install -r flask_requirements.txt -q
fi

# Create folders
mkdir -p uploads outputs

# Run app
echo ""
echo "========================================================================"
echo "🎯 LIVE DETECTION WEB APP"
echo "========================================================================"
echo "📺 Opening at: http://localhost:5000"
echo "Press Ctrl+C to stop"
echo "========================================================================"
echo ""

python3 app.py
