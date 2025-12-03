@echo off
echo 🚀 Starting Live Detection Web App...
echo.

REM Check if venv exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate venv
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt -q
pip install -r flask_requirements.txt -q

REM Create folders
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs

REM Run app
echo.
echo ========================================================================
echo 🎯 LIVE DETECTION WEB APP
echo ========================================================================
echo 📺 Opening at: http://localhost:5000
echo Press Ctrl+C to stop
echo ========================================================================
echo.

python app.py

pause
