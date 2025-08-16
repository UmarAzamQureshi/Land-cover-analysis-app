@echo off
echo 🌍 LandCover Change Detection API
echo =================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found in PATH
    echo.
    echo Please install Python from https://python.org
    echo Or add Python to your PATH environment variable
    echo.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

echo Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    echo.
    pause
    exit /b 1
)

echo ✅ Dependencies installed
echo.

echo Starting API server...
echo.
echo 🌐 API will be available at: http://localhost:8000
echo 📁 Upload interface: http://localhost:8000/static/upload.html
echo 📊 Dashboard: http://localhost:8000/static/dashboard.html
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause 