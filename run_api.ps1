# LandCover Change Detection API Runner
# PowerShell Script

Write-Host "🌍 LandCover Change Detection API" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python from https://python.org" -ForegroundColor Yellow
    Write-Host "Or add Python to your PATH environment variable" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
try {
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to continue"
        exit 1
    }
} catch {
    Write-Host "❌ Error installing dependencies: $_" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host ""

# Start API server
Write-Host "Starting API server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "🌐 API will be available at: http://localhost:8000" -ForegroundColor Green
Write-Host "📁 Upload interface: http://localhost:8000/static/upload.html" -ForegroundColor Green
Write-Host "📊 Dashboard: http://localhost:8000/static/dashboard.html" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

try {
    python app.py
} catch {
    Write-Host "❌ Error running API: $_" -ForegroundColor Red
}

Read-Host "Press Enter to continue" 