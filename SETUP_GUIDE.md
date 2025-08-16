# 🚀 Local Setup Guide - LandCover Change Detection API

This guide will help you set up and test the LandCover Change Detection API locally on your machine.

## 📋 Prerequisites

- **Python 3.8+** installed on your system
- **Git** for cloning the repository
- **Web browser** for testing the web interface

## 🛠️ Installation Steps

### 1. **Clone or Download the Project**

If you have Git:
```bash
git clone <your-repository-url>
cd landcover-railway
```

Or simply download and extract the project files to a folder named `landcover-railway`.

### 2. **Verify Project Structure**

Your project should look like this:
```
landcover-railway/
├── app.py                    # FastAPI application
├── requirements.txt          # Python dependencies
├── run_api.bat              # Windows batch file
├── run_api.ps1              # PowerShell script
├── create_sample_data.py    # Sample data generator
├── test_simple.py           # Structure test
├── landcover_pipeline/      # Core modules
└── static/                  # Web interfaces
```

### 3. **Install Python Dependencies**

**Option A: Using pip directly**
```bash
pip install -r requirements.txt
```

**Option B: Using Python module**
```bash
python -m pip install -r requirements.txt
```

**Option C: Windows batch file (easiest)**
```bash
# Double-click run_api.bat or run in Command Prompt
run_api.bat
```

**Option D: PowerShell script**
```powershell
# Right-click run_api.ps1 and "Run with PowerShell"
# Or run in PowerShell:
.\run_api.ps1
```

### 4. **Create Sample Data (Optional)**

To test the system without your own data:
```bash
python create_sample_data.py
```

This creates 5 sample NDBI raster files (2015-2019) in the `data/rasters/` directory.

### 5. **Run the API**

**Option A: Direct Python command**
```bash
python app.py
```

**Option B: Using the batch file (Windows)**
```bash
run_api.bat
```

**Option C: Using PowerShell script**
```powershell
.\run_api.ps1
```

## 🌐 Testing the System

### 1. **API Endpoints**

Once running, the API will be available at:
- **Main API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **Status**: http://localhost:8000/status

### 2. **Web Interface**

Open your browser and navigate to:
- **File Upload**: http://localhost:8000/static/upload.html
- **Dashboard**: http://localhost:8000/static/dashboard.html

### 3. **Test the Complete Workflow**

1. **Upload Raster Files**:
   - Go to the upload interface
   - Drag and drop your NDBI raster files
   - Files should contain years in their names (e.g., `ndbi_2015.tif`)

2. **Run Analysis**:
   - Go to the dashboard
   - Click "Run Analysis Pipeline"
   - Monitor the progress and logs

3. **View Results**:
   - Check the outputs section
   - Download generated files
   - View the comprehensive report

## 🔧 Troubleshooting

### Common Issues

#### **Python Not Found**
```
'python' is not recognized as an internal or external command
```

**Solutions**:
- Install Python from [python.org](https://python.org)
- Add Python to your PATH environment variable
- Use the full path to Python executable

#### **Dependencies Installation Failed**
```
ERROR: Could not find a version that satisfies the requirement rasterio
```

**Solutions**:
- Update pip: `python -m pip install --upgrade pip`
- Install GDAL dependencies first (on Windows, use conda)
- Try: `conda install -c conda-forge gdal rasterio`

#### **Port Already in Use**
```
Address already in use: ('0.0.0.0', 8000)
```

**Solutions**:
- Change the port in `app.py` (line 224)
- Kill the process using port 8000
- Use a different port

#### **Permission Errors**
```
Permission denied: 'data/rasters'
```

**Solutions**:
- Run as administrator (Windows)
- Check folder permissions
- Create directories manually

### **Testing Without Dependencies**

If you can't install the full dependencies, test the structure:
```bash
python test_simple.py
```

This will verify your project setup without requiring external packages.

## 📊 Sample Data Format

Your NDBI raster files should:
- **Format**: GeoTIFF (.tif or .tiff)
- **Naming**: Include 4-digit years (e.g., `ndbi_2015.tif`)
- **Content**: Single-band NDBI values (typically -1 to 1)
- **Size**: Any reasonable dimensions (100x100 to 2000x2000 pixels)

## 🎯 Quick Test Workflow

1. **Start the API**: `python app.py`
2. **Open Browser**: http://localhost:8000/static/dashboard.html
3. **Check Status**: Should show "No data available"
4. **Upload Files**: Use the upload interface
5. **Run Pipeline**: Click "Run Analysis Pipeline"
6. **View Results**: Check outputs and download files

## 📞 Getting Help

If you encounter issues:

1. **Check the logs** in the terminal where you ran the API
2. **Verify Python version**: `python --version`
3. **Test basic imports**: `python -c "import numpy; print('OK')"`
4. **Check file permissions** and directory structure
5. **Review error messages** for specific dependency issues

## 🚀 Next Steps

Once the local setup is working:

1. **Customize Configuration**: Modify environment variables
2. **Add Your Data**: Replace sample data with real NDBI rasters
3. **Deploy to Railway**: Push to GitHub and deploy
4. **Scale Up**: Add more features and optimizations

---

**Happy Testing! 🌍📊**

If you need help, check the main README.md or create an issue in the repository. 