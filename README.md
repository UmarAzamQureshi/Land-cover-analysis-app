# LandCover Change Detection API

An enterprise-grade machine learning pipeline for analyzing land cover changes using NDBI (Normalized Difference Built-up Index) time series data. This system provides both historical change detection and future change prediction capabilities.

## 🚀 Features

- **Multi-temporal Analysis**: Process time series of NDBI raster data
- **Machine Learning Pipeline**: Random Forest-based change detection
- **Future Predictions**: Trend extrapolation for future land cover changes
- **Interactive Visualizations**: Comprehensive reports and interactive maps
- **REST API**: FastAPI-based web service for easy integration
- **Cloud Ready**: Designed for deployment on Railway, Heroku, or similar platforms

## 🏗️ Architecture

```
landcover-railway/
├── app.py                 # FastAPI application entry point
├── requirements.txt       # Python dependencies
├── Procfile             # Railway deployment configuration
├── README.md            # This file
├── landcover_pipeline/  # Core pipeline modules
│   ├── __init__.py
│   ├── config.py        # Configuration management
│   ├── data_processor.py # Raster data processing
│   ├── feature_engineer.py # Temporal feature extraction
│   ├── model_trainer.py # ML model training
│   ├── predictor.py     # Change prediction
│   ├── visualizer.py    # Visualization and reporting
│   └── pipeline.py      # Main orchestration
└── static/              # Static output files
```

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd landcover-railway
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env with your configuration
   DATA_FOLDER=./data/rasters
   OUTPUT_DIR=./output
   QUICK_MODE=true
   QUICK_WINDOW_SIZE=600
   ```

5. **Run locally**
   ```bash
   python app.py
   ```

### Docker Deployment

1. **Build image**
   ```bash
   docker build -t landcover-api .
   ```

2. **Run container**
   ```bash
   docker run -p 8000:8000 -e DATA_FOLDER=/app/data landcover-api
   ```

## 🌐 API Endpoints

### Core Endpoints

- `GET /` - API information and status
- `GET /health` - Health check
- `GET /status` - Current pipeline status and configuration

### Pipeline Management

- `POST /run` - Start the land cover analysis pipeline
- `GET /outputs` - List all available output files
- `GET /outputs/{file_path}` - Download specific output file

### Data Management

- `POST /upload-rasters` - Upload raster files for analysis

## 📊 Usage

### 1. Prepare Your Data

Place your NDBI raster files in the data folder with filenames containing 4-digit years:

```
data/rasters/
├── ndbi_2015.tif
├── ndbi_2016.tif
├── ndbi_2017.tif
├── ndbi_2018.tif
└── ndbi_2019.tif
```

### 2. Run the Pipeline

**Via API:**
```bash
curl -X POST "http://localhost:8000/run"
```

**Via Python:**
```python
from landcover_pipeline.pipeline import LandCoverAnalysisPipeline
from landcover_pipeline.config import Config

config = Config.from_env()
pipeline = LandCoverAnalysisPipeline(config)
results = pipeline.run()
```

### 3. Access Results

The pipeline generates:
- **Prediction Maps**: GeoTIFF files showing land cover changes
- **Confidence Maps**: Pixel-level confidence scores
- **Comprehensive Report**: PNG visualization with all results
- **Interactive Map**: HTML file with Folium-based visualization
- **Executive Summary**: Markdown report with key findings

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_FOLDER` | `./data/rasters` | Directory containing raster files |
| `OUTPUT_DIR` | `./output` | Directory for output files |
| `MODEL_DIR` | `./models` | Directory for saved models |
| `LOGS_DIR` | `./logs` | Directory for log files |
| `QUICK_MODE` | `true` | Enable quick mode for faster processing |
| `QUICK_WINDOW_SIZE` | `600` | Pixel size for quick mode window |
| `FUTURE_YEARS` | `5` | Number of years to predict into future |
| `CONFIDENCE_THRESHOLD` | `0.55` | Minimum confidence for predictions |
| `RF_ESTIMATORS` | `120` | Number of Random Forest estimators |
| `TEST_SIZE` | `0.2` | Fraction of data for testing |
| `CV_FOLDS` | `3` | Cross-validation folds |

### Quick Mode

Enable quick mode for faster processing by cropping to a center window:
- Reduces processing time significantly
- Suitable for testing and development
- Maintains spatial relationships

## 🚀 Deployment

### Railway Deployment

1. **Connect to Railway**
   ```bash
   railway login
   railway init
   ```

2. **Set environment variables**
   ```bash
   railway variables set DATA_FOLDER=/app/data/rasters
   railway variables set OUTPUT_DIR=/app/output
   railway variables set QUICK_MODE=true
   ```

3. **Deploy**
   ```bash
   railway up
   ```

### Heroku Deployment

1. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

2. **Set environment variables**
   ```bash
   heroku config:set DATA_FOLDER=/app/data/rasters
   heroku config:set OUTPUT_DIR=/app/output
   heroku config:set QUICK_MODE=true
   ```

3. **Deploy**
   ```bash
   git push heroku main
   ```

### AWS/GCP Deployment

For production deployments, consider:
- **S3 Integration**: Store rasters and outputs in S3
- **Auto-scaling**: Handle variable workloads
- **Monitoring**: CloudWatch/Stackdriver integration
- **Security**: VPC, IAM roles, and encryption

## 📈 Performance

### Processing Times

| Data Size | Quick Mode | Full Mode |
|-----------|------------|-----------|
| 600x600 pixels | ~2-5 minutes | ~10-20 minutes |
| 1000x1000 pixels | ~5-10 minutes | ~20-40 minutes |
| 2000x2000 pixels | ~15-30 minutes | ~60-120 minutes |

### Memory Requirements

- **Minimum**: 2GB RAM
- **Recommended**: 4-8GB RAM
- **Large datasets**: 16GB+ RAM

## 🔧 Troubleshooting

### Common Issues

1. **Rasterio Import Error**
   ```bash
   # Install GDAL dependencies first
   conda install -c conda-forge gdal
   pip install rasterio
   ```

2. **Memory Issues**
   - Enable quick mode
   - Reduce raster resolution
   - Process smaller regions

3. **Slow Processing**
   - Check disk I/O performance
   - Use SSD storage
   - Optimize raster formats

### Logs

Check logs in the configured logs directory:
```bash
tail -f logs/land_cover_analysis_*.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Scientific Community**: For NDBI methodology and validation
- **Open Source**: Built on amazing open-source libraries
- **Research Partners**: For real-world testing and feedback

## 📞 Support

For technical support or questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting section

---

**Version**: 2.0.0  
**Last Updated**: August 2025  
**Maintainer**: LandCover Analysis Team 