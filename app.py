from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import time
from datetime import datetime
import threading

from landcover_pipeline.pipeline import LandCoverAnalysisPipeline
from landcover_pipeline.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LandCover Change Detection API",
    description="Enterprise-grade land cover change detection and prediction pipeline",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state for pipeline progress
pipeline_status = {
    "is_running": False,
    "progress": 0,
    "current_step": "",
    "start_time": None,
    "end_time": None,
    "error": None,
    "steps": [
        "Initializing pipeline",
        "Loading raster data",
        "Preprocessing data",
        "Feature extraction",
        "Model training",
        "Change detection analysis",
        "Generating predictions",
        "Saving results",
        "Pipeline completed"
    ]
}

def update_pipeline_progress(step: str, progress: int, error: str = None):
    """Update pipeline progress"""
    pipeline_status["current_step"] = step
    pipeline_status["progress"] = progress
    if error:
        pipeline_status["error"] = error
        pipeline_status["is_running"] = False
        pipeline_status["end_time"] = datetime.now().isoformat()

# Root endpoint - redirects to main interface
@app.get("/")
async def root():
    """Root endpoint - redirects to main dashboard"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/dashboard.html")

@app.get("/dashboard")
async def dashboard():
    """Direct dashboard access"""
    from fastapi.responses import FileResponse
    return FileResponse("static/dashboard.html")

@app.get("/upload")
async def upload():
    """Direct upload page access"""
    from fastapi.responses import FileResponse
    return FileResponse("static/upload.html")

# API Information
@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "status": "ok", 
        "message": "LandCover Change Detection API",
        "version": "2.0.0",
        "endpoints": {
            "run_pipeline": "/api/pipeline/run",
            "upload_rasters": "/api/upload",
            "list_outputs": "/api/outputs",
            "pipeline_status": "/api/pipeline/status",
            "model_parameters": "/api/parameters",
            "time_estimate": "/api/parameters/estimate",
            "health": "/api/health"
        },
        "web_interface": "/static/dashboard.html"
    }

# Health Check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

# Pipeline Management
@app.post("/api/pipeline/run")
async def run_pipeline(background_tasks: BackgroundTasks):
    """Start the land cover analysis pipeline in the background"""
    try:
        if pipeline_status["is_running"]:
            raise HTTPException(status_code=400, detail="Pipeline is already running")
        
        # Reset pipeline status
        pipeline_status["is_running"] = True
        pipeline_status["progress"] = 0
        pipeline_status["current_step"] = "Initializing"
        pipeline_status["start_time"] = datetime.now().isoformat()
        pipeline_status["end_time"] = None
        pipeline_status["error"] = None
        
        # Use environment configuration
        cfg = Config.from_env()
        
        # Create pipeline instance
        pipeline = LandCoverAnalysisPipeline(cfg)
        
        # Run in background
        background_tasks.add_task(run_pipeline_background, pipeline)
        
        logger.info("Pipeline started in background")
        
        return JSONResponse({
            "message": "Pipeline started successfully",
            "status": "running",
            "pipeline_id": int(time.time()),
            "config": {
                "data_folder": cfg.data_folder,
                "output_dir": cfg.output_dir,
                "quick_mode": cfg.quick_mode,
                "future_years": cfg.future_years
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to start pipeline: {str(e)}")
        pipeline_status["error"] = str(e)
        pipeline_status["is_running"] = False
        raise HTTPException(status_code=500, detail=f"Pipeline start failed: {str(e)}")

def run_pipeline_background(pipeline):
    """Run pipeline in background with progress updates"""
    try:
        # Simulate progress updates (replace with actual pipeline progress)
        for i, step in enumerate(pipeline_status["steps"]):
            if not pipeline_status["is_running"]:
                break
            update_pipeline_progress(step, int((i + 1) * 100 / len(pipeline_status["steps"])))
            import time
            time.sleep(2)  # Simulate processing time
        
        # Run actual pipeline
        pipeline.run_and_save()
        
        # Mark as completed
        update_pipeline_progress("Pipeline completed", 100)
        pipeline_status["is_running"] = False
        pipeline_status["end_time"] = datetime.now().isoformat()
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        update_pipeline_progress("Error occurred", 0, str(e))

@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Get current pipeline status and progress"""
    try:
        cfg = Config.from_env()
        
        # Check if outputs exist
        output_dir = Path(cfg.output_dir)
        has_outputs = output_dir.exists() and any(output_dir.iterdir())
        
        # Check if data exists
        data_dir = Path(cfg.data_folder)
        has_data = data_dir.exists() and any(data_dir.glob("*.tif*"))
        
        # Get output files for dashboard
        output_files = []
        if has_outputs:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(output_dir))
                    file_size = file_path.stat().st_size
                    output_files.append({
                        "path": rel_path,
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "type": get_file_type(rel_path),
                        "download_url": f"/api/outputs/download/{rel_path}"
                    })
        
        return {
            "pipeline": {
                "is_running": pipeline_status["is_running"],
                "progress": pipeline_status["progress"],
                "current_step": pipeline_status["current_step"],
                "start_time": pipeline_status["start_time"],
                "end_time": pipeline_status["end_time"],
                "error": pipeline_status["error"],
                "steps": pipeline_status["steps"]
            },
            "system": {
                "status": "ready" if has_data else "no_data",
                "has_data": has_data,
                "has_outputs": has_outputs,
                "data_files_count": len(list(data_dir.glob("*.tif*"))) if has_data else 0,
                "output_files_count": len(output_files)
            },
            "outputs": output_files,
            "configuration": {
                "data_folder": cfg.data_folder,
                "output_dir": cfg.output_dir,
                "quick_mode": cfg.quick_mode,
                "quick_window_size": cfg.quick_window_size,
                "future_years": cfg.future_years,
                "confidence_threshold": cfg.confidence_threshold
            }
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return {"status": "error", "error": str(e)}

def get_file_type(file_path: str) -> str:
    """Determine file type based on extension"""
    ext = Path(file_path).suffix.lower()
    if ext in ['.tif', '.tiff']:
        return 'raster'
    elif ext in ['.png', '.jpg', '.jpeg']:
        return 'image'
    elif ext in ['.csv', '.json']:
        return 'data'
    elif ext in ['.pdf']:
        return 'report'
    else:
        return 'other'

# File Upload
@app.post("/api/upload")
async def upload_rasters(files: List[UploadFile] = File(...)):
    """Upload raster files for analysis"""
    try:
        data_dir = Path(os.environ.get("DATA_FOLDER", "./data/rasters"))
        data_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        total_size = 0
        
        for file in files:
            if not file.filename.lower().endswith(('.tif', '.tiff')):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a valid raster file. Only .tif/.tiff files are supported."
                )
            
            dest_path = data_dir / file.filename
            
            # Save file
            content = await file.read()
            with open(dest_path, "wb") as out:
                out.write(content)
            
            saved_files.append({
                "filename": file.filename,
                "path": str(dest_path),
                "size_bytes": len(content),
                "size_mb": round(len(content) / (1024 * 1024), 2)
            })
            total_size += len(content)
            
            logger.info(f"Saved raster: {file.filename} ({len(content)} bytes)")
        
        return {
            "message": f"Successfully uploaded {len(files)} raster files",
            "files": saved_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "data_directory": str(data_dir)
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Output Management
@app.get("/api/outputs")
async def list_outputs():
    """List all available output files"""
    try:
        output_dir = Path(os.environ.get("OUTPUT_DIR", "./output"))
        
        if not output_dir.exists():
            return {"outputs": [], "message": "No outputs directory found"}
        
        files = []
        total_size = 0
        
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(output_dir))
                file_size = file_path.stat().st_size
                files.append({
                    "path": rel_path,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "type": get_file_type(rel_path),
                    "download_url": f"/api/outputs/download/{rel_path}"
                })
                total_size += file_size
        
        return {
            "outputs": files,
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "output_directory": str(output_dir)
        }
        
    except Exception as e:
        logger.error(f"Failed to list outputs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list outputs: {str(e)}")

@app.get("/api/outputs/download/{file_path:path}")
async def download_output(file_path: str):
    """Download a specific output file"""
    try:
        output_dir = Path(os.environ.get("OUTPUT_DIR", "./output"))
        file_full_path = output_dir / file_path
        
        if not file_full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not file_full_path.is_file():
            raise HTTPException(status_code=400, detail="Not a file")
        
        return FileResponse(
            path=str(file_full_path),
            filename=file_path.split("/")[-1],
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

# Legacy endpoints for backward compatibility (deprecated)
@app.get("/status")
async def legacy_status():
    """Legacy status endpoint - redirects to new API"""
    return await get_pipeline_status()

@app.get("/outputs")
async def legacy_outputs():
    """Legacy outputs endpoint - redirects to new API"""
    return await list_outputs()

@app.post("/run")
async def legacy_run():
    """Legacy run endpoint - redirects to new API"""
    return await run_pipeline(BackgroundTasks())

@app.post("/upload-rasters")
async def legacy_upload(files: List[UploadFile] = File(...)):
    """Legacy upload endpoint - redirects to new API"""
    return await upload_rasters(files)

# Add these imports at the top if not already present
from pydantic import BaseModel
from typing import Optional

# Add these new API endpoints after your existing endpoints

# Model Parameter Management
@app.get("/api/parameters")
async def get_model_parameters():
    """Get current model parameters"""
    try:
        cfg = Config.from_env()
        return {
            "runtime": {
                "quick_mode": cfg.quick_mode,
                "quick_window_size": cfg.quick_window_size
            },
            "model": {
                "rf_estimators": cfg.rf_estimators,
                "rf_max_depth": cfg.rf_max_depth,
                "test_size": cfg.test_size,
                "random_state": cfg.random_state,
                "cv_folds": cfg.cv_folds
            },
            "feature_engineering": {
                "ndbi_threshold": cfg.ndbi_threshold,
                "max_samples_per_class": cfg.max_samples_per_class,
                "feature_importance_threshold": cfg.feature_importance_threshold
            },
            "prediction": {
                "future_years": cfg.future_years,
                "confidence_threshold": cfg.confidence_threshold
            },
            "visualization": {
                "figure_dpi": cfg.figure_dpi,
                "map_zoom": cfg.map_zoom
            }
        }
    except Exception as e:
        logger.error(f"Failed to get parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get parameters: {str(e)}")

@app.post("/api/parameters")
async def update_model_parameters(parameters: dict):
    """Update model parameters"""
    try:
        # Validate parameters
        validated_params = {}
        
        # Runtime settings
        if "runtime" in parameters:
            runtime = parameters["runtime"]
            if "quick_mode" in runtime:
                validated_params["QUICK_MODE"] = str(runtime["quick_mode"]).lower()
            if "quick_window_size" in runtime:
                validated_params["QUICK_WINDOW_SIZE"] = str(runtime["quick_window_size"])
        
        # Model parameters
        if "model" in parameters:
            model = parameters["model"]
            if "rf_estimators" in model:
                validated_params["RF_ESTIMATORS"] = str(model["rf_estimators"])
            if "rf_max_depth" in model:
                validated_params["RF_MAX_DEPTH"] = str(model["rf_max_depth"]) if model["rf_max_depth"] else ""
            if "test_size" in model:
                validated_params["TEST_SIZE"] = str(model["test_size"])
            if "random_state" in model:
                validated_params["RANDOM_STATE"] = str(model["random_state"])
            if "cv_folds" in model:
                validated_params["CV_FOLDS"] = str(model["cv_folds"])
        
        # Feature engineering
        if "feature_engineering" in parameters:
            fe = parameters["feature_engineering"]
            if "ndbi_threshold" in fe:
                validated_params["NDBI_THRESHOLD"] = str(fe["ndbi_threshold"])
            if "max_samples_per_class" in fe:
                validated_params["MAX_SAMPLES_PER_CLASS"] = str(fe["max_samples_per_class"])
            if "feature_importance_threshold" in fe:
                validated_params["FEATURE_IMPORTANCE_THRESHOLD"] = str(fe["feature_importance_threshold"])
        
        # Prediction parameters
        if "prediction" in parameters:
            pred = parameters["prediction"]
            if "future_years" in pred:
                validated_params["FUTURE_YEARS"] = str(pred["future_years"])
            if "confidence_threshold" in pred:
                validated_params["CONFIDENCE_THRESHOLD"] = str(pred["confidence_threshold"])
        
        # Visualization
        if "visualization" in parameters:
            viz = parameters["visualization"]
            if "figure_dpi" in viz:
                validated_params["FIGURE_DPI"] = str(viz["figure_dpi"])
            if "map_zoom" in viz:
                validated_params["MAP_ZOOM"] = str(viz["map_zoom"])
        
        # Update environment variables
        for key, value in validated_params.items():
            os.environ[key] = value
        
        logger.info(f"Updated model parameters: {validated_params}")
        
        return {
            "message": "Parameters updated successfully",
            "updated_parameters": validated_params
        }
        
    except Exception as e:
        logger.error(f"Failed to update parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update parameters: {str(e)}")

@app.get("/api/parameters/estimate")
async def estimate_processing_time():
    """Estimate processing time based on current parameters"""
    try:
        cfg = Config.from_env()
        
        # Get data file count
        data_dir = Path(cfg.data_folder)
        data_files = list(data_dir.glob("*.tif*")) if data_dir.exists() else []
        num_files = len(data_files)
        
        # Base time estimates (in seconds)
        base_times = {
            "data_loading": 2 * num_files,
            "preprocessing": 5 * num_files,
            "feature_extraction": 10 * num_files,
            "model_training": 30 + (cfg.rf_estimators * 0.5),
            "cross_validation": cfg.cv_folds * 15,
            "prediction": 5 * cfg.future_years,
            "visualization": 10,
            "saving_results": 5
        }
        
        # Adjust for quick mode
        if cfg.quick_mode:
            for key in base_times:
                if key != "model_training":
                    base_times[key] *= 0.3
        
        # Calculate total time
        total_seconds = sum(base_times.values())
        total_minutes = total_seconds / 60
        
        # Format time estimate
        if total_minutes < 1:
            time_estimate = f"{int(total_seconds)} seconds"
        elif total_minutes < 60:
            time_estimate = f"{total_minutes:.1f} minutes"
        else:
            hours = total_minutes / 60
            time_estimate = f"{hours:.1f} hours"
        
        return {
            "time_estimate": time_estimate,
            "total_seconds": total_seconds,
            "breakdown": base_times,
            "parameters": {
                "num_files": num_files,
                "quick_mode": cfg.quick_mode,
                "rf_estimators": cfg.rf_estimators,
                "cv_folds": cfg.cv_folds,
                "future_years": cfg.future_years
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to estimate processing time: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to estimate processing time: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Get port from environment variable (Railway sets this) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 