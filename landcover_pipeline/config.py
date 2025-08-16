import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

@dataclass
class Config:
    """Configuration class for the land cover change detection system"""
    
    # Paths
    data_folder: str = os.environ.get("DATA_FOLDER", "./data/rasters")
    output_dir: str = os.environ.get("OUTPUT_DIR", "./output")
    model_dir: str = os.environ.get("MODEL_DIR", "./models")
    logs_dir: str = os.environ.get("LOGS_DIR", "./logs")
    
    # Runtime settings
    quick_mode: bool = os.environ.get("QUICK_MODE", "true").lower() == "true"
    quick_window_size: int = int(os.environ.get("QUICK_WINDOW_SIZE", "600"))
    
    # Model parameters
    rf_estimators: int = int(os.environ.get("RF_ESTIMATORS", "120"))
    rf_max_depth: int = int(os.environ.get("RF_MAX_DEPTH", "20")) if os.environ.get("RF_MAX_DEPTH") else None
    test_size: float = float(os.environ.get("TEST_SIZE", "0.2"))
    random_state: int = int(os.environ.get("RANDOM_STATE", "42"))
    cv_folds: int = int(os.environ.get("CV_FOLDS", "3"))
    
    # Feature engineering
    ndbi_threshold: float = float(os.environ.get("NDBI_THRESHOLD", "0.05"))
    max_samples_per_class: int = int(os.environ.get("MAX_SAMPLES_PER_CLASS", "5000"))
    feature_importance_threshold: float = float(os.environ.get("FEATURE_IMPORTANCE_THRESHOLD", "0.01"))
    
    # Prediction parameters
    future_years: int = int(os.environ.get("FUTURE_YEARS", "5"))
    confidence_threshold: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.55"))
    
    # Visualization
    figure_dpi: int = int(os.environ.get("FIGURE_DPI", "300"))
    map_zoom: int = int(os.environ.get("MAP_ZOOM", "11"))
    
    # Model hyperparameters for tuning
    rf_param_grid: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': [100, 120],
        'max_depth': [15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    })
    
    # S3 configuration (optional)
    s3_bucket: str = os.environ.get("S3_BUCKET", "")
    s3_region: str = os.environ.get("S3_REGION", "us-east-1")
    use_s3: bool = os.environ.get("USE_S3", "false").lower() == "true"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for folder in [self.output_dir, self.model_dir, self.logs_dir]:
            Path(folder).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls):
        """Create configuration from environment variables"""
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging"""
        return {
            'data_folder': self.data_folder,
            'output_dir': self.output_dir,
            'quick_mode': self.quick_mode,
            'quick_window_size': self.quick_window_size,
            'future_years': self.future_years,
            'confidence_threshold': self.confidence_threshold,
            'rf_estimators': self.rf_estimators,
            'use_s3': self.use_s3
        } 