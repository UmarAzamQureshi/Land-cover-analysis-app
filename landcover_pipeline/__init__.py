"""
LandCover Change Detection Pipeline

A comprehensive machine learning pipeline for analyzing land cover changes
using NDBI (Normalized Difference Built-up Index) time series data.
"""

__version__ = "2.0.0"
__author__ = "LandCover Analysis Team"

from .config import Config
from .data_processor import DataProcessor
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .predictor import Predictor
from .visualizer import Visualizer
from .pipeline import LandCoverAnalysisPipeline

__all__ = [
    "Config",
    "DataProcessor", 
    "FeatureEngineer",
    "ModelTrainer",
    "Predictor",
    "Visualizer",
    "LandCoverAnalysisPipeline"
] 