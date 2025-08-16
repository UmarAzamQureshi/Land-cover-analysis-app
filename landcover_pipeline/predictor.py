"""
Prediction module for land cover change detection
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import rasterio

from .feature_engineer import FeatureEngineer

class Predictor:
    """Enterprise prediction and visualization system"""

    def __init__(self, config, model_artifacts: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Load model artifacts
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        self.feature_names = model_artifacts['feature_names']
        self.training_history = model_artifacts['training_history']

    def predict_historical_changes(self, time_series: np.ndarray, years: List[int],
                                 metadata: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict historical land cover changes with confidence scores

        Args:
            time_series: Shape (T, H, W) time series data
            years: List of years
            metadata: Raster metadata

        Returns:
            Tuple[np.ndarray, np.ndarray]: (predictions, confidence_scores)
        """
        self.logger.info("🔮 Generating historical change predictions...")

        T, H, W = time_series.shape
        pixels = H * W

        # Extract features using same method as training
        feature_engineer = FeatureEngineer(self.config)
        X, _ = feature_engineer.extract_temporal_features(time_series, years)

        # Handle invalid pixels
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]

        # Scale features
        X_scaled = self.scaler.transform(X_valid)

        # Predictions and confidence scores
        predictions_valid = self.model.predict(X_scaled)
        probabilities_valid = self.model.predict_proba(X_scaled)
        confidence_valid = np.max(probabilities_valid, axis=1)

        # Reconstruct full arrays
        predictions_full = np.full(pixels, 255, dtype=np.uint8)  # 255 = nodata
        confidence_full = np.full(pixels, 0.0, dtype=np.float32)

        predictions_full[valid_mask] = predictions_valid
        confidence_full[valid_mask] = confidence_valid

        # Reshape to spatial dimensions
        predictions_map = predictions_full.reshape(H, W)
        confidence_map = confidence_full.reshape(H, W)

        # Apply confidence threshold
        low_confidence_mask = confidence_map < self.config.confidence_threshold
        predictions_map[low_confidence_mask] = 0  # Set to "no change" if low confidence

        # Log statistics
        unique, counts = np.unique(predictions_map[predictions_map != 255], return_counts=True)
        total_valid = np.sum(counts)

        self.logger.info(" 📊 Historical Prediction Statistics:")
        class_names = {0: "No Change", 1: "Urban Increase", 2: "Urban Decrease"}
        for u, c in zip(unique, counts):
            percentage = (c / total_valid) * 100 if total_valid > 0 else 0
            self.logger.info(f"   - {class_names.get(u, f'Class {u}')}: {c:,} pixels ({percentage:.2f}%)")

        avg_confidence = confidence_full[valid_mask].mean()
        self.logger.info(f"   - Average confidence: {avg_confidence:.3f}")

        return predictions_map, confidence_map

    def predict_future_changes(self, time_series: np.ndarray, years: List[int],
                             future_years: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict future land cover changes using trend extrapolation

        Args:
            time_series: Historical time series (T, H, W)
            years: Historical years
            future_years: Years to predict

        Returns:
            Tuple[np.ndarray, np.ndarray]: (future_predictions, confidence_scores)
        """
        self.logger.info(f"🔮 Predicting future changes for years: {future_years}")

        T, H, W = time_series.shape
        pixels = H * W

        # Vectorized trend extrapolation
        y_vec = np.array(years)
        y_mean = y_vec.mean()
        denominator = ((y_vec - y_mean) ** 2).sum()

        # Reshape time series for vectorized operations
        ts_reshaped = time_series.reshape(T, pixels)  # (T, pixels)

        # Calculate linear trends per pixel
        val_mean = np.nanmean(ts_reshaped, axis=0)  # (pixels,)

        # Vectorized slope calculation
        numerator = np.nansum(
            (y_vec[:, np.newaxis] - y_mean) * (ts_reshaped - val_mean[np.newaxis, :]),
            axis=0
        )
        slopes = numerator / denominator
        intercepts = val_mean - slopes * y_mean

        # Predict for target future year (using the last year in future_years)
        target_year = future_years[-1]
        future_ndbi = slopes * target_year + intercepts

        # Create synthetic time series for prediction
        # Use recent historical data + projected future value
        synthetic_ts = time_series.copy()

        # Replace last year with projected values
        future_map = future_ndbi.reshape(H, W)
        synthetic_ts[-1] = future_map

        # Extract features for the synthetic time series
        feature_engineer = FeatureEngineer(self.config)
        X_future, _ = feature_engineer.extract_temporal_features(synthetic_ts, years)

        # Handle invalid pixels
        valid_mask = ~np.isnan(X_future).any(axis=1)
        X_future_valid = X_future[valid_mask]

        # Scale and predict
        X_future_scaled = self.scaler.transform(X_future_valid)
        predictions_valid = self.model.predict(X_future_scaled)
        probabilities_valid = self.model.predict_proba(X_future_scaled)
        confidence_valid = np.max(probabilities_valid, axis=1)

        # Reconstruct full arrays
        predictions_full = np.full(pixels, 255, dtype=np.uint8)
        confidence_full = np.full(pixels, 0.0, dtype=np.float32)

        predictions_full[valid_mask] = predictions_valid
        confidence_full[valid_mask] = confidence_valid

        predictions_map = predictions_full.reshape(H, W)
        confidence_map = confidence_full.reshape(H, W)

        # Apply confidence threshold
        low_confidence_mask = confidence_map < self.config.confidence_threshold
        predictions_map[low_confidence_mask] = 0

        self.logger.info(f" ✅ Future predictions generated for {target_year}")

        return predictions_map, confidence_map

    def save_geotiff(self, data: np.ndarray, metadata: Dict, filename: str, 
                     dtype: str = 'float32', nodata: float = -9999) -> str:
        """
        Save data as GeoTIFF file
        
        Args:
            data: Array to save
            metadata: Raster metadata
            filename: Output filename
            dtype: Data type for output
            nodata: NoData value
            
        Returns:
            str: Path to saved file
        """
        # Prepare output metadata
        out_meta = {
            'driver': 'GTiff',
            'height': data.shape[0],
            'width': data.shape[1],
            'count': 1,
            'transform': metadata['reference']['transform'],
            'crs': metadata['reference']['crs'],
            'compress': 'lzw',
            'nodata': nodata,
            'dtype': dtype
        }
        
        # Create output path
        output_path = Path(self.config.output_dir) / filename
        
        # Save file
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(data, 1)
            
        self.logger.info(f" 💾 Saved GeoTIFF: {output_path}")
        return str(output_path)

    def save_predictions(self, predictions: np.ndarray, confidence: np.ndarray,
                        metadata: Dict, suffix: str) -> Tuple[str, str]:
        """Save prediction results as GeoTIFF files"""

        # Save predictions
        pred_path = self.save_geotiff(
            predictions, metadata, 
            f"land_cover_predictions_{suffix}.tif",
            dtype='uint8', nodata=255
        )

        # Save confidence scores
        conf_path = self.save_geotiff(
            confidence, metadata,
            f"prediction_confidence_{suffix}.tif",
            dtype='float32', nodata=-9999
        )

        return pred_path, conf_path 