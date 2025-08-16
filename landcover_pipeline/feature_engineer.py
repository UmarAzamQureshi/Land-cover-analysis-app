"""
Feature engineering module for land cover change detection
"""

import logging
import numpy as np
from scipy import stats
from typing import List, Tuple

class FeatureEngineer:
    """Advanced feature engineering for time-series land cover analysis"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_pseudo_labels(self, time_series: np.ndarray, years: List[int]) -> np.ndarray:
        """
        Create sophisticated pseudo-labels using multiple criteria

        Args:
            time_series: Shape (T, H, W) time series data
            years: List of years corresponding to time dimension

        Returns:
            np.ndarray: Label array with shape (H, W)
        """
        self.logger.info("🏷️  Creating pseudo-labels with advanced criteria...")

        T, H, W = time_series.shape

        # Multiple change detection approaches
        start_ndbi = time_series[0]  # First year
        end_ndbi = time_series[-1]   # Last year

        # 1. Simple difference
        simple_diff = end_ndbi - start_ndbi

        # 2. Trend analysis using linear regression per pixel
        x_years = np.array(years)
        trends = np.full((H, W), np.nan, dtype=np.float32)

        # Vectorized trend calculation for efficiency
        for i in range(H):
            for j in range(W):
                pixel_series = time_series[:, i, j]
                if not np.isnan(pixel_series).any():
                    slope, _, _, _, _ = stats.linregress(x_years, pixel_series)
                    trends[i, j] = slope

        # 3. Volatility analysis (standard deviation over time)
        volatility = np.nanstd(time_series, axis=0)

        # 4. Multi-criteria labeling
        labels = np.zeros((H, W), dtype=np.uint8)

        # Dynamic thresholds based on data distribution
        diff_threshold = np.nanpercentile(np.abs(simple_diff), 85)
        trend_threshold = np.nanpercentile(np.abs(trends), 80)
        volatility_threshold = np.nanpercentile(volatility, 90)

        # Conservative criteria for high-confidence labels
        # Class 1: Strong urban increase
        urban_increase_mask = (
            (simple_diff > diff_threshold) &
            (trends > trend_threshold * 0.5) &
            (volatility < volatility_threshold)  # Low volatility = consistent change
        )

        # Class 2: Strong urban decrease
        urban_decrease_mask = (
            (simple_diff < -diff_threshold) &
            (trends < -trend_threshold * 0.5) &
            (volatility < volatility_threshold)
        )

        labels[urban_increase_mask] = 1
        labels[urban_decrease_mask] = 2

        # Log label distribution
        unique, counts = np.unique(labels, return_counts=True)
        total_valid = H * W - np.isnan(simple_diff).sum()

        self.logger.info(" 📊 Pseudo-label distribution:")
        class_names = {0: "No Change", 1: "Urban Increase", 2: "Urban Decrease"}
        for u, c in zip(unique, counts):
            percentage = (c / total_valid) * 100 if total_valid > 0 else 0
            self.logger.info(f"   - {class_names.get(u, f'Class {u}')}: {c:,} pixels ({percentage:.2f}%)")

        return labels

    def extract_temporal_features(self, time_series: np.ndarray, years: List[int]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive temporal features from time series

        Args:
            time_series: Shape (T, H, W) time series data
            years: List of years

        Returns:
            Tuple[np.ndarray, List[str]]: (Feature array, Feature names)
        """
        self.logger.info("🔧 Extracting temporal features...")

        T, H, W = time_series.shape
        pixels = H * W

        # Reshape for easier processing: (pixels, time)
        ts_reshaped = time_series.reshape(T, pixels).T

        # Feature list
        features = []
        feature_names = []

        # 1. Original time series values
        features.append(ts_reshaped)
        feature_names.extend([f'ndbi_year_{year}' for year in years])

        # 2. Statistical features
        features.append(np.nanmean(ts_reshaped, axis=1, keepdims=True))
        features.append(np.nanstd(ts_reshaped, axis=1, keepdims=True))
        features.append(np.nanmin(ts_reshaped, axis=1, keepdims=True))
        features.append(np.nanmax(ts_reshaped, axis=1, keepdims=True))
        feature_names.extend(['mean_ndbi', 'std_ndbi', 'min_ndbi', 'max_ndbi'])

        # 3. Temporal trend features
        x_years = np.array(years)
        slopes = np.full((pixels, 1), np.nan)
        r_values = np.full((pixels, 1), np.nan)

        # Vectorized slope calculation
        y_mean = x_years.mean()
        denominator = ((x_years - y_mean) ** 2).sum()
        
        for i in range(pixels):
            pixel_series = ts_reshaped[i]
            if not np.isnan(pixel_series).any():
                slope, _, r_value, _, _ = stats.linregress(x_years, pixel_series)
                slopes[i, 0] = slope
                r_values[i, 0] = r_value

        features.extend([slopes, r_values])
        feature_names.extend(['linear_trend', 'trend_r_value'])

        # 4. Difference features
        if T >= 2:
            first_half_mean = np.nanmean(ts_reshaped[:, :T//2], axis=1, keepdims=True)
            second_half_mean = np.nanmean(ts_reshaped[:, T//2:], axis=1, keepdims=True)
            half_diff = second_half_mean - first_half_mean

            total_change = ts_reshaped[:, -1:] - ts_reshaped[:, :1]

            features.extend([half_diff, total_change])
            feature_names.extend(['first_second_half_diff', 'total_change'])

        # 5. Relative features
        range_vals = np.nanmax(ts_reshaped, axis=1, keepdims=True) - np.nanmin(ts_reshaped, axis=1, keepdims=True)
        cv = np.nanstd(ts_reshaped, axis=1, keepdims=True) / (np.nanmean(ts_reshaped, axis=1, keepdims=True) + 1e-8)

        features.extend([range_vals, cv])
        feature_names.extend(['ndbi_range', 'coefficient_variation'])

        # Concatenate all features
        X = np.concatenate(features, axis=1)

        self.logger.info(f" ✅ Extracted {X.shape[1]} temporal features")
        self.logger.info(f"   - Feature names: {feature_names[:5]}... (showing first 5)")

        return X, feature_names 