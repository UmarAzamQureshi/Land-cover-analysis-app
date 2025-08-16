"""
Data processing module for land cover change detection
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.plot import show
from rasterio.transform import from_bounds
import geopandas as gpd

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

class DataProcessor:
    """Enterprise-grade data processing pipeline for NDBI time-series analysis"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metadata = {}

    def discover_and_validate_files(self) -> List[Tuple[int, str]]:
        """
        Discover and validate NDBI TIFF files with comprehensive error handling

        Returns:
            List[Tuple[int, str]]: List of (year, filepath) tuples sorted by year
        """
        self.logger.info("🔍 Discovering and validating NDBI files...")

        if not Path(self.config.data_folder).exists():
            raise FileNotFoundError(f"Data folder not found: {self.config.data_folder}")

        tif_files = []
        skipped_files = []

        # Pattern matching for years (more robust)
        year_pattern = re.compile(r'(19|20)\d{2}')

        for file_path in Path(self.config.data_folder).glob("*.tif*"):
            try:
                # Extract year from filename
                year_match = year_pattern.search(file_path.name)
                if not year_match:
                    skipped_files.append((file_path.name, "No year found in filename"))
                    continue

                year = int(year_match.group(0))

                # Validate year range (reasonable bounds)
                if not (1990 <= year <= 2030):
                    skipped_files.append((file_path.name, f"Year {year} out of valid range"))
                    continue

                # Basic file validation
                try:
                    with rasterio.open(file_path) as src:
                        if src.count != 1:
                            skipped_files.append((file_path.name, f"Expected 1 band, got {src.count}"))
                            continue
                        if src.width == 0 or src.height == 0:
                            skipped_files.append((file_path.name, "Invalid raster dimensions"))
                            continue
                except Exception as e:
                    skipped_files.append((file_path.name, f"Raster read error: {str(e)}"))
                    continue

                tif_files.append((year, str(file_path)))

            except Exception as e:
                skipped_files.append((file_path.name, f"Processing error: {str(e)}"))

        # Log results
        if skipped_files:
            self.logger.warning(f"  Skipped {len(skipped_files)} files:")
            for filename, reason in skipped_files:
                self.logger.warning(f"   - {filename}: {reason}")

        if len(tif_files) < 2:
            raise ValueError(f"Insufficient valid files found. Need at least 2, got {len(tif_files)}")

        # Sort by year
        tif_files = sorted(tif_files, key=lambda x: x[0])

        self.logger.info(f" ✅ Found {len(tif_files)} valid files from {tif_files[0][0]} to {tif_files[-1][0]}")

        return tif_files

    def load_and_preprocess_rasters(self, file_list: List[Tuple[int, str]]) -> Tuple[np.ndarray, Dict]:
        """
        Load and preprocess raster data with comprehensive quality checks

        Args:
            file_list: List of (year, filepath) tuples

        Returns:
            Tuple[np.ndarray, Dict]: (time_series_stack, metadata)
        """
        self.logger.info("📊 Loading and preprocessing raster data...")

        # Get reference metadata from first file
        years, filepaths = zip(*file_list)

        with rasterio.open(filepaths[0]) as src:
            reference_meta = {
                'height': src.height,
                'width': src.width,
                'transform': src.transform,
                'crs': src.crs,
                'bounds': src.bounds,
                'nodata': src.nodata
            }

        self.logger.info(f"  Reference raster: {reference_meta['width']}x{reference_meta['height']} pixels")
        self.logger.info(f"  CRS: {reference_meta['crs']}")

        # Apply quick mode if enabled
        if self.config.quick_mode:
            self.logger.info(f"  🚀 Quick mode enabled - cropping to {self.config.quick_window_size}x{self.config.quick_window_size} center window")
            
            # Calculate crop window
            center_h = reference_meta['height'] // 2
            center_w = reference_meta['width'] // 2
            half_window = self.config.quick_window_size // 2
            
            start_h = max(0, center_h - half_window)
            end_h = min(reference_meta['height'], center_h + half_window)
            start_w = max(0, center_w - half_window)
            end_w = min(reference_meta['width'], center_w + half_window)
            
            # Update reference metadata
            reference_meta['height'] = end_h - start_h
            reference_meta['width'] = end_w - start_w
            reference_meta['crop_window'] = (start_h, end_h, start_w, end_w)
            
            self.logger.info(f"  Cropped to: {reference_meta['width']}x{reference_meta['height']} pixels")

        # Pre-allocate array for efficiency
        stack = np.empty((len(years), reference_meta['height'], reference_meta['width']), dtype=np.float32)

        # Statistics tracking
        stats_per_year = []

        # Load each raster with progress tracking
        for i, (year, filepath) in enumerate(file_list):
            try:
                with rasterio.open(filepath) as src:
                    # Validate consistency
                    if (src.height, src.width) != (reference_meta.get('original_height', src.height), 
                                                   reference_meta.get('original_width', src.width)):
                        self.logger.warning(f"  Resampling {year} from {src.width}x{src.height} to reference size")

                        # Resample to match reference
                        data = src.read(
                            1,
                            out_shape=(reference_meta['height'], reference_meta['width']),
                            resampling=Resampling.bilinear
                        ).astype(np.float32)
                    else:
                        data = src.read(1).astype(np.float32)

                    # Apply crop window if in quick mode
                    if self.config.quick_mode and 'crop_window' in reference_meta:
                        start_h, end_h, start_w, end_w = reference_meta['crop_window']
                        data = data[start_h:end_h, start_w:end_w]

                    # Handle nodata values
                    if src.nodata is not None:
                        data[data == src.nodata] = np.nan

                    # Quality checks
                    finite_pixels = np.isfinite(data).sum()
                    total_pixels = data.size
                    valid_percentage = (finite_pixels / total_pixels) * 100

                    if valid_percentage < 50:
                        self.logger.warning(f"  {year}: Only {valid_percentage:.1f}% valid pixels")

                    # Store data
                    stack[i] = data

                    # Calculate statistics
                    year_stats = {
                        'year': year,
                        'mean': np.nanmean(data),
                        'std': np.nanstd(data),
                        'min': np.nanmin(data),
                        'max': np.nanmax(data),
                        'valid_pixels': finite_pixels,
                        'valid_percentage': valid_percentage
                    }
                    stats_per_year.append(year_stats)

            except Exception as e:
                self.logger.error(f" ❌ Error loading {year} from {filepath}: {str(e)}")
                raise

        # Create comprehensive metadata
        metadata = {
            'reference': reference_meta,
            'years': list(years),
            'statistics': stats_per_year,
            'shape': stack.shape,
            'temporal_span': years[-1] - years[0],
            'temporal_resolution': (years[-1] - years[0]) / (len(years) - 1) if len(years) > 1 else 1
        }

        # Log summary statistics
        overall_stats = pd.DataFrame(stats_per_year)
        self.logger.info(" 📈 Data Quality Summary:")
        self.logger.info(f"   - Temporal span: {metadata['temporal_span']} years")
        self.logger.info(f"   - Average valid pixels: {overall_stats['valid_percentage'].mean():.1f}%")
        self.logger.info(f"   - NDBI range: [{overall_stats['min'].min():.3f}, {overall_stats['max'].max():.3f}]")

        return stack, metadata 