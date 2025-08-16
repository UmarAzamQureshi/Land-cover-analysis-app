"""
Main pipeline orchestration for land cover change detection
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import joblib
import numpy as np

from .data_processor import DataProcessor
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .predictor import Predictor
from .visualizer import Visualizer

class LandCoverAnalysisPipeline:
    """Main enterprise pipeline orchestrating the entire analysis"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run(self) -> Dict[str, Any]:
        """
        Run the complete land cover change detection and prediction pipeline

        Returns:
            Dict: Summary of all results and output paths
        """
        self.logger.info("🚀 Starting complete land cover analysis pipeline...")

        results = {
            'start_time': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'outputs': {},
            'performance': {}
        }

        try:
            # 1. Data Processing
            self.logger.info("=" * 60)
            self.logger.info("PHASE 1: DATA PROCESSING")
            self.logger.info("=" * 60)

            processor = DataProcessor(self.config)
            file_list = processor.discover_and_validate_files()
            time_series, metadata = processor.load_and_preprocess_rasters(file_list)
            years = [item[0] for item in file_list]

            results['data_info'] = {
                'years': years,
                'shape': time_series.shape,
                'metadata': metadata
            }

            # 2. Feature Engineering
            self.logger.info("=" * 60)
            self.logger.info("PHASE 2: FEATURE ENGINEERING")
            self.logger.info("=" * 60)

            feature_engineer = FeatureEngineer(self.config)
            labels = feature_engineer.create_pseudo_labels(time_series, years)
            features, feature_names = feature_engineer.extract_temporal_features(time_series, years)

            # 3. Model Training
            self.logger.info("=" * 60)
            self.logger.info("PHASE 3: MODEL TRAINING")
            self.logger.info("=" * 60)

            trainer = ModelTrainer(self.config)
            X_train, X_test, y_train, y_test = trainer.prepare_training_data(
                features, labels.ravel(), feature_names
            )

            evaluation_results = trainer.train_and_evaluate(
                X_train, X_test, y_train, y_test
            )

            model_path = trainer.save_model()
            results['performance'] = evaluation_results
            results['outputs']['model_path'] = model_path

            # 4. Predictions
            self.logger.info("=" * 60)
            self.logger.info("PHASE 4: GENERATING PREDICTIONS")
            self.logger.info("=" * 60)

            # Load model artifacts
            model_artifacts = joblib.load(model_path)
            predictor = Predictor(self.config, model_artifacts)

            # Historical predictions
            predictions_hist, confidence_hist = predictor.predict_historical_changes(
                time_series, years, metadata
            )

            # Future predictions
            future_years = list(range(years[-1] + 1, years[-1] + self.config.future_years + 1))
            predictions_future, confidence_future = predictor.predict_future_changes(
                time_series, years, future_years
            )

            # Save predictions
            hist_pred_path, hist_conf_path = predictor.save_predictions(
                predictions_hist, confidence_hist, metadata, "historical"
            )

            future_pred_path, future_conf_path = predictor.save_predictions(
                predictions_future, confidence_future, metadata, "future"
            )

            results['outputs'].update({
                'historical_predictions': hist_pred_path,
                'historical_confidence': hist_conf_path,
                'future_predictions': future_pred_path,
                'future_confidence': future_conf_path
            })

            # 5. Visualization and Reporting
            self.logger.info("=" * 60)
            self.logger.info("PHASE 5: CREATING VISUALIZATIONS")
            self.logger.info("=" * 60)

            visualizer = Visualizer(self.config)

            # Comprehensive report
            report_path = visualizer.create_comprehensive_report(
                time_series, years, predictions_hist, predictions_future,
                confidence_hist, confidence_future, evaluation_results, metadata
            )

            # Interactive map
            map_path = visualizer.create_interactive_map(
                predictions_hist, predictions_future,
                confidence_hist, confidence_future,
                metadata, years
            )

            results['outputs'].update({
                'comprehensive_report': report_path,
                'interactive_map': map_path
            })

            # 6. Generate Executive Summary
            self.logger.info("=" * 60)
            self.logger.info("PHASE 6: GENERATING EXECUTIVE SUMMARY")
            self.logger.info("=" * 60)

            summary_path = self._create_executive_summary(
                results, time_series, years, predictions_hist, predictions_future
            )
            results['outputs']['executive_summary'] = summary_path

            # Complete
            results['end_time'] = datetime.now().isoformat()
            results['status'] = 'SUCCESS'

            self.logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 60)
            self.logger.info("OUTPUT SUMMARY:")
            for output_name, output_path in results['outputs'].items():
                self.logger.info(f"   - {output_name}: {output_path}")
            self.logger.info("=" * 60)

        except Exception as e:
            results['end_time'] = datetime.now().isoformat()
            results['status'] = 'FAILED'
            results['error'] = str(e)
            self.logger.error(f" ❌ Pipeline failed: {str(e)}")
            raise

        return results

    def run_and_save(self) -> str:
        """
        Run pipeline and save results manifest
        
        Returns:
            str: Path to manifest file
        """
        try:
            # Run the pipeline
            results = self.run()
            
            # Save manifest
            manifest_path = Path(self.config.output_dir) / "manifest.json"
            
            with open(manifest_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"📋 Manifest saved: {manifest_path}")
            return str(manifest_path)
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {str(e)}")
            raise

    def _create_executive_summary(self, results: Dict, time_series: np.ndarray,
                                years: List[int], predictions_hist: np.ndarray,
                                predictions_future: np.ndarray) -> str:
        """Create executive summary document"""

        # Calculate key statistics
        total_pixels = np.prod(predictions_hist.shape)
        valid_pixels = np.sum(predictions_hist != 255)

        hist_stats = np.bincount(predictions_hist[predictions_hist != 255].astype(int))
        future_stats = np.bincount(predictions_future[predictions_future != 255].astype(int))

        # Ensure we have stats for all classes
        while len(hist_stats) < 3:
            hist_stats = np.append(hist_stats, 0)
        while len(future_stats) < 3:
            future_stats = np.append(future_stats, 0)

        urban_increase_hist = (hist_stats[1] / valid_pixels * 100) if valid_pixels > 0 else 0
        urban_decrease_hist = (hist_stats[2] / valid_pixels * 100) if valid_pixels > 0 else 0

        urban_increase_future = (future_stats[1] / valid_pixels * 100) if valid_pixels > 0 else 0
        urban_decrease_future = (future_stats[2] / valid_pixels * 100) if valid_pixels > 0 else 0

        # Performance metrics
        test_accuracy = results['performance']['test_accuracy']
        cv_f1_mean = results['performance']['cv_scores']['mean']
        cv_f1_std = results['performance']['cv_scores']['std']

        # Create summary
        summary_content = f"""

# EXECUTIVE SUMMARY: LAND COVER CHANGE ANALYSIS

## Project Overview
**Analysis Period:** {years[0]} - {years[-1]} ({len(years)} years)
**Prediction Year:** {years[-1] + self.config.future_years}
**Study Area:** {predictions_hist.shape[1]} × {predictions_hist.shape[0]} pixels
**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}

## Key Findings

### Historical Changes ({years[0]}-{years[-1]})
- **Urban Expansion:** {urban_increase_hist:.2f}% of study area
- **Urban Reduction:** {urban_decrease_hist:.2f}% of study area
- **Stable Areas:** {100 - urban_increase_hist - urban_decrease_hist:.2f}% of study area

### Future Projections ({years[-1] + self.config.future_years})
- **Predicted Urban Expansion:** {urban_increase_future:.2f}% of study area
- **Predicted Urban Reduction:** {urban_decrease_future:.2f}% of study area
- **Expected Stable Areas:** {100 - urban_increase_future - urban_decrease_future:.2f}% of study area

### Change Trends
- **Net Urban Change (Historical):** {urban_increase_hist - urban_decrease_hist:+.2f}% of study area
- **Net Urban Change (Predicted):** {urban_increase_future - urban_decrease_future:+.2f}% of study area
- **Acceleration Factor:** {((urban_increase_future - urban_decrease_future) / max(urban_increase_hist - urban_decrease_hist, 0.01)):.2f}x

## Model Performance
- **Test Accuracy:** {test_accuracy:.1%}
- **Cross-Validation F1-Score:** {cv_f1_mean:.3f} ± {cv_f1_std:.3f}
- **Model Type:** Random Forest Classifier with {len(results['data_info']['years'])} temporal features
- **Confidence Threshold:** {self.config.confidence_threshold:.1%}

## Technical Specifications
- **Data Source:** NDBI (Normalized Difference Built-up Index) time series
- **Spatial Resolution:** {predictions_hist.shape[1]} × {predictions_hist.shape[0]} pixels
- **Temporal Resolution:** {(years[-1] - years[0]) / (len(years) - 1):.1f} years average
- **Processing Framework:** Enterprise Machine Learning Pipeline
- **Validation Method:** Stratified K-Fold Cross-Validation

## Confidence Assessment
The model predictions include pixel-level confidence scores:
- High confidence predictions (>{self.config.confidence_threshold:.0%}) are retained
- Low confidence areas are classified as "No Change" (conservative approach)
- Average prediction confidence varies by region and time period

## Recommendations

### Urban Planning
1. **Priority Areas:** Focus development planning on areas predicted for high urban expansion
2. **Conservation Zones:** Protect areas showing urban decrease trends for ecological value
3. **Infrastructure Planning:** Prepare infrastructure for projected urban growth areas

### Environmental Management
1. **Monitoring:** Establish regular monitoring in high-change areas
2. **Mitigation:** Develop strategies for areas experiencing rapid urbanization
3. **Sustainability:** Balance urban development with environmental conservation

### Data Collection
1. **Validation:** Collect ground truth data for model validation and improvement
2. **Frequency:** Increase temporal resolution for better trend detection
3. **Integration:** Combine with other data sources (demographics, economic indicators)

## Limitations and Uncertainties
- Predictions based on historical trends may not account for policy changes
- Model confidence varies spatially and should be considered in decision-making
- Future predictions assume continuation of observed patterns
- External factors (economic crises, natural disasters) not explicitly modeled

## Deliverables
- Comprehensive Analysis Report (PNG)
- Interactive Web Map (HTML)
- Prediction Rasters (GeoTIFF format)
- Confidence Maps (GeoTIFF format)
- Trained Model (PKL format)
- Technical Documentation

---
*Generated by Enterprise Land Cover Analysis System v2.0.0*
*For technical questions, contact the Data Science Team*
        """

        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = Path(self.config.output_dir) / f"executive_summary_{timestamp}.md"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)

        self.logger.info(f"📋 Executive summary saved: {summary_path}")
        return str(summary_path) 