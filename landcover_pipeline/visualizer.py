"""
Visualization module for land cover change detection
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
from folium import plugins
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from PIL import Image

class Visualizer:
    """Enterprise-grade visualization and reporting system"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_comprehensive_report(self, time_series: np.ndarray, years: List[int],
                                  predictions_hist: np.ndarray, predictions_future: np.ndarray,
                                  confidence_hist: np.ndarray, confidence_future: np.ndarray,
                                  evaluation_results: Dict, metadata: Dict) -> str:
        """
        Create comprehensive analysis report with multiple visualizations

        Returns:
            str: Path to saved report
        """
        self.logger.info("📊 Creating comprehensive analysis report...")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)

        # Color scheme for classes
        class_colors = {0: '#E5E5E5', 1: '#FF4444', 2: '#4444FF', 255: '#000000'}
        class_names = {0: 'No Change', 1: 'Urban Increase', 2: 'Urban Decrease', 255: 'No Data'}

        # 1. Time series overview
        ax1 = fig.add_subplot(gs[0, :])
        mean_ndbi = np.nanmean(time_series.reshape(len(years), -1), axis=1)
        std_ndbi = np.nanstd(time_series.reshape(len(years), -1), axis=1)

        ax1.plot(years, mean_ndbi, 'b-', linewidth=2, label='Mean NDBI')
        ax1.fill_between(years, mean_ndbi - std_ndbi, mean_ndbi + std_ndbi, alpha=0.3)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('NDBI Value')
        ax1.set_title('NDBI Time Series Overview', fontsize=16, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Historical predictions
        ax2 = fig.add_subplot(gs[1, 0])
        pred_rgb = self._create_class_rgb(predictions_hist, class_colors)
        ax2.imshow(pred_rgb)
        ax2.set_title(f'Historical Changes\n({years[0]}-{years[-1]})', fontweight='bold')
        ax2.axis('off')

        # 3. Future predictions
        ax3 = fig.add_subplot(gs[1, 1])
        future_rgb = self._create_class_rgb(predictions_future, class_colors)
        ax3.imshow(future_rgb)
        future_year = years[-1] + self.config.future_years
        ax3.set_title(f'Future Predictions\n({future_year})', fontweight='bold')
        ax3.axis('off')

        # 4. Confidence maps
        ax4 = fig.add_subplot(gs[1, 2])
        im4 = ax4.imshow(confidence_hist, cmap='viridis', vmin=0, vmax=1)
        ax4.set_title('Historical\nConfidence', fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        ax5 = fig.add_subplot(gs[1, 3])
        im5 = ax5.imshow(confidence_future, cmap='viridis', vmin=0, vmax=1)
        ax5.set_title('Future\nConfidence', fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        # 5. Class distribution
        ax6 = fig.add_subplot(gs[2, :2])
        hist_stats = self._calculate_class_statistics(predictions_hist)
        future_stats = self._calculate_class_statistics(predictions_future)

        x = np.arange(len(class_names) - 1)  # Exclude nodata class
        width = 0.35

        hist_counts = [hist_stats.get(i, 0) for i in range(3)]
        future_counts = [future_stats.get(i, 0) for i in range(3)]

        ax6.bar(x - width/2, hist_counts, width, label='Historical', alpha=0.8)
        ax6.bar(x + width/2, future_counts, width, label='Future', alpha=0.8)

        ax6.set_xlabel('Change Class')
        ax6.set_ylabel('Number of Pixels')
        ax6.set_title('Change Class Distribution Comparison', fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels([class_names[i] for i in range(3)])
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 6. Model performance metrics
        ax7 = fig.add_subplot(gs[2, 2:])
        metrics = evaluation_results['classification_report']
        classes = [str(i) for i in range(len(metrics)-3)]  # Exclude avg metrics

        precision = [metrics[c]['precision'] for c in classes]
        recall = [metrics[c]['recall'] for c in classes]
        f1 = [metrics[c]['f1-score'] for c in classes]

        x_pos = np.arange(len(classes))
        ax7.bar(x_pos - 0.25, precision, 0.25, label='Precision', alpha=0.8)
        ax7.bar(x_pos, recall, 0.25, label='Recall', alpha=0.8)
        ax7.bar(x_pos + 0.25, f1, 0.25, label='F1-Score', alpha=0.8)

        ax7.set_xlabel('Class')
        ax7.set_ylabel('Score')
        ax7.set_title('Model Performance by Class', fontweight='bold')
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels([class_names[int(c)] for c in classes])
        ax7.legend()
        ax7.set_ylim(0, 1)
        ax7.grid(True, alpha=0.3)

        # 7. Confusion matrix
        ax8 = fig.add_subplot(gs[3, :2])
        cm = np.array(evaluation_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax8,
                   xticklabels=[class_names[i] for i in range(len(cm))],
                   yticklabels=[class_names[i] for i in range(len(cm))])
        ax8.set_title('Confusion Matrix', fontweight='bold')
        ax8.set_xlabel('Predicted')
        ax8.set_ylabel('Actual')

        # 8. Feature importance
        ax9 = fig.add_subplot(gs[3, 2:])
        feature_importance = evaluation_results['feature_importance']
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]

        features, importances = zip(*top_features)
        y_pos = np.arange(len(features))

        ax9.barh(y_pos, importances, alpha=0.8)
        ax9.set_yticks(y_pos)
        ax9.set_yticklabels(features)
        ax9.set_xlabel('Importance')
        ax9.set_title('Top 15 Feature Importances', fontweight='bold')
        ax9.grid(True, alpha=0.3)

        # 9. Learning curves
        ax10 = fig.add_subplot(gs[4, :2])
        lc = evaluation_results['learning_curves']
        train_sizes = lc['train_sizes']
        train_mean = lc['train_scores_mean']
        train_std = lc['train_scores_std']
        val_mean = lc['val_scores_mean']
        val_std = lc['val_scores_std']

        ax10.plot(train_sizes, train_mean, 'b-', label='Training Score')
        ax10.fill_between(train_sizes,
                         np.array(train_mean) - np.array(train_std),
                         np.array(train_mean) + np.array(train_std), alpha=0.3)
        ax10.plot(train_sizes, val_mean, 'r-', label='Validation Score')
        ax10.fill_between(train_sizes,
                         np.array(val_mean) - np.array(val_std),
                         np.array(val_mean) + np.array(val_std), alpha=0.3)

        ax10.set_xlabel('Training Set Size')
        ax10.set_ylabel('Score')
        ax10.set_title('Learning Curves', fontweight='bold')
        ax10.legend()
        ax10.grid(True, alpha=0.3)

        # 10. Cross-validation scores
        ax11 = fig.add_subplot(gs[4, 2:])
        cv_scores = evaluation_results['cv_scores']['scores']
        ax11.boxplot(cv_scores)
        ax11.set_ylabel('F1-Score')
        ax11.set_title(f'Cross-Validation Scores\n(Mean: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f})',
                      fontweight='bold')
        ax11.grid(True, alpha=0.3)

        # 11. Legend
        ax12 = fig.add_subplot(gs[5, :])
        ax12.axis('off')

        # Create legend patches
        legend_patches = [mpatches.Patch(color=class_colors[i], label=class_names[i])
                         for i in range(3)]
        ax12.legend(handles=legend_patches, loc='center', ncol=3, fontsize=12)

        # Add metadata text
        info_text = f"""
Analysis Summary:
• Temporal Coverage: {years[0]} - {years[-1]} ({len(years)} years)
• Spatial Resolution: {metadata['reference']['width']} × {metadata['reference']['height']} pixels
• Test Accuracy: {evaluation_results['test_accuracy']:.4f}
• Cross-Validation F1: {evaluation_results['cv_scores']['mean']:.4f} ± {evaluation_results['cv_scores']['std']:.4f}
• Future Prediction Year: {years[-1] + self.config.future_years}
        """

        ax12.text(0.1, 0.5, info_text, transform=ax12.transAxes, fontsize=10,
                 verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(self.config.output_dir) / f"comprehensive_analysis_report_{timestamp}.png"
        plt.savefig(report_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Comprehensive report saved: {report_path}")
        return str(report_path)

    def create_interactive_map(self, predictions_hist: np.ndarray, predictions_future: np.ndarray,
                             confidence_hist: np.ndarray, confidence_future: np.ndarray,
                             metadata: Dict, years: List[int]) -> str:
        """Create interactive Folium map with all results"""

        self.logger.info("🗺️  Creating interactive map...")

        # Get bounds from metadata
        bounds = metadata['reference']['bounds']
        center_lat = (bounds.top + bounds.bottom) / 2
        center_lon = (bounds.left + bounds.right) / 2

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=self.config.map_zoom,
            tiles='CartoDB positron'
        )

        # Add satellite imagery as base layer
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite Imagery',
        ).add_to(m)

        # Create RGB overlays
        class_colors = {0: [200, 200, 200, 100], 1: [255, 68, 68, 180], 2: [68, 68, 255, 180], 255: [0, 0, 0, 0]}

        hist_rgb = self._create_class_rgba(predictions_hist, class_colors)
        future_rgb = self._create_class_rgba(predictions_future, class_colors)

        # Save temporary PNG files
        hist_png_path = Path(self.config.output_dir) / "temp_hist_overlay.png"
        future_png_path = Path(self.config.output_dir) / "temp_future_overlay.png"

        Image.fromarray(hist_rgb).save(hist_png_path)
        Image.fromarray(future_rgb).save(future_png_path)

        # Add image overlays
        img_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

        folium.raster_layers.ImageOverlay(
            image=str(hist_png_path),
            bounds=img_bounds,
            opacity=0.8,
            name=f'Historical Changes ({years[0]}-{years[-1]})'
        ).add_to(m)

        folium.raster_layers.ImageOverlay(
            image=str(future_png_path),
            bounds=img_bounds,
            opacity=0.8,
            name=f'Future Predictions ({years[-1] + self.config.future_years})'
        ).add_to(m)

        # Add legend
        legend_html = """
        <div style="position: fixed;
             bottom: 50px; left: 50px; width: 200px; height: 140px;
             background-color: white; border:2px solid grey; z-index:9999;
             font-size:14px; padding: 10px; border-radius: 5px;
             box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h4 style="margin:0 0 10px 0; color:#333;">Land Cover Changes</h4>
        <div style="display:flex; align-items:center; margin:5px 0;">
            <div style="width:18px; height:18px; background:rgb(200,200,200);
                       margin-right:8px; border:1px solid #ccc;"></div>
            <span>No Change</span>
        </div>
        <div style="display:flex; align-items:center; margin:5px 0;">
            <div style="width:18px; height:18px; background:rgb(255,68,68);
                       margin-right:8px; border:1px solid #ccc;"></div>
            <span>Urban Increase</span>
        </div>
        <div style="display:flex; align-items:center; margin:5px 0;">
            <div style="width:18px; height:18px; background:rgb(68,68,255);
                       margin-right:8px; border:1px solid #ccc;"></div>
            <span>Urban Decrease</span>
        </div>
        </div>
        """

        m.get_root().html.add_child(folium.Element(legend_html))

        # Add layer control
        folium.LayerControl().add_to(m)

        # Add fullscreen plugin
        plugins.Fullscreen().add_to(m)

        # Add measure control
        plugins.MeasureControl().add_to(m)

        # Save map
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_path = Path(self.config.output_dir) / f"interactive_map_{timestamp}.html"
        m.save(str(map_path))

        # Clean up temporary files
        hist_png_path.unlink(missing_ok=True)
        future_png_path.unlink(missing_ok=True)

        self.logger.info(f"✅ Interactive map saved: {map_path}")
        return str(map_path)

    def _create_class_rgb(self, predictions: np.ndarray, class_colors: Dict) -> np.ndarray:
        """Create RGB image from class predictions"""
        H, W = predictions.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        for class_id, color_hex in class_colors.items():
            if class_id == 255:  # Skip nodata
                continue
            mask = predictions == class_id
            # Convert hex to RGB if needed
            if isinstance(color_hex, str):
                color_hex = color_hex.lstrip('#')
                color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
            else:
                color_rgb = color_hex[:3]  # Take first 3 values for RGB
            rgb[mask] = color_rgb

        return rgb

    def _create_class_rgba(self, predictions: np.ndarray, class_colors: Dict) -> np.ndarray:
        """Create RGBA image from class predictions"""
        H, W = predictions.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)

        for class_id, color_rgba in class_colors.items():
            mask = predictions == class_id
            rgba[mask] = color_rgba

        return rgba

    def _calculate_class_statistics(self, predictions: np.ndarray) -> Dict:
        """Calculate pixel count statistics by class"""
        unique, counts = np.unique(predictions[predictions != 255], return_counts=True)
        return dict(zip(unique.astype(int), counts.astype(int))) 