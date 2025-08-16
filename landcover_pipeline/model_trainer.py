"""
Model training module for land cover change detection
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any
import joblib

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    StratifiedKFold, learning_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

class ModelTrainer:
    """Enterprise-grade model training with comprehensive validation"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_history = {}

    def prepare_training_data(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> Tuple:
        """
        Prepare and clean training data with advanced preprocessing

        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names

        Returns:
            Tuple of processed training data
        """
        self.logger.info("🧹 Preparing training data...")

        # Remove pixels with any NaN features
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        self.logger.info(f"   - Valid samples: {X_clean.shape[0]:,} / {X.shape[0]:,} ({100*X_clean.shape[0]/X.shape[0]:.1f}%)")

        # Handle class imbalance by sampling
        class_counts = np.bincount(y_clean)
        min_class_size = min(class_counts[class_counts > 0])
        max_samples_per_class = min(self.config.max_samples_per_class, min_class_size * 5)

        # Stratified sampling for balanced training
        indices_to_keep = []
        for class_label in np.unique(y_clean):
            class_indices = np.where(y_clean == class_label)[0]
            if len(class_indices) > max_samples_per_class:
                selected_indices = np.random.choice(
                    class_indices,
                    size=max_samples_per_class,
                    replace=False
                )
            else:
                selected_indices = class_indices
            indices_to_keep.extend(selected_indices)

        indices_to_keep = np.array(indices_to_keep)
        np.random.shuffle(indices_to_keep)

        X_sampled = X_clean[indices_to_keep]
        y_sampled = y_clean[indices_to_keep]

        # Feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_sampled)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_sampled,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_sampled
        )

        self.feature_names = feature_names

        self.logger.info(f"   - Training samples: {X_train.shape[0]:,}")
        self.logger.info(f"   - Test samples: {X_test.shape[0]:,}")

        # Store class distribution
        train_dist = np.bincount(y_train)
        self.logger.info(f"   - Training class distribution: {dict(enumerate(train_dist))}")

        return X_train, X_test, y_train, y_test

    def hyperparameter_optimization(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Perform comprehensive hyperparameter optimization

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            RandomForestClassifier: Optimized model
        """
        self.logger.info("🔍 Starting hyperparameter optimization...")

        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        # Create base model with class weights
        base_rf = RandomForestClassifier(
            class_weight=class_weight_dict,
            random_state=self.config.random_state,
            n_jobs=1
        )

        # Grid search with stratified cross-validation
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)

        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=self.config.rf_param_grid,
            cv=cv,
            scoring='f1_macro',  # Better for imbalanced data
            n_jobs=1,
            verbose=1
        )

        # Fit grid search
        self.logger.info("   - Running grid search... (this may take several minutes)")
        grid_search.fit(X_train, y_train)

        # Log results
        self.logger.info(f"   - Best CV score: {grid_search.best_score_:.4f}")
        self.logger.info(f"   - Best parameters: {grid_search.best_params_}")

        # Store training history
        self.training_history['hyperparameter_search'] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }

        return grid_search.best_estimator_

    def train_and_evaluate(self, X_train, X_test, y_train, y_test) -> Dict:
        """
        Train model and perform comprehensive evaluation

        Returns:
            Dict: Comprehensive evaluation results
        """
        self.logger.info("🚀 Training optimized model...")

        # Hyperparameter optimization
        self.model = self.hyperparameter_optimization(X_train, y_train)

        # Train on full training set
        self.model.fit(X_train, y_train)

        # Predictions on all sets
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Probabilities for confidence analysis
        y_test_proba = self.model.predict_proba(X_test)

        # Comprehensive evaluation
        evaluation_results = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
        }

        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.random_state),
            scoring='f1_macro'
        )
        evaluation_results['cv_scores'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }

        # Learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X_train, y_train,
            cv=3, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 10)
        )

        evaluation_results['learning_curves'] = {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': train_scores.mean(axis=1).tolist(),
            'train_scores_std': train_scores.std(axis=1).tolist(),
            'val_scores_mean': val_scores.mean(axis=1).tolist(),
            'val_scores_std': val_scores.std(axis=1).tolist()
        }

        # Store evaluation results
        self.training_history['evaluation'] = evaluation_results

        # Log key metrics
        self.logger.info(" 📊 Model Performance Summary:")
        self.logger.info(f"   - Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
        self.logger.info(f"   - CV F1-Score: {evaluation_results['cv_scores']['mean']:.4f} ± {evaluation_results['cv_scores']['std']:.4f}")

        # Feature importance analysis
        feature_importance = evaluation_results['feature_importance']
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        self.logger.info("🔝 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(top_features, 1):
            self.logger.info(f"   {i}. {feature}: {importance:.4f}")

        return evaluation_results

    def save_model(self) -> str:
        """Save trained model and preprocessing objects"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'config': self.config,
            'timestamp': timestamp
        }

        model_path = Path(self.config.model_dir) / f"land_cover_model_{timestamp}.pkl"

        joblib.dump(model_artifacts, model_path)

        self.logger.info(f" 💾 Model saved: {model_path}")

        return str(model_path) 