#!/usr/bin/env python3
"""
Multimodal Fusion Module for Trading Intelligence

This module implements fusion techniques to combine CLIP vision-language
predictions with XGBoost traditional ML predictions for enhanced trading signals.

Features:
- Ensemble methods (weighted averaging, stacking)
- Confidence-based fusion
- Temporal fusion for time-series data
- Uncertainty quantification
- Performance monitoring and adaptation
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalFusion:
    """
    Fuses CLIP and XGBoost predictions for enhanced trading intelligence
    """

    def __init__(self,
                 clip_model_path: Optional[str] = None,
                 xgb_model_path: Optional[str] = None,
                 fusion_method: str = "weighted_average",
                 device: Optional[str] = None):
        """
        Initialize multimodal fusion

        Args:
            clip_model_path: Path to calibrated CLIP model
            xgb_model_path: Path to XGBoost model
            fusion_method: Fusion strategy ("weighted_average", "stacking", "confidence_weighted")
            device: Device for computation
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.fusion_method = fusion_method

        # Load models
        self.clip_calibrator = None
        self.xgb_model = None
        self.fusion_model = None

        if clip_model_path:
            self._load_clip_model(clip_model_path)
        if xgb_model_path:
            self._load_xgb_model(xgb_model_path)

        # Fusion weights (learned or predefined)
        self.clip_weight = 0.6
        self.xgb_weight = 0.4

        # Performance tracking
        self.performance_history = []

        logger.info(f"[INIT] Multimodal Fusion ({fusion_method}) initialized")

    def _load_clip_model(self, model_path: str):
        """Load calibrated CLIP model"""
        try:
            from vlm.clip_calibration import CLIPCalibrator
            self.clip_calibrator = CLIPCalibrator(model_path, self.device)
            logger.info(f"[LOAD] CLIP model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")

    def _load_xgb_model(self, model_path: str):
        """Load XGBoost model"""
        try:
            import joblib
            self.xgb_model = joblib.load(model_path)
            logger.info(f"[LOAD] XGBoost model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")

    def predict_clip(self, chart_data: Any, text_descriptions: List[str]) -> Dict[str, Any]:
        """
        Get CLIP predictions for chart

        Args:
            chart_data: Chart image or path
            text_descriptions: List of pattern descriptions

        Returns:
            CLIP prediction results
        """
        if not self.clip_calibrator:
            raise RuntimeError("CLIP model not loaded")

        return self.clip_calibrator.predict_calibrated(
            chart_data, text_descriptions, calibration_method="temperature"
        )

    def predict_xgb(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get XGBoost predictions for technical indicators

        Args:
            features: Dictionary of technical indicators

        Returns:
            XGBoost prediction results
        """
        if not self.xgb_model:
            raise RuntimeError("XGBoost model not loaded")

        # Extract features in correct order
        feature_vector = self._extract_feature_vector(features)

        # Make prediction
        prediction = self.xgb_model.predict([feature_vector])[0]
        probabilities = self.xgb_model.predict_proba([feature_vector])[0]

        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'confidence': max(probabilities)
        }

    def _extract_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Extract feature vector for XGBoost from feature dictionary

        Args:
            features: Dictionary of technical indicators

        Returns:
            Feature vector as numpy array
        """
        # This should match the feature engineering in XGBoost training
        feature_order = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'volume_ratio', 'price_change', 'volatility'
        ]

        feature_vector = []
        for feature_name in feature_order:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0.0)  # Default value

        return np.array(feature_vector)

    def fuse_predictions(self,
                        clip_result: Dict[str, Any],
                        xgb_result: Dict[str, Any],
                        vpa_result: Optional[Dict[str, Any]] = None,
                        method: Optional[str] = None) -> Dict[str, Any]:
        """
        Fuse CLIP, XGBoost, and optional VPA predictions

        Args:
            clip_result: CLIP prediction results
            xgb_result: XGBoost prediction results
            vpa_result: Optional VPA prediction results
            method: Fusion method override

        Returns:
            Fused prediction results
        """
        method = method or self.fusion_method

        if vpa_result:
            return self._fuse_three_predictions(clip_result, xgb_result, vpa_result, method)
        else:
            return self._fuse_two_predictions(clip_result, xgb_result, method)

    def _fuse_two_predictions(self, clip_result: Dict[str, Any], xgb_result: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Fuse two predictions (original logic)"""
        if method == "weighted_average":
            return self._weighted_average_fusion(clip_result, xgb_result)
        elif method == "confidence_weighted":
            return self._confidence_weighted_fusion(clip_result, xgb_result)
        elif method == "stacking":
            return self._stacking_fusion(clip_result, xgb_result)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def _fuse_three_predictions(self,
                               clip_result: Dict[str, Any],
                               xgb_result: Dict[str, Any],
                               vpa_result: Dict[str, Any],
                               method: str) -> Dict[str, Any]:
        """Fuse three predictions including VPA"""
        # Adjust weights for three models
        total_weight = self.clip_weight + self.xgb_weight
        vpa_weight = 0.3  # VPA gets 30% weight
        clip_adj = self.clip_weight / total_weight * (1 - vpa_weight)
        xgb_adj = self.xgb_weight / total_weight * (1 - vpa_weight)

        if method == "weighted_average":
            # Convert predictions to probabilities
            clip_probs = clip_result['probabilities']
            xgb_probs = xgb_result['probabilities']

            # VPA signal to probabilities (simplified)
            vpa_signal = vpa_result['signal']
            if vpa_signal == 'bullish':
                vpa_probs = np.array([0.8, 0.1, 0.1])  # bullish, bearish, neutral
            elif vpa_signal == 'bearish':
                vpa_probs = np.array([0.1, 0.8, 0.1])
            else:
                vpa_probs = np.array([0.33, 0.33, 0.34])

            # Ensure same shape
            if len(clip_probs.shape) > 1:
                clip_probs = clip_probs[0]
            if len(xgb_probs.shape) > 1:
                xgb_probs = xgb_probs[0]

            # Weighted average with VPA
            fused_probs = (clip_adj * clip_probs + xgb_adj * xgb_probs + vpa_weight * vpa_probs)
            fused_probs = fused_probs / fused_probs.sum()

            return {
                'prediction': np.argmax(fused_probs),
                'probabilities': fused_probs,
                'confidence': max(fused_probs),
                'fusion_method': 'weighted_average_vpa',
                'clip_weight': clip_adj,
                'xgb_weight': xgb_adj,
                'vpa_weight': vpa_weight
            }

        elif method == "confidence_weighted":
            # Similar logic but confidence-weighted
            clip_conf = clip_result['confidences'][0] if isinstance(clip_result['confidences'], np.ndarray) else clip_result['confidences']
            xgb_conf = xgb_result['confidence']
            vpa_conf = vpa_result['confidence']

            # Dynamic weights
            total_conf = clip_conf + xgb_conf + vpa_conf
            clip_w = clip_conf / total_conf if total_conf > 0 else 1/3
            xgb_w = xgb_conf / total_conf if total_conf > 0 else 1/3
            vpa_w = vpa_conf / total_conf if total_conf > 0 else 1/3

            # Apply weights (similar to above)
            clip_probs = clip_result['probabilities']
            xgb_probs = xgb_result['probabilities']
            vpa_signal = vpa_result['signal']
            if vpa_signal == 'bullish':
                vpa_probs = np.array([0.8, 0.1, 0.1])
            elif vpa_signal == 'bearish':
                vpa_probs = np.array([0.1, 0.8, 0.1])
            else:
                vpa_probs = np.array([0.33, 0.33, 0.34])

            if len(clip_probs.shape) > 1:
                clip_probs = clip_probs[0]
            if len(xgb_probs.shape) > 1:
                xgb_probs = xgb_probs[0]

            fused_probs = (clip_w * clip_probs + xgb_w * xgb_probs + vpa_w * vpa_probs)
            fused_probs = fused_probs / fused_probs.sum()

            return {
                'prediction': np.argmax(fused_probs),
                'probabilities': fused_probs,
                'confidence': max(fused_probs),
                'fusion_method': 'confidence_weighted_vpa',
                'clip_weight': clip_w,
                'xgb_weight': xgb_w,
                'vpa_weight': vpa_w
            }

        else:
            # Default to weighted average for stacking/other methods
            return self._fuse_three_predictions(clip_result, xgb_result, vpa_result, "weighted_average")

    def train_stacking_fusion(self,
                             training_data: List[Tuple[Dict[str, Any], Dict[str, Any], int]],
                             model_type: str = "rf"):
        """
        Train stacking fusion model

        Args:
            training_data: List of (clip_result, xgb_result, true_label) tuples
            model_type: Type of meta-model ("rf" for RandomForest, "lr" for LogisticRegression)
        """
        logger.info("[TRAIN] Training stacking fusion model...")

        features = []
        labels = []

        for clip_result, xgb_result, true_label in training_data:
            # Extract features
            clip_features = np.concatenate([
                clip_result['probabilities'].flatten(),
                [clip_result['confidences'][0] if isinstance(clip_result['confidences'], np.ndarray) else clip_result['confidences']]
            ])

            xgb_features = np.concatenate([
                xgb_result['probabilities'],
                [xgb_result['confidence']]
            ])

            combined_features = np.concatenate([clip_features, xgb_features])
            features.append(combined_features)
            labels.append(true_label)

        features = np.array(features)
        labels = np.array(labels)

        # Train meta-model
        if model_type == "rf":
            self.fusion_model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "lr":
            self.fusion_model = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.fusion_model.fit(features, labels)

        logger.info("[TRAIN] Stacking fusion model trained")

    def predict_multimodal(self,
                          chart_data: Any,
                          text_descriptions: List[str],
                          technical_features: Dict[str, Any],
                          vpa_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Make multimodal prediction combining CLIP, XGBoost, and VPA

        Args:
            chart_data: Chart image or path
            text_descriptions: List of pattern descriptions
            technical_features: Dictionary of technical indicators
            vpa_data: Optional DataFrame with VPA features

        Returns:
            Multimodal prediction results
        """
        # Get individual predictions
        clip_result = self.predict_clip(chart_data, text_descriptions)
        xgb_result = self.predict_xgb(technical_features)

        # VPA prediction if data provided
        vpa_result = None
        if vpa_data is not None:
            try:
                from volume.volume_price_action import VPAAnalyzer
                vpa_analyzer = VPAAnalyzer()
                vpa_result = vpa_analyzer.predict_vpa_signal(vpa_data)
            except ImportError:
                logger.warning("[WARNING] VPA module not available")

        # Fuse predictions
        if vpa_result:
            fused_result = self.fuse_predictions(clip_result, xgb_result, vpa_result)
        else:
            fused_result = self.fuse_predictions(clip_result, xgb_result)

        # Add individual results for reference
        fused_result['clip_result'] = clip_result
        fused_result['xgb_result'] = xgb_result
        if vpa_result:
            fused_result['vpa_result'] = vpa_result

        return fused_result

    def evaluate_fusion(self,
                        test_data: List[Tuple[Any, List[str], Dict[str, Any], int]],
                        method: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate fusion performance

        Args:
            test_data: List of (chart_data, text_descriptions, technical_features, true_label) tuples
            method: Fusion method override

        Returns:
            Dictionary with evaluation metrics
        """
        method = method or self.fusion_method
        predictions = []
        true_labels = []

        for chart_data, text_descriptions, technical_features, true_label in tqdm(test_data, desc="Evaluating fusion"):
            try:
                result = self.predict_multimodal(chart_data, text_descriptions, technical_features)
                predictions.append(result['prediction'])
                true_labels.append(true_label)
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                continue

        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fusion_method': method,
            'num_samples': len(predictions)
        }

        logger.info(f"[EVAL] Fusion Performance ({method}):")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")

        # Store performance
        self.performance_history.append({
            'timestamp': datetime.now(),
            'method': method,
            'metrics': metrics
        })

        return metrics

    def optimize_weights(self,
                        val_data: List[Tuple[Any, List[str], Dict[str, Any], int]],
                        weight_range: Tuple[float, float] = (0.0, 1.0),
                        step: float = 0.1):
        """
        Optimize fusion weights using validation data

        Args:
            val_data: Validation data
            weight_range: Range for weight optimization
            step: Step size for weight grid search
        """
        logger.info("[OPTIMIZE] Optimizing fusion weights...")

        best_accuracy = 0.0
        best_weights = (self.clip_weight, self.xgb_weight)

        # Grid search over weights
        for clip_w in np.arange(weight_range[0], weight_range[1] + step, step):
            xgb_w = 1.0 - clip_w

            # Temporarily set weights
            old_clip_w, old_xgb_w = self.clip_weight, self.xgb_weight
            self.clip_weight, self.xgb_weight = clip_w, xgb_w

            # Evaluate
            metrics = self.evaluate_fusion(val_data, method="weighted_average")
            accuracy = metrics['accuracy']

            # Restore weights
            self.clip_weight, self.xgb_weight = old_clip_w, old_xgb_w

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = (clip_w, xgb_w)

        # Set best weights
        self.clip_weight, self.xgb_weight = best_weights

        logger.info(f"[OPTIMIZE] Best weights found: CLIP={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, Accuracy={best_accuracy:.4f}")
    def temporal_fusion(self,
                       predictions_history: List[Dict[str, Any]],
                       time_window: int = 5) -> Dict[str, Any]:
        """
        Apply temporal fusion to smooth predictions over time

        Args:
            predictions_history: List of recent predictions
            time_window: Number of recent predictions to consider

        Returns:
            Temporally fused prediction
        """
        if len(predictions_history) < time_window:
            # Not enough history, return latest prediction
            return predictions_history[-1] if predictions_history else {}

        # Get recent predictions
        recent = predictions_history[-time_window:]

        # Extract probabilities
        probs_history = []
        for pred in recent:
            if 'probabilities' in pred:
                probs = pred['probabilities']
                if isinstance(probs, np.ndarray) and len(probs.shape) > 1:
                    probs = probs[0]  # Take first row
                probs_history.append(probs)

        if not probs_history:
            return recent[-1]

        # Temporal averaging with exponential decay
        weights = np.exp(np.linspace(-1, 0, len(probs_history)))
        weights = weights / weights.sum()

        # Weighted average of probabilities
        temporal_probs = np.average(probs_history, axis=0, weights=weights)
        temporal_probs = temporal_probs / temporal_probs.sum()

        return {
            'prediction': np.argmax(temporal_probs),
            'probabilities': temporal_probs,
            'confidence': max(temporal_probs),
            'fusion_method': 'temporal',
            'time_window': time_window,
            'num_predictions': len(probs_history)
        }

    def plot_performance_history(self, save_path: Optional[str] = None):
        """Plot fusion performance over time"""
        if not self.performance_history:
            logger.warning("No performance history available")
            return

        timestamps = [entry['timestamp'] for entry in self.performance_history]
        accuracies = [entry['metrics']['accuracy'] for entry in self.performance_history]
        f1_scores = [entry['metrics']['f1_score'] for entry in self.performance_history]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(timestamps, accuracies, 'b-o', label='Accuracy')
        plt.title('Fusion Accuracy Over Time')
        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(timestamps, f1_scores, 'r-o', label='F1-Score')
        plt.title('Fusion F1-Score Over Time')
        plt.xlabel('Time')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"[PLOT] Performance history saved to {save_path}")

        plt.show()

    def save_fusion_model(self, path: str):
        """Save fusion model and configuration"""
        config = {
            'fusion_method': self.fusion_method,
            'clip_weight': self.clip_weight,
            'xgb_weight': self.xgb_weight,
            'performance_history': self.performance_history,
            'timestamp': datetime.now().isoformat()
        }

        # Save configuration
        with open(f"{path}_config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)

        # Save fusion model if exists
        if self.fusion_model:
            import joblib
            joblib.dump(self.fusion_model, f"{path}_model.pkl")

        logger.info(f"[SAVE] Fusion model saved to {path}")

    def load_fusion_model(self, path: str):
        """Load fusion model and configuration"""
        # Load configuration
        with open(f"{path}_config.json", 'r') as f:
            config = json.load(f)

        self.fusion_method = config['fusion_method']
        self.clip_weight = config['clip_weight']
        self.xgb_weight = config['xgb_weight']
        self.performance_history = config.get('performance_history', [])

        # Load fusion model if exists
        model_path = f"{path}_model.pkl"
        if os.path.exists(model_path):
            import joblib
            self.fusion_model = joblib.load(model_path)

        logger.info(f"[LOAD] Fusion model loaded from {path}")

# Utility functions
def create_multimodal_fusion(clip_model_path: str,
                           xgb_model_path: str,
                           fusion_method: str = "weighted_average") -> MultimodalFusion:
    """
    Create and configure multimodal fusion instance

    Args:
        clip_model_path: Path to CLIP model
        xgb_model_path: Path to XGBoost model
        fusion_method: Fusion method

    Returns:
        Configured MultimodalFusion instance
    """
    fusion = MultimodalFusion(
        clip_model_path=clip_model_path,
        xgb_model_path=xgb_model_path,
        fusion_method=fusion_method
    )

    return fusion

def benchmark_fusion_methods(fusion: MultimodalFusion,
                           test_data: List[Tuple[Any, List[str], Dict[str, Any], int]]) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different fusion methods

    Args:
        fusion: MultimodalFusion instance
        test_data: Test data

    Returns:
        Dictionary with performance for each method
    """
    methods = ["weighted_average", "confidence_weighted"]
    if fusion.fusion_model:
        methods.append("stacking")

    results = {}

    for method in methods:
        logger.info(f"[BENCHMARK] Evaluating {method}...")
        metrics = fusion.evaluate_fusion(test_data, method=method)
        results[method] = metrics

    return results

if __name__ == "__main__":
    # Example usage
    fusion = create_multimodal_fusion(
        clip_model_path="vlm/models/best_model.pt",
        xgb_model_path="xgboost_trading_model.pkl",
        fusion_method="weighted_average"
    )

    # Example prediction
    # result = fusion.predict_multimodal(chart_image, text_descriptions, technical_features)
    # Example prediction
    # result = fusion.predict_multimodal(chart_image, text_descriptions, technical_features)
    # print(f"Fused prediction: {result['prediction']}, confidence: {result['confidence']:.4f}")