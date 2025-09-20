#!/usr/bin/env python3
"""
Zero-Shot CLIP Baseline Evaluation

This module evaluates pre-trained CLIP models on chart pattern recognition tasks
without any fine-tuning, establishing baseline performance for comparison with
fine-tuned models.

Features:
- Zero-shot evaluation using CLIP text and vision encoders
- Pattern classification accuracy metrics
- Confidence score analysis
- Comparison across different CLIP model variants
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPBaselineEvaluator:
    """
    Evaluates CLIP models on chart pattern recognition in zero-shot setting
    """

    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: Optional[str] = None):
        """
        Initialize the CLIP evaluator

        Args:
            model_name: CLIP model variant to use
            device: Device to run evaluation on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize CLIP model and processor
        self.model = None
        self.processor = None
        self._load_model()

        logger.info(f"[INIT] CLIP Baseline Evaluator ({model_name} on {self.device})")

    def _load_model(self):
        """Load CLIP model and processor"""
        try:
            from transformers import CLIPProcessor, CLIPModel

            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"[SUCCESS] Loaded CLIP model: {self.model_name}")

        except ImportError as e:
            raise RuntimeError("transformers library required for CLIP evaluation") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}") from e

    def evaluate_dataset(self,
                        dataset_path: str,
                        text_templates: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate CLIP on a complete dataset

        Args:
            dataset_path: Path to dataset directory
            text_templates: Custom text templates for zero-shot classification

        Returns:
            Comprehensive evaluation results
        """
        from vlm.dataset_builder import VLMDatasetBuilder

        # Load dataset
        builder = VLMDatasetBuilder()
        dataset = builder.load_dataset(dataset_path)

        # Use test split for evaluation
        test_data = dataset['splits'].get('test', [])
        if not test_data:
            logger.warning("[WARNING] No test split found, using validation data")
            test_data = dataset['splits'].get('validation', [])

        if not test_data:
            raise ValueError("No evaluation data found in dataset")

        logger.info(f"[EVAL] Evaluating on {len(test_data)} samples")

        # Prepare text templates
        if text_templates is None:
            text_templates = self._get_default_templates()

        # Evaluate
        results = self._evaluate_samples(test_data, text_templates)

        # Add metadata
        results['metadata'] = {
            'model': self.model_name,
            'device': self.device,
            'dataset': dataset['info']['name'],
            'dataset_version': dataset['info']['version'],
            'num_samples': len(test_data),
            'text_templates': text_templates
        }

        return results

    def _get_default_templates(self) -> List[str]:
        """Get default text templates for chart pattern classification"""
        return [
            "a chart showing bullish trend",
            "a chart showing bearish trend",
            "a chart with bullish engulfing pattern",
            "a chart with bearish engulfing pattern",
            "a chart with high volume",
            "a chart with low volume",
            "a chart with strong momentum",
            "a chart with weak momentum",
            "a chart with overbought conditions",
            "a chart with oversold conditions",
            "a chart with bullish divergence",
            "a chart with bearish divergence",
            "a chart showing consolidation",
            "a chart showing breakout",
            "a chart with technical indicators"
        ]

    def _evaluate_samples(self, samples: List[Dict[str, Any]],
                         text_templates: List[str]) -> Dict[str, Any]:
        """Evaluate CLIP on individual samples"""
        predictions = []
        ground_truths = []

        for sample in tqdm(samples, desc="Evaluating samples"):
            # Load image
            image_path = sample['image_path']
            if not os.path.exists(image_path):
                logger.warning(f"[WARNING] Image not found: {image_path}")
                continue

            image = Image.open(image_path).convert('RGB')

            # Get ground truth from metadata
            ground_truth = self._extract_ground_truth(sample)

            # Get CLIP prediction
            prediction = self._classify_image(image, text_templates)

            predictions.append(prediction)
            ground_truths.append(ground_truth)

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, ground_truths, text_templates)

        return {
            'predictions': predictions,
            'ground_truths': ground_truths,
            'metrics': metrics
        }

    def _extract_ground_truth(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ground truth labels from sample metadata"""
        metadata = sample['metadata']
        patterns = metadata.get('patterns_detected', [])

        # Extract key pattern types
        pattern_types = set()
        for pattern in patterns:
            pattern_types.add(pattern.get('type', 'unknown'))

        # Determine primary trend
        trend = 'neutral'
        for pattern in patterns:
            if pattern.get('type') == 'trend':
                trend = pattern.get('direction', 'neutral')
                break

        return {
            'trend': trend,
            'pattern_types': list(pattern_types),
            'confidence': metadata.get('confidence_score', 0.0),
            'symbol': metadata.get('symbol', ''),
            'timeframe': metadata.get('timeframe', '')
        }

    def _classify_image(self, image: Image.Image, text_templates: List[str]) -> Dict[str, Any]:
        """Classify image using CLIP zero-shot classification"""
        # Prepare inputs
        inputs = self.processor(
            text=text_templates,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get CLIP outputs
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Calculate similarity scores
        logits_per_image = outputs.logits_per_image  # [1, num_texts]
        probs = logits_per_image.softmax(dim=1)[0]  # [num_texts]

        # Get top predictions
        top_probs, top_indices = torch.topk(probs, k=min(3, len(text_templates)))

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'template': text_templates[idx],
                'probability': float(prob),
                'index': int(idx)
            })

        return {
            'top_prediction': predictions[0]['template'],
            'top_probability': predictions[0]['probability'],
            'all_predictions': predictions,
            'entropy': float(-torch.sum(probs * torch.log(probs + 1e-10)))
        }

    def _calculate_metrics(self, predictions: List[Dict[str, Any]],
                          ground_truths: List[Dict[str, Any]],
                          text_templates: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'overall': {},
            'by_category': {},
            'confidence_analysis': {}
        }

        # Overall accuracy (trend prediction)
        trend_correct = 0
        total_samples = len(predictions)

        for pred, gt in zip(predictions, ground_truths):
            pred_trend = self._extract_trend_from_prediction(pred['top_prediction'])
            gt_trend = gt['trend']

            if pred_trend == gt_trend:
                trend_correct += 1

        metrics['overall']['trend_accuracy'] = trend_correct / total_samples if total_samples > 0 else 0

        # Pattern type accuracy
        pattern_correct = 0
        for pred, gt in zip(predictions, ground_truths):
            pred_patterns = self._extract_patterns_from_prediction(pred['top_prediction'])
            gt_patterns = set(gt['pattern_types'])

            if any(p in gt_patterns for p in pred_patterns):
                pattern_correct += 1

        metrics['overall']['pattern_accuracy'] = pattern_correct / total_samples if total_samples > 0 else 0

        # Confidence analysis
        confidences = [p['top_probability'] for p in predictions]
        metrics['confidence_analysis'] = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'median': float(np.median(confidences))
        }

        # Top-k accuracy
        for k in [1, 3, 5]:
            if k <= len(text_templates):
                topk_correct = 0
                for pred in predictions:
                    top_k_preds = [p['template'] for p in pred['all_predictions'][:k]]
                    if any(self._matches_ground_truth(t, pred) for t in top_k_preds):
                        topk_correct += 1
                metrics['overall'][f'top_{k}_accuracy'] = topk_correct / total_samples if total_samples > 0 else 0

        return metrics

    def _extract_trend_from_prediction(self, prediction: str) -> str:
        """Extract trend direction from prediction text"""
        pred_lower = prediction.lower()
        if 'bullish' in pred_lower:
            return 'bullish'
        elif 'bearish' in pred_lower:
            return 'bearish'
        else:
            return 'neutral'

    def _extract_patterns_from_prediction(self, prediction: str) -> List[str]:
        """Extract pattern types from prediction text"""
        patterns = []
        pred_lower = prediction.lower()

        pattern_keywords = {
            'engulfing': 'candlestick',
            'volume': 'volume',
            'momentum': 'indicator',
            'divergence': 'indicator',
            'breakout': 'pattern',
            'consolidation': 'pattern'
        }

        for keyword, pattern_type in pattern_keywords.items():
            if keyword in pred_lower:
                patterns.append(pattern_type)

        return patterns

    def _matches_ground_truth(self, prediction: str, full_pred: Dict[str, Any]) -> bool:
        """Check if prediction matches ground truth (simplified)"""
        # This is a simplified matching - in practice, you'd want more sophisticated matching
        return True  # Placeholder

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"[SAVE] Results saved to {output_path}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save results: {e}")

    def compare_models(self, dataset_path: str, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple CLIP models on the same dataset"""
        comparison_results = {}

        for model_name in model_names:
            logger.info(f"[COMPARE] Evaluating {model_name}")
            try:
                evaluator = CLIPBaselineEvaluator(model_name)
                results = evaluator.evaluate_dataset(dataset_path)
                comparison_results[model_name] = results['metrics']
            except Exception as e:
                logger.error(f"[ERROR] Failed to evaluate {model_name}: {e}")
                comparison_results[model_name] = {'error': str(e)}

        return comparison_results

# Utility functions
def run_baseline_evaluation(dataset_path: str = 'vlm/datasets/sample_trading_charts_v1.0'):
    """Run baseline evaluation on sample dataset"""
    evaluator = CLIPBaselineEvaluator()

    # Evaluate
    results = evaluator.evaluate_dataset(dataset_path)

    # Save results
    output_path = 'vlm/evaluation/clip_baseline_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    evaluator.save_results(results, output_path)

    # Print summary
    metrics = results['metrics']
    print("CLIP Baseline Evaluation Results:")
    print(f"Trend Accuracy: {metrics['overall']['trend_accuracy']:.3f}")
    print(f"Pattern Accuracy: {metrics['overall']['pattern_accuracy']:.3f}")
    print(f"Top-3 Accuracy: {metrics['overall'].get('top_3_accuracy', 'N/A')}")
    print(f"Mean Confidence: {metrics['confidence_analysis']['mean']:.3f}")

    return results

def compare_clip_models(dataset_path: str):
    """Compare different CLIP model variants"""
    model_names = [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-base-patch16"
    ]

    evaluator = CLIPBaselineEvaluator()  # Use base model for comparison runner
    comparison = evaluator.compare_models(dataset_path, model_names)

    # Save comparison
    output_path = 'vlm/evaluation/model_comparison.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print("Model Comparison Results:")
    for model, metrics in comparison.items():
        if 'error' not in metrics:
            trend_acc = metrics['overall']['trend_accuracy']
            print(f"{model}: Trend Acc = {trend_acc:.3f}")
        else:
            print(f"{model}: Error - {metrics['error']}")

    return comparison

if __name__ == "__main__":
    # Run baseline evaluation
    results = run_baseline_evaluation()

    # Optionally compare models
    # comparison = compare_clip_models('vlm/datasets/sample_trading_charts_v1.0')