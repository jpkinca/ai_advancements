#!/usr/bin/env python3
"""
Candlestick Pattern Detection Model Training

This script provides tools for training YOLOv8 and Vision Transformer models
for candlestick pattern recognition using historical market data.

Features:
- Synthetic candlestick chart generation for training
- YOLOv8 fine-tuning for pattern detection
- Vision Transformer training for classification
- Data augmentation and preprocessing
- Model evaluation and validation
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import mplfinance as mpf
from PIL import Image
import cv2
import json
import yaml
from sklearn.model_selection import train_test_split

# AI/ML imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from transformers import (
        ViTForImageClassification,
        ViTImageProcessor,
        TrainingArguments,
        Trainer
    )
    from datasets import Dataset, DatasetDict
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CandlestickDataGenerator:
    """
    Generate synthetic candlestick chart data for model training
    """

    def __init__(self, output_dir: str = 'data/candlestick_training'):
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, 'images')
        self.label_dir = os.path.join(output_dir, 'labels')

        # Ensure directories exist
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        # Pattern definitions with generation rules
        self.pattern_generators = {
            'doji': self._generate_doji,
            'hammer': self._generate_hammer,
            'shooting_star': self._generate_shooting_star,
            'marubozu': self._generate_marubozu,
            'bullish_engulfing': self._generate_bullish_engulfing,
            'bearish_engulfing': self._generate_bearish_engulfing,
            'morning_star': self._generate_morning_star,
            'evening_star': self._generate_evening_star
        }

    def generate_training_dataset(self, num_samples: int = 1000,
                                image_size: Tuple[int, int] = (640, 640)) -> str:
        """
        Generate complete training dataset

        Args:
            num_samples: Number of training samples to generate
            image_size: Size of output images

        Returns:
            Path to dataset YAML configuration
        """
        logger.info(f"[DATA] Generating {num_samples} training samples")

        # Generate samples for each pattern
        samples_per_pattern = num_samples // len(self.pattern_generators)

        for pattern_name, generator in self.pattern_generators.items():
            logger.info(f"[DATA] Generating {samples_per_pattern} samples for {pattern_name}")

            for i in range(samples_per_pattern):
                try:
                    # Generate data and chart
                    data, bbox = generator()

                    # Create chart image
                    image_path = self._create_chart_image(data, pattern_name, i, image_size)

                    # Save label (YOLO format)
                    self._save_yolo_label(pattern_name, bbox, i, image_size)

                except Exception as e:
                    logger.warning(f"[WARNING] Failed to generate sample {i} for {pattern_name}: {e}")

        # Create dataset configuration
        config_path = self._create_dataset_config()
        logger.info(f"[DATA] Training dataset created at {config_path}")

        return config_path

    def _generate_doji(self) -> Tuple[pd.DataFrame, List[float]]:
        """Generate Doji pattern data"""
        # Doji: Open and Close very close, long wicks
        base_price = 100 + np.random.randn() * 10

        # Generate OHLC with doji characteristics
        data = []
        for i in range(20):  # 20 candlesticks
            if i == 19:  # Last candle is the doji
                high = base_price + np.random.uniform(2, 5)
                low = base_price - np.random.uniform(2, 5)
                open_price = base_price + np.random.uniform(-0.5, 0.5)
                close_price = open_price + np.random.uniform(-0.1, 0.1)  # Very close to open
            else:
                # Normal candles
                high = base_price + np.random.uniform(0.5, 2)
                low = base_price - np.random.uniform(0.5, 2)
                open_price = base_price + np.random.uniform(-1, 1)
                close_price = base_price + np.random.uniform(-1, 1)

            data.append({
                'date': datetime.now() - timedelta(days=20-i),
                'open': open_price,
                'high': max(open_price, close_price, high),
                'low': min(open_price, close_price, low),
                'close': close_price,
                'volume': np.random.lognormal(15, 1)
            })

            base_price = close_price

        df = pd.DataFrame(data)
        # Bounding box for the doji pattern (last candle)
        bbox = [0.9, 0.4, 0.95, 0.6]  # [x1, y1, x2, y2] normalized

        return df, bbox

    def _generate_hammer(self) -> Tuple[pd.DataFrame, List[float]]:
        """Generate Hammer pattern data"""
        # Hammer: Small body, long lower wick
        base_price = 100 + np.random.randn() * 10

        data = []
        for i in range(20):
            if i == 19:  # Hammer candle
                body_size = np.random.uniform(0.2, 0.8)
                direction = np.random.choice([-1, 1])  # Bullish or bearish hammer
                close_price = base_price + direction * body_size
                open_price = base_price
                high = max(open_price, close_price) + np.random.uniform(0.1, 0.5)
                low = min(open_price, close_price) - np.random.uniform(2, 4)  # Long lower wick
            else:
                # Normal candles
                high = base_price + np.random.uniform(0.5, 2)
                low = base_price - np.random.uniform(0.5, 2)
                open_price = base_price + np.random.uniform(-1, 1)
                close_price = base_price + np.random.uniform(-1, 1)

            data.append({
                'date': datetime.now() - timedelta(days=20-i),
                'open': open_price,
                'high': max(open_price, close_price, high),
                'low': min(open_price, close_price, low),
                'close': close_price,
                'volume': np.random.lognormal(15, 1)
            })

            base_price = close_price

        df = pd.DataFrame(data)
        bbox = [0.9, 0.3, 0.95, 0.7]  # Covers the hammer with long wick

        return df, bbox

    def _generate_shooting_star(self) -> Tuple[pd.DataFrame, List[float]]:
        """Generate Shooting Star pattern data"""
        # Shooting Star: Small body, long upper wick
        base_price = 100 + np.random.randn() * 10

        data = []
        for i in range(20):
            if i == 19:  # Shooting star candle
                body_size = np.random.uniform(0.2, 0.8)
                close_price = base_price + body_size
                open_price = base_price
                high = max(open_price, close_price) + np.random.uniform(2, 4)  # Long upper wick
                low = min(open_price, close_price) - np.random.uniform(0.1, 0.5)
            else:
                # Normal candles
                high = base_price + np.random.uniform(0.5, 2)
                low = base_price - np.random.uniform(0.5, 2)
                open_price = base_price + np.random.uniform(-1, 1)
                close_price = base_price + np.random.uniform(-1, 1)

            data.append({
                'date': datetime.now() - timedelta(days=20-i),
                'open': open_price,
                'high': max(open_price, close_price, high),
                'low': min(open_price, close_price, low),
                'close': close_price,
                'volume': np.random.lognormal(15, 1)
            })

            base_price = close_price

        df = pd.DataFrame(data)
        bbox = [0.9, 0.2, 0.95, 0.6]  # Covers the shooting star

        return df, bbox

    def _generate_marubozu(self) -> Tuple[pd.DataFrame, List[float]]:
        """Generate Marubozu pattern data"""
        # Marubozu: No wicks, full body
        base_price = 100 + np.random.randn() * 10

        data = []
        for i in range(20):
            if i == 19:  # Marubozu candle
                direction = np.random.choice([-1, 1])
                body_size = np.random.uniform(2, 4)
                if direction > 0:  # Bullish
                    open_price = base_price
                    close_price = base_price + body_size
                    high = close_price
                    low = open_price
                else:  # Bearish
                    open_price = base_price
                    close_price = base_price - body_size
                    high = open_price
                    low = close_price
            else:
                # Normal candles
                high = base_price + np.random.uniform(0.5, 2)
                low = base_price - np.random.uniform(0.5, 2)
                open_price = base_price + np.random.uniform(-1, 1)
                close_price = base_price + np.random.uniform(-1, 1)

            data.append({
                'date': datetime.now() - timedelta(days=20-i),
                'open': open_price,
                'high': max(open_price, close_price, high),
                'low': min(open_price, close_price, low),
                'close': close_price,
                'volume': np.random.lognormal(15, 1)
            })

            base_price = close_price

        df = pd.DataFrame(data)
        bbox = [0.9, 0.4, 0.95, 0.6]

        return df, bbox

    def _generate_bullish_engulfing(self) -> Tuple[pd.DataFrame, List[float]]:
        """Generate Bullish Engulfing pattern data"""
        base_price = 100 + np.random.randn() * 10

        data = []
        for i in range(20):
            if i == 18:  # First candle (small bearish)
                open_price = base_price + 1
                close_price = base_price - 0.5
                high = open_price + 0.5
                low = close_price - 0.5
            elif i == 19:  # Second candle (large bullish engulfing)
                open_price = close_price - 0.2  # Gap down
                close_price = open_price + 3  # Large bullish body
                high = close_price + 0.5
                low = open_price - 0.5
            else:
                # Normal candles
                high = base_price + np.random.uniform(0.5, 2)
                low = base_price - np.random.uniform(0.5, 2)
                open_price = base_price + np.random.uniform(-1, 1)
                close_price = base_price + np.random.uniform(-1, 1)

            data.append({
                'date': datetime.now() - timedelta(days=20-i),
                'open': open_price,
                'high': max(open_price, close_price, high),
                'low': min(open_price, close_price, low),
                'close': close_price,
                'volume': np.random.lognormal(15, 1)
            })

            base_price = close_price

        df = pd.DataFrame(data)
        bbox = [0.85, 0.3, 0.95, 0.7]  # Covers both engulfing candles

        return df, bbox

    def _generate_bearish_engulfing(self) -> Tuple[pd.DataFrame, List[float]]:
        """Generate Bearish Engulfing pattern data"""
        base_price = 100 + np.random.randn() * 10

        data = []
        for i in range(20):
            if i == 18:  # First candle (small bullish)
                open_price = base_price - 1
                close_price = base_price + 0.5
                high = close_price + 0.5
                low = open_price - 0.5
            elif i == 19:  # Second candle (large bearish engulfing)
                open_price = close_price + 0.2  # Gap up
                close_price = open_price - 3  # Large bearish body
                high = open_price + 0.5
                low = close_price - 0.5
            else:
                # Normal candles
                high = base_price + np.random.uniform(0.5, 2)
                low = base_price - np.random.uniform(0.5, 2)
                open_price = base_price + np.random.uniform(-1, 1)
                close_price = base_price + np.random.uniform(-1, 1)

            data.append({
                'date': datetime.now() - timedelta(days=20-i),
                'open': open_price,
                'high': max(open_price, close_price, high),
                'low': min(open_price, close_price, low),
                'close': close_price,
                'volume': np.random.lognormal(15, 1)
            })

            base_price = close_price

        df = pd.DataFrame(data)
        bbox = [0.85, 0.3, 0.95, 0.7]

        return df, bbox

    def _generate_morning_star(self) -> Tuple[pd.DataFrame, List[float]]:
        """Generate Morning Star pattern data"""
        base_price = 100 + np.random.randn() * 10

        data = []
        for i in range(20):
            if i == 17:  # First candle (large bearish)
                open_price = base_price + 2
                close_price = base_price - 1
                high = open_price + 0.5
                low = close_price - 0.5
            elif i == 18:  # Second candle (small, star)
                open_price = close_price - 0.3
                close_price = open_price + np.random.uniform(-0.2, 0.2)
                high = open_price + 1
                low = close_price - 1
            elif i == 19:  # Third candle (large bullish)
                open_price = close_price + 0.3
                close_price = open_price + 2.5
                high = close_price + 0.5
                low = open_price - 0.5
            else:
                # Normal candles
                high = base_price + np.random.uniform(0.5, 2)
                low = base_price - np.random.uniform(0.5, 2)
                open_price = base_price + np.random.uniform(-1, 1)
                close_price = base_price + np.random.uniform(-1, 1)

            data.append({
                'date': datetime.now() - timedelta(days=20-i),
                'open': open_price,
                'high': max(open_price, close_price, high),
                'low': min(open_price, close_price, low),
                'close': close_price,
                'volume': np.random.lognormal(15, 1)
            })

            base_price = close_price

        df = pd.DataFrame(data)
        bbox = [0.8, 0.2, 0.95, 0.8]  # Covers the three-star pattern

        return df, bbox

    def _generate_evening_star(self) -> Tuple[pd.DataFrame, List[float]]:
        """Generate Evening Star pattern data"""
        base_price = 100 + np.random.randn() * 10

        data = []
        for i in range(20):
            if i == 17:  # First candle (large bullish)
                open_price = base_price - 2
                close_price = base_price + 1
                high = close_price + 0.5
                low = open_price - 0.5
            elif i == 18:  # Second candle (small, star)
                open_price = close_price + 0.3
                close_price = open_price + np.random.uniform(-0.2, 0.2)
                high = open_price + 1
                low = close_price - 1
            elif i == 19:  # Third candle (large bearish)
                open_price = close_price - 0.3
                close_price = open_price - 2.5
                high = open_price + 0.5
                low = close_price - 0.5
            else:
                # Normal candles
                high = base_price + np.random.uniform(0.5, 2)
                low = base_price - np.random.uniform(0.5, 2)
                open_price = base_price + np.random.uniform(-1, 1)
                close_price = base_price + np.random.uniform(-1, 1)

            data.append({
                'date': datetime.now() - timedelta(days=20-i),
                'open': open_price,
                'high': max(open_price, close_price, high),
                'low': min(open_price, close_price, low),
                'close': close_price,
                'volume': np.random.lognormal(15, 1)
            })

            base_price = close_price

        df = pd.DataFrame(data)
        bbox = [0.8, 0.2, 0.95, 0.8]

        return df, bbox

    def _create_chart_image(self, df: pd.DataFrame, pattern_name: str,
                           sample_id: int, image_size: Tuple[int, int]) -> str:
        """Create candlestick chart image"""
        # Set up chart style
        mc = mpf.make_marketcolors(up='green', down='red', edge='black', wick='black', volume='in')
        s = mpf.make_mpf_style(marketcolors=mc, style='charles')

        # Generate filename
        filename = f"{pattern_name}_{sample_id:04d}.png"
        filepath = os.path.join(self.image_dir, filename)

        # Generate chart
        fig, axlist = mpf.plot(df.set_index('date'), type='candle', volume=True,
                              style=s, warn_too_much_data=1000,
                              savefig=dict(fname=filepath, dpi=150, bbox_inches='tight'),
                              figsize=(10, 6))

        plt.close('all')

        return filepath

    def _save_yolo_label(self, pattern_name: str, bbox: List[float],
                        sample_id: int, image_size: Tuple[int, int]):
        """Save YOLO format label file"""
        # Map pattern to class ID
        class_mapping = {
            'doji': 0, 'hammer': 1, 'shooting_star': 2, 'marubozu': 3,
            'bullish_engulfing': 4, 'bearish_engulfing': 5,
            'morning_star': 6, 'evening_star': 7
        }

        class_id = class_mapping.get(pattern_name, 0)

        # YOLO format: class_id x_center y_center width height (normalized)
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        # Save label file
        label_filename = f"{pattern_name}_{sample_id:04d}.txt"
        label_path = os.path.join(self.label_dir, label_filename)

        with open(label_path, 'w') as f:
            f.write(label_line)

    def _create_dataset_config(self) -> str:
        """Create YOLO dataset configuration YAML"""
        config = {
            'path': os.path.abspath(self.output_dir),
            'train': 'images',
            'val': 'images',  # Using same data for now
            'names': {
                0: 'doji', 1: 'hammer', 2: 'shooting_star', 3: 'marubozu',
                4: 'bullish_engulfing', 5: 'bearish_engulfing',
                6: 'morning_star', 7: 'evening_star'
            }
        }

        config_path = os.path.join(self.output_dir, 'data.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_path

class CandlestickModelTrainer:
    """
    Train YOLO and ViT models for candlestick pattern recognition
    """

    def __init__(self):
        self.yolo_model = None
        self.vit_model = None
        self.vit_processor = None

    def train_yolo_model(self, data_config: str, epochs: int = 50,
                        model_size: str = 'yolov8n') -> str:
        """
        Train YOLOv8 model for pattern detection

        Args:
            data_config: Path to data.yaml
            epochs: Number of training epochs
            model_size: YOLO model size (n, s, m, l, x)

        Returns:
            Path to trained model
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 not available. Install with: pip install ultralytics")

        logger.info(f"[TRAIN] Starting YOLO training with {model_size} for {epochs} epochs")

        # Load model
        self.yolo_model = YOLO(f'{model_size}.pt')

        # Train model
        results = self.yolo_model.train(
            data=data_config,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name='candlestick_yolo',
            save=True,
            save_period=10
        )

        # Get best model path
        model_path = results.save_dir / 'weights' / 'best.pt'
        logger.info(f"[TRAIN] YOLO training completed. Best model: {model_path}")

        return str(model_path)

    def prepare_vit_dataset(self, image_dir: str, label_file: str = None) -> DatasetDict:
        """
        Prepare dataset for ViT training

        Args:
            image_dir: Directory containing images
            label_file: Optional label mapping file

        Returns:
            HuggingFace DatasetDict
        """
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace datasets not available")

        # This is a simplified implementation
        # In practice, you'd load images and create proper dataset
        logger.info("[DATA] Preparing ViT dataset (simplified)")

        # Placeholder - would need actual image loading and labeling
        return None

    def train_vit_model(self, dataset: DatasetDict, num_classes: int = 8,
                       output_dir: str = 'models/candlestick_vit') -> str:
        """
        Train Vision Transformer for pattern classification

        Args:
            dataset: Training dataset
            num_classes: Number of pattern classes
            output_dir: Output directory for model

        Returns:
            Path to trained model
        """
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers not available")

        logger.info(f"[TRAIN] Starting ViT training for {num_classes} classes")

        # Load model and processor
        self.vit_model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_classes
        )
        self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            save_steps=500,
            save_total_limit=2,
            evaluation_strategy="steps",
            logging_steps=100,
            load_best_model_at_end=True,
        )

        # Trainer
        trainer = Trainer(
            model=self.vit_model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
        )

        # Train
        trainer.train()

        # Save model
        model_path = os.path.join(output_dir, 'final_model')
        trainer.save_model(model_path)

        logger.info(f"[TRAIN] ViT training completed. Model saved to {model_path}")
        return model_path

# Utility functions

def create_training_pipeline(num_samples: int = 1000) -> Dict[str, str]:
    """
    Complete training pipeline from data generation to model training

    Args:
        num_samples: Number of training samples

    Returns:
        Dictionary with paths to trained models
    """
    logger.info("[PIPELINE] Starting candlestick training pipeline")

    # Generate training data
    data_gen = CandlestickDataGenerator()
    data_config = data_gen.generate_training_dataset(num_samples)

    # Train YOLO model
    trainer = CandlestickModelTrainer()
    yolo_model_path = trainer.train_yolo_model(data_config, epochs=30)

    # Note: ViT training would require more complex dataset preparation
    # vit_model_path = trainer.train_vit_model(vit_dataset)

    results = {
        'data_config': data_config,
        'yolo_model': yolo_model_path,
        # 'vit_model': vit_model_path
    }

    logger.info("[PIPELINE] Training pipeline completed")
    return results

if __name__ == "__main__":
    # Test data generation
    generator = CandlestickDataGenerator()

    # Generate small dataset for testing
    config_path = generator.generate_training_dataset(num_samples=50)
    print(f"Dataset config created: {config_path}")

    # Test training (if YOLO available)
    if YOLO_AVAILABLE:
        trainer = CandlestickModelTrainer()
        try:
            model_path = trainer.train_yolo_model(config_path, epochs=5)
            print(f"YOLO model trained: {model_path}")
        except Exception as e:
            print(f"Training failed: {e}")
    else:
        print("YOLO not available for training")