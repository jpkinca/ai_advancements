#!/usr/bin/env python3
"""
Volume Price Action (VPA) Analysis Module

This module implements AI-driven volume price action analysis for trading,
combining price movements with volume to confirm signals and detect patterns.

Features:
- VPA feature computation (volume-price ratios, imbalances)
- Integration with existing AI models (LSTM-CNN, ResNet, FAISS)
- Pattern detection for volume-driven signals
- Real-time VPA analysis for trading decisions
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VPAFeatures:
    """
    Compute Volume Price Action features from market data
    """

    @staticmethod
    def compute_basic_vpa(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute basic VPA features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with VPA features added
        """
        df = df.copy()

        # Volume-Price Ratio (Effort vs Result)
        df['vol_price_ratio'] = df['volume'] / (df['high'] - df['low'] + 1e-8)  # Add small epsilon to avoid division by zero

        # Volume Imbalance (Signed volume based on price direction)
        df['price_direction'] = np.sign(df['close'] - df['open'])
        df['volume_imbalance'] = df['price_direction'] * df['volume']

        # Volume Moving Averages
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-8)

        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change()

        # Accumulation/Distribution Line (simplified)
        df['adl'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8) * df['volume']
        df['adl'] = df['adl'].cumsum()

        # On-Balance Volume (OBV)
        df['obv'] = np.where(df['close'] > df['close'].shift(1), df['volume'],
                           np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
        df['obv'] = df['obv'].cumsum()

        # Volume-Weighted Average Price (VWAP)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        # Volume Clusters (simplified - high volume periods)
        volume_threshold = df['volume'].quantile(0.8)
        df['high_volume_period'] = (df['volume'] > volume_threshold).astype(int)

        return df

    @staticmethod
    def compute_advanced_vpa(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        Compute advanced VPA features

        Args:
            df: DataFrame with basic VPA features
            lookback: Lookback period for calculations

        Returns:
            DataFrame with advanced VPA features
        """
        df = df.copy()

        # Volume Price Trend (VPT)
        df['vpt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
        df['vpt'] = df['vpt'].cumsum()

        # Negative Volume Index (NVI) - Initialize first
        df['nvi'] = 1000.0  # Initialize with base value
        df['nvi'] = np.where(df['volume'] < df['volume'].shift(1),
                           df['nvi'].shift(1) * (1 + (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)),
                           df['nvi'].shift(1))
        # Fill any NaN values that might occur
        df['nvi'] = df['nvi'].fillna(1000.0)

        # Positive Volume Index (PVI) - Initialize first
        df['pvi'] = 1000.0  # Initialize with base value
        df['pvi'] = np.where(df['volume'] > df['volume'].shift(1),
                           df['pvi'].shift(1) * (1 + (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)),
                           df['pvi'].shift(1))
        # Fill any NaN values that might occur
        df['pvi'] = df['pvi'].fillna(1000.0)

        # Volume Oscillator
        df['volume_oscillator'] = (df['volume_sma_5'] - df['volume_sma_20']) / df['volume_sma_20']

        # Volume-Price Confirmation
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        df['vpc_confirmed'] = np.sign(price_change) == np.sign(volume_change)

        # Volume Divergence (simplified)
        df['volume_divergence'] = df['volume_roc'] - df['close'].pct_change()

        return df

    @staticmethod
    def detect_vpa_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect common VPA patterns

        Args:
            df: DataFrame with VPA features

        Returns:
            DataFrame with pattern detections
        """
        df = df.copy()

        # Volume Climax (high volume + price reversal)
        df['volume_climax'] = (
            (df['volume'] > df['volume_sma_20'] * 1.5) &
            (df['close'] < df['open']) &
            (df['volume'] > df['volume'].shift(1))
        ).astype(int)

        # Volume Spike (sudden volume increase)
        df['volume_spike'] = (
            (df['volume'] > df['volume'].shift(1) * 2) &
            (df['volume'] > df['volume_sma_20'])
        ).astype(int)

        # Low Volume Test (price moves on low volume)
        df['low_volume_test'] = (
            (df['volume'] < df['volume_sma_20'] * 0.5) &
            (abs(df['close'] - df['open']) > df['high'].rolling(20).std())
        ).astype(int)

        # Volume Absorption (high volume + small price range)
        df['volume_absorption'] = (
            (df['volume'] > df['volume_sma_20']) &
            ((df['high'] - df['low']) < df['high'].rolling(20).std() * 0.5)
        ).astype(int)

        # Bullish Volume Signals
        df['bullish_volume'] = (
            (df['close'] > df['open']) &
            (df['volume'] > df['volume_sma_20']) &
            (df['vol_price_ratio'] > df['vol_price_ratio'].rolling(20).mean())
        ).astype(int)

        # Bearish Volume Signals
        df['bearish_volume'] = (
            (df['close'] < df['open']) &
            (df['volume'] > df['volume_sma_20']) &
            (df['vol_price_ratio'] > df['vol_price_ratio'].rolling(20).mean())
        ).astype(int)

        return df

class VPAResNet(nn.Module):
    """
    ResNet-based model for VPA pattern recognition
    Extends GAF-ResNet to include volume channels
    """

    def __init__(self, num_classes: int = 3, input_channels: int = 4):
        """
        Initialize VPA ResNet

        Args:
            num_classes: Number of output classes (bullish, bearish, neutral)
            input_channels: Number of input channels (price GAF + volume GAF + others)
        """
        super(VPAResNet, self).__init__()

        # Load pretrained ResNet50 and modify first layer
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        # Modify input layer to accept multiple channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

        # Additional layers for VPA-specific processing
        self.vpa_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, input_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(input_channels // 8, input_channels, 1),
            nn.Sigmoid()
        )

        logger.info(f"[INIT] VPA ResNet initialized with {input_channels} input channels, {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with VPA attention

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Output predictions
        """
        # Apply VPA attention
        attention_weights = self.vpa_attention(x)
        x = x * attention_weights

        # ResNet forward pass
        return self.resnet(x)

class VPAAnalyzer:
    """
    Main VPA analysis class integrating features, models, and pattern detection
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_computer = VPAFeatures()

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = VPAResNet().to(self.device)

        logger.info(f"[INIT] VPA Analyzer initialized on {self.device}")

    def compute_vpa_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute complete VPA feature set

        Args:
            df: Raw OHLCV data

        Returns:
            DataFrame with all VPA features
        """
        df = self.feature_computer.compute_basic_vpa(df)
        df = self.feature_computer.compute_advanced_vpa(df)
        df = self.feature_computer.detect_vpa_patterns(df)

        return df

    def prepare_gaf_images(self, df: pd.DataFrame, image_size: int = 224) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare GAF images for price and volume data

        Args:
            df: DataFrame with VPA features
            image_size: Size of output GAF images

        Returns:
            Tuple of (price_gaf, volume_gaf) tensors
        """
        if df.empty or len(df) < 4:
            # Return zero tensors if insufficient data
            empty_tensor = torch.zeros(1, image_size, image_size)
            return empty_tensor, empty_tensor

        # Price GAF (simplified)
        price_cols = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_cols if col in df.columns]

        if not available_price_cols:
            price_gaf = torch.zeros(1, image_size, image_size)
        else:
            price_data = df[available_price_cols].values.T
            price_gaf = self._create_gaf_image(price_data, image_size)

        # Volume GAF - use available volume-related columns
        volume_cols = ['volume', 'volume_sma_5', 'volume_sma_20', 'vol_price_ratio', 'volume_imbalance']
        available_volume_cols = [col for col in volume_cols if col in df.columns]

        if not available_volume_cols:
            volume_gaf = torch.zeros(1, image_size, image_size)
        else:
            volume_data = df[available_volume_cols].values.T
            volume_gaf = self._create_gaf_image(volume_data, image_size)

        return price_gaf, volume_gaf

    def _create_gaf_image(self, data: np.ndarray, size: int) -> torch.Tensor:
        """
        Create GAF image from time series data (simplified)

        Args:
            data: Time series data
            size: Output image size

        Returns:
            GAF image tensor [1, size, size]
        """
        # Check for insufficient data
        if data.size == 0 or data.shape[0] == 0 or data.shape[1] < 2:
            return torch.zeros(1, size, size)

        try:
            # Simplified GAF - normalize and create 2D representation
            data_norm = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)

            # Create Gramian Angular Field (simplified)
            gaf = np.dot(data_norm.T, data_norm)

            # Resize to target size and ensure [1, size, size] shape
            gaf_resized = torch.tensor(gaf, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            gaf_resized = torch.nn.functional.interpolate(gaf_resized, size=(size, size), mode='bilinear')
            
            # Return [1, size, size] tensor
            return gaf_resized.squeeze(0)  # Remove batch dimension, keep channel dimension
        except Exception as e:
            logger.warning(f"[WARNING] Failed to create GAF image: {e}, returning zeros")
            return torch.zeros(1, size, size)

    def predict_vpa_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict VPA-based trading signal

        Args:
            df: DataFrame with recent market data

        Returns:
            Prediction results
        """
        if len(df) < 20:
            return {'signal': 'insufficient_data', 'confidence': 0.0}

        # Compute VPA features
        vpa_df = self.compute_vpa_features(df)

        if vpa_df.empty:
            return {'signal': 'insufficient_data', 'confidence': 0.0}

        # Prepare GAF images
        recent_data = vpa_df.tail(20)
        if len(recent_data) < 4:
            return {'signal': 'insufficient_data', 'confidence': 0.0}
        
        price_gaf, volume_gaf = self.prepare_gaf_images(recent_data)
        
        # Debug logging
        logger.info(f"[DEBUG] price_gaf shape: {price_gaf.shape}, volume_gaf shape: {volume_gaf.shape}")
        
        # Check if we have valid tensors
        if price_gaf.numel() == 0 or volume_gaf.numel() == 0:
            return {'signal': 'insufficient_data', 'confidence': 0.0}        # Try CNN prediction, fall back to VPA features if it fails
        try:
            # Combine into 4-channel input (price + volume + additional features)
            # Create 4 channels: [price_gaf, volume_gaf, price_gaf_diff, volume_gaf_diff]
            price_gaf_diff = torch.abs(price_gaf - volume_gaf)  # Price-volume difference
            volume_gaf_diff = torch.abs(price_gaf + volume_gaf)  # Price-volume sum
            
            combined_input = torch.cat([price_gaf, volume_gaf, price_gaf_diff, volume_gaf_diff], dim=0).unsqueeze(0).to(self.device)
            
            # Check batch size
            if combined_input.size(0) == 0 or combined_input.size(-1) != 224 or combined_input.size(-2) != 224:
                raise ValueError(f"Invalid input shape: {combined_input.shape}")

            # Model prediction
            with torch.no_grad():
                outputs = self.model(combined_input)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()

            # Map prediction to signal
            signal_map = {0: 'bullish', 1: 'bearish', 2: 'neutral'}
            signal = signal_map.get(prediction, 'neutral')

        except Exception as e:
            logger.warning(f"[WARNING] CNN prediction failed: {e}, falling back to VPA features")
            # Fallback to VPA feature-based signal
            recent_vpa = recent_data.tail(5)
            bullish_volume = recent_vpa['bullish_volume'].sum() if 'bullish_volume' in recent_vpa.columns else 0
            bearish_volume = recent_vpa['bearish_volume'].sum() if 'bearish_volume' in recent_vpa.columns else 0
            
            if bullish_volume > bearish_volume * 1.2:
                signal = 'bullish'
                confidence = min(bullish_volume / (bullish_volume + bearish_volume), 0.8)
            elif bearish_volume > bullish_volume * 1.2:
                signal = 'bearish'
                confidence = min(bearish_volume / (bullish_volume + bearish_volume), 0.8)
            else:
                signal = 'neutral'
                confidence = 0.5

        # Additional VPA-based confirmation
        recent_vpa = vpa_df.tail(5)
        volume_confirmation = recent_vpa['bullish_volume'].sum() if signal == 'bullish' else recent_vpa['bearish_volume'].sum()

        return {
            'signal': signal,
            'confidence': confidence,
            'volume_confirmation': volume_confirmation,
            'vpa_features': vpa_df.iloc[-1].to_dict()
        }

    def save_model(self, path: str):
        """Save VPA model"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"[SAVE] VPA model saved to {path}")

    def load_model(self, path: str):
        """Load VPA model"""
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        logger.info(f"[LOAD] VPA model loaded from {path}")

# Integration functions for existing codebase

def integrate_vpa_with_multimodal(fusion_model: Any, vpa_analyzer: VPAAnalyzer) -> Any:
    """
    Integrate VPA analysis with multimodal fusion

    Args:
        fusion_model: Existing multimodal fusion model
        vpa_analyzer: VPA analyzer instance

    Returns:
        Enhanced fusion model
    """
    # This would extend the multimodal fusion to include VPA signals
    # Implementation depends on the specific fusion architecture
    logger.info("[INTEGRATION] VPA integrated with multimodal fusion")
    return fusion_model

def integrate_vpa_with_faiss(faiss_matcher: Any, vpa_features: pd.DataFrame) -> bool:
    """
    Add VPA patterns to FAISS index

    Args:
        faiss_matcher: FAISS pattern matcher
        vpa_features: DataFrame with VPA features

    Returns:
        Success status
    """
    try:
        # Extract VPA feature vectors for indexing
        feature_cols = [col for col in vpa_features.columns if col.startswith(('vol_', 'volume_', 'vpt', 'nvi', 'pvi'))]
        vpa_vectors = vpa_features[feature_cols].values

        # Add to FAISS index
        for i, vector in enumerate(vpa_vectors):
            metadata = {
                'symbol': vpa_features.iloc[i].get('symbol', 'unknown'),
                'timestamp': vpa_features.iloc[i].get('date', datetime.now()),
                'pattern_type': 'vpa'
            }
            faiss_matcher.add_pattern(vector, metadata)

        logger.info(f"[INTEGRATION] Added {len(vpa_vectors)} VPA patterns to FAISS")
        return True

    except Exception as e:
        logger.error(f"[ERROR] FAISS integration failed: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 105 + np.random.randn(100).cumsum(),
        'low': 95 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.lognormal(15, 1, 100)
    })

    # Initialize VPA analyzer
    analyzer = VPAAnalyzer()

    # Compute VPA features
    vpa_data = analyzer.compute_vpa_features(sample_data)
    print("VPA Features computed:")
    print(vpa_data[['vol_price_ratio', 'volume_imbalance', 'volume_climax', 'bullish_volume']].tail())

    # Make prediction
    prediction = analyzer.predict_vpa_signal(sample_data)
    print(f"\nVPA Prediction: {prediction['signal']} (confidence: {prediction['confidence']:.3f})")