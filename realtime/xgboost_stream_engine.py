#!/usr/bin/env python3
"""
XGBoost Stream Engine for Real-Time Predictions

This module provides real-time prediction capabilities for XGBoost trading models,
handling streaming market data with incremental feature computation and async inference.

Features:
- Rolling feature buffer for indicator calculations
- Async prediction pipeline
- Integration with existing XGBoostTradingModel
- Signal dispatch with confidence scores
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Project imports
from xgboost_trading_model import XGBoostTradingModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostStreamEngine:
    """
    Real-time XGBoost prediction engine with streaming data support
    """

    def __init__(self, symbol: str, buffer_size: int = 100, lookback_periods: int = 50):
        """
        Initialize the stream engine

        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            buffer_size: Maximum size of rolling data buffer
            lookback_periods: Number of periods needed for feature calculations
        """
        self.symbol = symbol
        self.buffer_size = buffer_size
        self.lookback_periods = lookback_periods

        # Data buffer
        self.data_buffer = pd.DataFrame()
        self.feature_buffer = pd.DataFrame()

        # Models
        self.regression_model = None
        self.classification_model = None
        self.scaler = None

        # Load models
        self._load_models()

        logger.info(f"[INIT] XGBoost Stream Engine for {symbol} (buffer: {buffer_size})")

    def _load_models(self):
        """Load trained models and scaler"""
        try:
            model = XGBoostTradingModel(self.symbol)
            if model.load_models():
                self.regression_model = model.regression_model
                self.classification_model = model.classification_model
                self.scaler = model.scaler
                logger.info(f"[SUCCESS] Models loaded for {self.symbol}")
            else:
                logger.warning(f"[WARNING] No saved models found for {self.symbol}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load models: {e}")

    def update_buffer(self, new_data: Dict[str, Any]) -> bool:
        """
        Update the rolling data buffer with new market data

        Args:
            new_data: Dictionary with OHLCV data and timestamp

        Returns:
            bool: True if buffer updated successfully
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([new_data])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Append to buffer
            self.data_buffer = pd.concat([self.data_buffer, df])

            # Maintain buffer size
            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer = self.data_buffer.tail(self.buffer_size)

            # Update features if we have enough data
            if len(self.data_buffer) >= self.lookback_periods:
                self._update_features()

            logger.debug(f"[BUFFER] Updated with new data point for {self.symbol}")
            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to update buffer: {e}")
            return False

    def _update_features(self):
        """Compute features from current data buffer"""
        try:
            df = self.data_buffer.copy()

            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Moving averages
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

            # Volatility measures
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            df['volatility_50'] = df['returns'].rolling(window=50).std()

            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']

            # RSI
            def calculate_rsi(data, window=14):
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            df['rsi_14'] = calculate_rsi(df['close'], 14)

            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            df['bb_middle'] = sma_20
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # Momentum
            for period in [5, 10, 20]:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

            # Drop NaN values
            df = df.dropna()

            self.feature_buffer = df

            logger.debug(f"[FEATURES] Updated features for {self.symbol}")

        except Exception as e:
            logger.error(f"[ERROR] Failed to update features: {e}")

    async def predict_async(self) -> Optional[Dict[str, Any]]:
        """
        Async prediction using latest features

        Returns:
            Dict with prediction results or None if insufficient data
        """
        try:
            if self.feature_buffer.empty or len(self.feature_buffer) < 1:
                logger.debug(f"[PREDICT] Insufficient data for {self.symbol}")
                return None

            if self.classification_model is None or self.scaler is None:
                logger.warning(f"[PREDICT] Models not loaded for {self.symbol}")
                return None

            # Get latest features
            latest_features = self.feature_buffer.iloc[-1:]

            # Prepare features for prediction
            feature_cols = [col for col in latest_features.columns
                          if col not in ['close', 'timestamp'] and not col.startswith('target_')]
            X = latest_features[feature_cols]

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Make predictions
            signal_pred = self.classification_model.predict(X_scaled)[0]
            signal_proba = self.classification_model.predict_proba(X_scaled)[0]

            # Price prediction if regression model available
            price_pred = None
            if self.regression_model is not None:
                price_pred = self.regression_model.predict(X_scaled)[0]

            # Prepare result
            result = {
                'symbol': self.symbol,
                'timestamp': latest_features.index[-1],
                'current_price': latest_features['close'].iloc[-1],
                'signal': int(signal_pred),  # -1: Sell, 0: Hold, 1: Buy
                'signal_confidence': float(max(signal_proba)),
                'signal_probabilities': {
                    'sell': float(signal_proba[0]),
                    'hold': float(signal_proba[1]),
                    'buy': float(signal_proba[2])
                },
                'predicted_price': float(price_pred) if price_pred is not None else None,
                'buffer_size': len(self.data_buffer),
                'features_computed': len(self.feature_buffer)
            }

            logger.info(f"[PREDICT] {self.symbol} signal: {signal_pred} (conf: {result['signal_confidence']:.3f})")
            return result

        except Exception as e:
            logger.error(f"[ERROR] Prediction failed: {e}")
            return None

    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status"""
        return {
            'symbol': self.symbol,
            'data_points': len(self.data_buffer),
            'feature_points': len(self.feature_buffer),
            'models_loaded': self.classification_model is not None,
            'ready_for_prediction': len(self.feature_buffer) >= 1 and self.classification_model is not None
        }

    async def process_stream_data(self, data_stream: asyncio.Queue) -> asyncio.Queue:
        """
        Process streaming data and generate predictions

        Args:
            data_stream: Async queue with incoming market data

        Returns:
            Async queue with prediction results
        """
        prediction_queue = asyncio.Queue()

        async def process_loop():
            while True:
                try:
                    # Get new data
                    new_data = await data_stream.get()

                    # Update buffer
                    if self.update_buffer(new_data):
                        # Make prediction
                        prediction = await self.predict_async()

                        if prediction:
                            await prediction_queue.put(prediction)

                    data_stream.task_done()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[STREAM] Processing error: {e}")

        # Start processing task
        task = asyncio.create_task(process_loop())

        # Return queue and task for external management
        return prediction_queue

# Example usage
async def example_stream_processing():
    """Example of using the stream engine"""
    engine = XGBoostStreamEngine('AAPL')

    # Simulate data stream
    data_queue = asyncio.Queue()

    # Add some sample data
    sample_data = [
        {'timestamp': datetime.now() - timedelta(minutes=i), 'open': 150 + i*0.1, 'high': 151 + i*0.1,
         'low': 149 + i*0.1, 'close': 150.5 + i*0.1, 'volume': 1000000 + i*10000}
        for i in range(60, 0, -1)  # 60 data points
    ]

    for data in sample_data:
        await data_queue.put(data)

    # Process stream
    prediction_queue = await engine.process_stream_data(data_queue)

    # Collect predictions
    predictions = []
    while not prediction_queue.empty():
        pred = await prediction_queue.get()
        predictions.append(pred)

    print(f"Generated {len(predictions)} predictions")
    if predictions:
        print(f"Latest prediction: {predictions[-1]}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_stream_processing())