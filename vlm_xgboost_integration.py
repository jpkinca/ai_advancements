#!/usr/bin/env python3
"""
VLM-XGBoost Integration Module

This module provides seamless integration between VLM (Vision-Language Model)
capabilities and the existing XGBoost trading system, creating a unified
multimodal trading intelligence platform.

Features:
- Unified prediction pipeline
- Enhanced feature engineering with VLM
- Adaptive model selection
- Performance comparison and monitoring
- Trading strategy integration
- Backtesting with multimodal signals
"""

import os
import json
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMXGBoostIntegrator:
    """
    Integrates VLM capabilities with XGBoost trading system
    """

    def __init__(self,
                 xgb_model_path: Optional[str] = None,
                 vlm_model_path: Optional[str] = None,
                 fusion_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integrator

        Args:
            xgb_model_path: Path to XGBoost model
            vlm_model_path: Path to VLM/CLIP model
            fusion_config: Configuration for multimodal fusion
        """
        self.xgb_model_path = xgb_model_path
        self.vlm_model_path = vlm_model_path
        self.fusion_config = fusion_config or {
            'method': 'weighted_average',
            'clip_weight': 0.6,
            'xgb_weight': 0.4,
            'confidence_threshold': 0.7
        }

        # Initialize components
        self.xgb_model = None
        self.vlm_model = None
        self.fusion_model = None
        self.chart_generator = None
        self.text_generator = None

        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = {}

        # Load components
        self._load_components()

        logger.info("[INIT] VLM-XGBoost Integrator initialized")

    def _load_components(self):
        """Load all necessary components"""
        try:
            # Load XGBoost model
            if self.xgb_model_path and os.path.exists(self.xgb_model_path):
                import joblib
                self.xgb_model = joblib.load(self.xgb_model_path)
                logger.info(f"[LOAD] XGBoost model loaded from {self.xgb_model_path}")
            else:
                logger.warning("[LOAD] XGBoost model path not provided or file not found")

            # Load VLM components
            if self.vlm_model_path and os.path.exists(self.vlm_model_path):
                from multimodal_fusion import MultimodalFusion
                self.fusion_model = MultimodalFusion(
                    clip_model_path=self.vlm_model_path,
                    xgb_model_path=self.xgb_model_path,
                    fusion_method=self.fusion_config['method']
                )
                self.fusion_model.clip_weight = self.fusion_config['clip_weight']
                self.fusion_model.xgb_weight = self.fusion_config['xgb_weight']
                logger.info(f"[LOAD] VLM fusion model loaded from {self.vlm_model_path}")
            else:
                logger.warning("[LOAD] VLM model path not provided or file not found")

            # Load chart and text generators
            try:
                from vlm.chart_image_generator import ChartImageGenerator
                self.chart_generator = ChartImageGenerator()
                logger.info("[LOAD] Chart generator loaded")
            except ImportError:
                logger.warning("[LOAD] Chart generator not available")

            try:
                from vlm.text_label_generator import TextLabelGenerator
                self.text_generator = TextLabelGenerator()
                logger.info("[LOAD] Text label generator loaded")
            except ImportError:
                logger.warning("[LOAD] Text label generator not available")

        except Exception as e:
            logger.error(f"[ERROR] Failed to load components: {e}")
            raise

    def predict_unified(self,
                       market_data: pd.DataFrame,
                       symbol: str = "UNKNOWN",
                       generate_chart: bool = True) -> Dict[str, Any]:
        """
        Make unified prediction using both XGBoost and VLM

        Args:
            market_data: DataFrame with OHLCV and technical indicators
            symbol: Trading symbol
            generate_chart: Whether to generate chart for VLM

        Returns:
            Unified prediction results
        """
        start_time = datetime.now()

        try:
            # Extract technical features for XGBoost
            technical_features = self._extract_technical_features(market_data)

            # XGBoost prediction
            xgb_result = None
            if self.xgb_model:
                xgb_result = self._predict_xgb(technical_features)

            # VLM prediction
            vlm_result = None
            if self.fusion_model and generate_chart:
                vlm_result = self._predict_vlm(market_data, symbol, technical_features)

            # Multimodal fusion
            if xgb_result and vlm_result and self.fusion_model:
                fused_result = self.fusion_model.fuse_predictions(vlm_result, xgb_result)
                prediction_type = "multimodal"
            elif xgb_result:
                fused_result = xgb_result
                prediction_type = "xgboost_only"
            elif vlm_result:
                fused_result = vlm_result
                prediction_type = "vlm_only"
            else:
                raise RuntimeError("No models available for prediction")

            # Add metadata
            result = {
                'prediction': fused_result['prediction'],
                'confidence': fused_result['confidence'],
                'probabilities': fused_result.get('probabilities', []),
                'prediction_type': prediction_type,
                'symbol': symbol,
                'timestamp': market_data.index[-1] if hasattr(market_data, 'index') else datetime.now(),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'technical_features': technical_features,
                'xgb_result': xgb_result,
                'vlm_result': vlm_result,
                'fusion_config': self.fusion_config
            }

            # Store prediction history
            self.prediction_history.append(result)

            # Keep only recent history (last 1000 predictions)
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]

            return result

        except Exception as e:
            logger.error(f"[ERROR] Unified prediction failed: {e}")
            raise

    def _extract_technical_features(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract technical features from market data"""
        try:
            # Get latest data point
            latest = market_data.iloc[-1] if hasattr(market_data, 'iloc') else market_data

            # Basic price data
            features = {
                'close': latest.get('close', latest.get('Close', 0)),
                'volume': latest.get('volume', latest.get('Volume', 0)),
                'high': latest.get('high', latest.get('High', 0)),
                'low': latest.get('low', latest.get('Low', 0)),
                'open': latest.get('open', latest.get('Open', 0))
            }

            # Technical indicators (if available)
            technical_indicators = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'volume_ratio', 'price_change', 'volatility'
            ]

            for indicator in technical_indicators:
                if hasattr(latest, indicator):
                    features[indicator] = latest[indicator]
                elif indicator in latest.index:
                    features[indicator] = latest[indicator]
                else:
                    features[indicator] = 0.0  # Default value

            return features

        except Exception as e:
            logger.error(f"[ERROR] Feature extraction failed: {e}")
            return {}

    def _predict_xgb(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make XGBoost prediction"""
        if not self.xgb_model:
            raise RuntimeError("XGBoost model not loaded")

        try:
            # Convert features to array
            feature_vector = []
            feature_order = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'volume_ratio', 'price_change', 'volatility'
            ]

            for feature_name in feature_order:
                feature_vector.append(features.get(feature_name, 0.0))

            # Make prediction
            prediction = self.xgb_model.predict([feature_vector])[0]
            probabilities = self.xgb_model.predict_proba([feature_vector])[0]

            return {
                'prediction': int(prediction),
                'probabilities': probabilities.tolist(),
                'confidence': float(max(probabilities))
            }

        except Exception as e:
            logger.error(f"[ERROR] XGBoost prediction failed: {e}")
            raise

    def _predict_vlm(self, market_data: pd.DataFrame, symbol: str,
                    technical_features: Dict[str, Any]) -> Dict[str, Any]:
        """Make VLM prediction"""
        try:
            # Generate chart
            if not self.chart_generator:
                raise RuntimeError("Chart generator not available")

            chart_image = self.chart_generator.generate_chart_image(
                market_data,
                symbol=symbol,
                timeframe="1D"  # Default timeframe
            )

            # Generate text descriptions
            text_descriptions = [
                "bullish trend pattern",
                "bearish reversal pattern",
                "sideways consolidation",
                "breakout formation",
                "high volatility period"
            ]

            # Make multimodal prediction
            if self.fusion_model:
                # Ensure text_descriptions is a list of strings
                if isinstance(text_descriptions, list) and all(isinstance(t, str) for t in text_descriptions):
                    result = self.fusion_model.predict_multimodal(
                        chart_image,
                        text_descriptions,
                        technical_features
                    )
                else:
                    # Fallback to default descriptions
                    fallback_descriptions = [
                        "bullish trend pattern",
                        "bearish reversal pattern"
                    ]
                    result = self.fusion_model.predict_multimodal(
                        chart_image,
                        fallback_descriptions,
                        technical_features
                    )
            else:
                raise RuntimeError("Fusion model not available")

            return result

        except Exception as e:
            logger.error(f"[ERROR] VLM prediction failed: {e}")
            raise

    def backtest_unified_strategy(self,
                                historical_data: pd.DataFrame,
                                initial_capital: float = 10000.0,
                                position_size: float = 0.1,
                                stop_loss: float = 0.02,
                                take_profit: float = 0.05) -> Dict[str, Any]:
        """
        Backtest unified trading strategy

        Args:
            historical_data: Historical market data
            initial_capital: Starting capital
            position_size: Position size as fraction of capital
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage

        Returns:
            Backtest results
        """
        logger.info("[BACKTEST] Starting unified strategy backtest...")

        try:
            capital = initial_capital
            position = 0  # 0: no position, 1: long, -1: short
            entry_price = 0.0
            shares = 0.0  # Initialize shares
            trades = []
            equity_curve = [capital]

            # Process each data point
            for i in tqdm(range(len(historical_data)), desc="Backtesting"):
                current_data = historical_data.iloc[:i+1]

                if len(current_data) < 50:  # Need minimum data for indicators
                    equity_curve.append(capital)
                    continue

                try:
                    # Get unified prediction
                    prediction_result = self.predict_unified(
                        current_data,
                        symbol="BACKTEST",
                        generate_chart=True
                    )

                    current_price = current_data.iloc[-1]['close']

                    # Trading logic
                    if position == 0:  # No position
                        if prediction_result['prediction'] == 1 and prediction_result['confidence'] > self.fusion_config['confidence_threshold']:
                            # Enter long position
                            position = 1
                            entry_price = current_price
                            position_value = capital * position_size
                            shares = position_value / current_price

                            trades.append({
                                'type': 'BUY',
                                'price': current_price,
                                'shares': shares,
                                'timestamp': current_data.index[-1],
                                'prediction_confidence': prediction_result['confidence']
                            })

                        elif prediction_result['prediction'] == 0 and prediction_result['confidence'] > self.fusion_config['confidence_threshold']:
                            # Enter short position
                            position = -1
                            entry_price = current_price
                            position_value = capital * position_size
                            shares = position_value / current_price

                            trades.append({
                                'type': 'SELL',
                                'price': current_price,
                                'shares': shares,
                                'timestamp': current_data.index[-1],
                                'prediction_confidence': prediction_result['confidence']
                            })

                    else:  # Have position
                        # Check stop loss and take profit
                        if position == 1:  # Long position
                            pnl_pct = (current_price - entry_price) / entry_price

                            if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                                # Exit position
                                exit_value = position * shares * current_price
                                capital += exit_value

                                trades.append({
                                    'type': 'SELL_EXIT',
                                    'price': current_price,
                                    'shares': shares,
                                    'pnl': exit_value,
                                    'timestamp': current_data.index[-1]
                                })

                                position = 0
                                entry_price = 0.0

                        elif position == -1:  # Short position
                            pnl_pct = (entry_price - current_price) / entry_price

                            if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                                # Exit position
                                exit_value = position * shares * current_price
                                capital += exit_value

                                trades.append({
                                    'type': 'BUY_EXIT',
                                    'price': current_price,
                                    'shares': shares,
                                    'pnl': exit_value,
                                    'timestamp': current_data.index[-1]
                                })

                                position = 0
                                entry_price = 0.0

                    equity_curve.append(capital)

                except Exception as e:
                    logger.warning(f"[BACKTEST] Prediction failed at step {i}: {e}")
                    equity_curve.append(capital)

            # Calculate performance metrics
            total_return = (capital - initial_capital) / initial_capital
            num_trades = len([t for t in trades if 'EXIT' in t['type']])

            # Sharpe ratio (simplified)
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

            # Maximum drawdown
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            max_drawdown = np.max(drawdown)

            results = {
                'total_return': total_return,
                'final_capital': capital,
                'num_trades': num_trades,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trades': trades,
                'equity_curve': equity_curve,
                'win_rate': len([t for t in trades if t.get('pnl', 0) > 0]) / max(num_trades, 1)
            }

            logger.info(f"[BACKTEST] Completed. Return: {total_return:.2%}, Trades: {num_trades}, Sharpe: {sharpe_ratio:.2f}")

            return results

        except Exception as e:
            logger.error(f"[ERROR] Backtest failed: {e}")
            raise

    def optimize_fusion_weights(self,
                              validation_data: pd.DataFrame,
                              weight_range: Tuple[float, float] = (0.0, 1.0),
                              step: float = 0.1) -> Dict[str, Any]:
        """
        Optimize fusion weights using validation data

        Args:
            validation_data: Validation dataset
            weight_range: Range for weight optimization
            step: Step size for grid search

        Returns:
            Optimization results
        """
        logger.info("[OPTIMIZE] Optimizing fusion weights...")

        best_score = 0.0
        best_weights = (self.fusion_config['clip_weight'], self.fusion_config['xgb_weight'])

        results = []

        if not self.fusion_model:
            logger.warning("[OPTIMIZE] No fusion model available for optimization")
            return {'error': 'No fusion model available'}

        # Grid search over weights
        for clip_w in np.arange(weight_range[0], weight_range[1] + step, step):
            xgb_w = 1.0 - clip_w

            # Temporarily set weights
            old_clip_w = self.fusion_model.clip_weight
            old_xgb_w = self.fusion_model.xgb_weight

            self.fusion_model.clip_weight = clip_w
            self.fusion_model.xgb_weight = xgb_w

            try:
                # Evaluate on validation data
                predictions = []
                true_labels = []

                for i in range(len(validation_data)):
                    current_data = validation_data.iloc[:i+1]
                    if len(current_data) < 50:
                        continue

                    # Get prediction
                    result = self.predict_unified(current_data, generate_chart=False)

                    # Assume next price movement as label (simplified)
                    if i < len(validation_data) - 1:
                        next_return = (validation_data.iloc[i+1]['close'] - validation_data.iloc[i]['close']) / validation_data.iloc[i]['close']
                        true_label = 1 if next_return > 0 else 0
                        predictions.append(result['prediction'])
                        true_labels.append(true_label)

                # Calculate accuracy
                if predictions:
                    accuracy = np.mean([p == t for p, t in zip(predictions, true_labels)])
                    results.append({
                        'clip_weight': clip_w,
                        'xgb_weight': xgb_w,
                        'accuracy': accuracy
                    })

                    if accuracy > best_score:
                        best_score = accuracy
                        best_weights = (clip_w, xgb_w)

            except Exception as e:
                logger.warning(f"[OPTIMIZE] Failed for weights {clip_w:.2f}, {xgb_w:.2f}: {e}")

            # Restore weights
            if self.fusion_model:
                self.fusion_model.clip_weight = old_clip_w
                self.fusion_model.xgb_weight = old_xgb_w

        # Set best weights
        self.fusion_config['clip_weight'] = best_weights[0]
        self.fusion_config['xgb_weight'] = best_weights[1]
        if self.fusion_model:
            self.fusion_model.clip_weight = best_weights[0]
            self.fusion_model.xgb_weight = best_weights[1]

        logger.info(f"[OPTIMIZE] Best weights: CLIP={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, Accuracy={best_score:.4f}")

        return {
            'best_weights': best_weights,
            'best_score': best_score,
            'optimization_results': results
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if not self.prediction_history:
            return {"error": "No prediction history available"}

        # Calculate metrics
        predictions = [p['prediction'] for p in self.prediction_history]
        confidences = [p['confidence'] for p in self.prediction_history]
        processing_times = [p['processing_time'] for p in self.prediction_history]

        # Prediction distribution
        pred_counts = pd.Series(predictions).value_counts()

        # Confidence distribution
        conf_bins = pd.cut(confidences, bins=10)

        report = {
            'total_predictions': len(self.prediction_history),
            'prediction_distribution': pred_counts.to_dict(),
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'average_processing_time': np.mean(processing_times),
            'processing_time_std': np.std(processing_times),
            'prediction_types': pd.Series([p['prediction_type'] for p in self.prediction_history]).value_counts().to_dict(),
            'recent_predictions': self.prediction_history[-10:]  # Last 10 predictions
        }

        return report

    def save_integration_config(self, path: str):
        """Save integration configuration"""
        config = {
            'xgb_model_path': self.xgb_model_path,
            'vlm_model_path': self.vlm_model_path,
            'fusion_config': self.fusion_config,
            'performance_metrics': self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }

        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        logger.info(f"[SAVE] Integration config saved to {path}")

    def load_integration_config(self, path: str):
        """Load integration configuration"""
        with open(path, 'r') as f:
            config = json.load(f)

        self.xgb_model_path = config.get('xgb_model_path')
        self.vlm_model_path = config.get('vlm_model_path')
        self.fusion_config = config.get('fusion_config', self.fusion_config)
        self.performance_metrics = config.get('performance_metrics', {})

        # Reload components with new paths
        self._load_components()

        logger.info(f"[LOAD] Integration config loaded from {path}")

# Utility functions
def create_integrator(xgb_model_path: str,
                     vlm_model_path: str,
                     fusion_config: Optional[Dict[str, Any]] = None) -> VLMXGBoostIntegrator:
    """
    Create and configure VLM-XGBoost integrator

    Args:
        xgb_model_path: Path to XGBoost model
        vlm_model_path: Path to VLM model
        fusion_config: Fusion configuration

    Returns:
        Configured VLMXGBoostIntegrator instance
    """
    integrator = VLMXGBoostIntegrator(
        xgb_model_path=xgb_model_path,
        vlm_model_path=vlm_model_path,
        fusion_config=fusion_config
    )

    return integrator

def run_unified_prediction_demo(market_data_path: str,
                               xgb_model_path: str,
                               vlm_model_path: str):
    """
    Run a demonstration of unified prediction

    Args:
        market_data_path: Path to market data CSV
        xgb_model_path: Path to XGBoost model
        vlm_model_path: Path to VLM model
    """
    # Load market data
    market_data = pd.read_csv(market_data_path, index_col=0, parse_dates=True)

    # Create integrator
    integrator = create_integrator(xgb_model_path, vlm_model_path)

    # Make prediction
    result = integrator.predict_unified(market_data, symbol="DEMO")

    print("=== Unified Prediction Results ===")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Prediction Type: {result['prediction_type']}")
    print(f"Processing Time: {result['processing_time']:.4f}s")

    return result

if __name__ == "__main__":
    # Example usage
    integrator = create_integrator(
        xgb_model_path="xgboost_trading_model.pkl",
        vlm_model_path="vlm/models/best_model.pt"
    )

    # Load some sample data and make prediction
    # result = run_unified_prediction_demo("sample_market_data.csv", "xgboost_trading_model.pkl", "vlm/models/best_model.pt")
    print("VLM-XGBoost Integration module ready!")