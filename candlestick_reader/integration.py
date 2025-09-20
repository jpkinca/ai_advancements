#!/usr/bin/env python3
"""
Candlestick Analysis Integration

This module integrates AI-powered candlestick analysis with the existing
VPA, multimodal fusion, and trading systems.

Features:
- Real-time candlestick signal generation
- Integration with VPA for enhanced signals
- Multimodal fusion with chart images
- FAISS-based pattern similarity matching
- Celery task integration for async processing
"""

import os
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

# Local imports
from candlestick_reader.candlestick_analyzer import CandlestickAnalyzer
from volume.volume_price_action import VPAAnalyzer
from multimodal_fusion import MultimodalFusion
from ai_data_accessor import AIDataAccessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedCandlestickSystem:
    """
    Integrated candlestick analysis system combining multiple AI models
    """

    def __init__(self, faiss_matcher=None, multimodal_fusion=None):
        self.candlestick_analyzer = CandlestickAnalyzer()
        self.vpa_analyzer = VPAAnalyzer()
        self.multimodal_fusion = multimodal_fusion
        self.faiss_matcher = faiss_matcher
        self.data_accessor = None

        # Signal thresholds
        self.min_candlestick_confidence = 0.6
        self.min_combined_confidence = 0.7
        self.vpa_weight = 0.4
        self.candlestick_weight = 0.4
        self.multimodal_weight = 0.2

        logger.info("[INIT] Integrated Candlestick System initialized")

    async def initialize_data_accessor(self):
        """Initialize database connection"""
        if not self.data_accessor:
            self.data_accessor = await AIDataAccessor.create_ai_data_accessor()

    async def analyze_symbol_comprehensive(self, symbol: str, timeframe: str = '5min',
                                         lookback_periods: int = 50) -> Dict[str, Any]:
        """
        Comprehensive analysis combining candlesticks, VPA, and multimodal fusion

        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            lookback_periods: Number of periods to analyze

        Returns:
            Complete analysis results
        """
        try:
            await self.initialize_data_accessor()

            # Get market data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_periods//(24*60//int(timeframe.replace('min', ''))) + 1)

            df = await self.data_accessor.get_symbol_data(symbol, timeframe, start_date, end_date)

            if df.empty:
                return {'error': 'No data available', 'symbol': symbol}

            # Generate candlestick chart
            chart_path = self.candlestick_analyzer.generate_candlestick_chart(df, symbol, timeframe)

            # Individual analyses
            candlestick_result = self.candlestick_analyzer.analyze_candlestick_signal(df, symbol, chart_path)
            vpa_result = self.vpa_analyzer.predict_vpa_signal(df)

            # Multimodal fusion if available
            multimodal_result = None
            if self.multimodal_fusion:
                try:
                    # Load chart image for multimodal analysis
                    chart_image = self._load_chart_image(chart_path)

                    # Get technical features
                    technical_features = self._extract_technical_features(df)

                    multimodal_result = self.multimodal_fusion.predict_multimodal(
                        chart_image, ['bullish', 'bearish', 'neutral'], technical_features, df
                    )
                except Exception as e:
                    logger.warning(f"[WARNING] Multimodal fusion failed: {e}")

            # Combine all signals
            combined_result = self._combine_all_signals(
                candlestick_result, vpa_result, multimodal_result, df
            )

            # Add FAISS similarity if available
            if self.faiss_matcher:
                similar_patterns = self._find_similar_patterns(combined_result, df)
                combined_result['similar_patterns'] = similar_patterns

            # Add metadata
            combined_result.update({
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_timestamp': datetime.now(),
                'data_points': len(df),
                'chart_path': chart_path,
                'models_used': {
                    'candlestick': True,
                    'vpa': True,
                    'multimodal': self.multimodal_fusion is not None,
                    'faiss': self.faiss_matcher is not None
                }
            })

            logger.info(f"[ANALYSIS] {symbol} comprehensive analysis: {combined_result['signal']} (conf: {combined_result['confidence']:.3f})")
            return combined_result

        except Exception as e:
            logger.error(f"[ERROR] Comprehensive analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now()
            }

    def _load_chart_image(self, chart_path: str):
        """Load chart image for multimodal analysis"""
        try:
            from PIL import Image
            return Image.open(chart_path).convert('RGB')
        except Exception as e:
            logger.warning(f"[WARNING] Failed to load chart image: {e}")
            return None

    def _extract_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract technical indicators for multimodal fusion"""
        try:
            features = {}

            # Basic price features
            features['close'] = df['close'].iloc[-1]
            features['returns'] = df['close'].pct_change().iloc[-1]
            features['volume'] = df['volume'].iloc[-1]

            # Moving averages
            for period in [5, 10, 20, 50]:
                if len(df) >= period:
                    features[f'sma_{period}'] = df['close'].rolling(period).mean().iloc[-1]
                    features[f'ema_{period}'] = df['close'].ewm(span=period).mean().iloc[-1]

            # Volatility
            features['volatility'] = df['close'].pct_change().std()

            # RSI (simplified)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]

            # MACD (simplified)
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            features['macd'] = (ema12 - ema26).iloc[-1]

            return features

        except Exception as e:
            logger.warning(f"[WARNING] Failed to extract technical features: {e}")
            return {}

    def _combine_all_signals(self, candlestick: Dict, vpa: Dict,
                           multimodal: Optional[Dict], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Combine signals from all analysis methods

        Args:
            candlestick: Candlestick analysis result
            vpa: VPA analysis result
            multimodal: Multimodal fusion result (optional)
            df: Market data

        Returns:
            Combined analysis result
        """
        combined = {
            'signal': 'neutral',
            'confidence': 0.0,
            'components': {
                'candlestick': candlestick,
                'vpa': vpa,
                'multimodal': multimodal
            }
        }

        # Extract individual signals and confidences
        signals = []
        confidences = []
        weights = []

        # Candlestick signal
        if 'signal' in candlestick and candlestick.get('confidence', 0) > 0:
            signals.append(candlestick['signal'])
            confidences.append(candlestick['confidence'])
            weights.append(self.candlestick_weight)

        # VPA signal
        if 'signal' in vpa and vpa.get('confidence', 0) > 0:
            signals.append(vpa['signal'])
            confidences.append(vpa['confidence'])
            weights.append(self.vpa_weight)

        # Multimodal signal
        if multimodal and 'prediction' in multimodal:
            # Map prediction to signal
            pred_map = {0: 'bullish', 1: 'bearish', 2: 'neutral'}
            multi_signal = pred_map.get(multimodal['prediction'], 'neutral')
            multi_conf = multimodal.get('confidence', 0.5)

            signals.append(multi_signal)
            confidences.append(multi_conf)
            weights.append(self.multimodal_weight)

        # Combine signals using weighted voting
        if signals:
            # Count votes for each signal type
            signal_votes = {}
            total_weight = 0

            for signal, conf, weight in zip(signals, confidences, weights):
                if signal not in signal_votes:
                    signal_votes[signal] = 0
                signal_votes[signal] += conf * weight
                total_weight += weight

            # Select signal with highest weighted votes
            best_signal = max(signal_votes.items(), key=lambda x: x[1])
            combined['signal'] = best_signal[0]
            combined['confidence'] = best_signal[1] / total_weight if total_weight > 0 else 0

            # Add agreement analysis
            combined['signal_agreement'] = len([s for s in signals if s == combined['signal']]) / len(signals)
            combined['signal_consensus'] = combined['signal_agreement'] >= 0.67  # 2/3 agreement

        # Add market context
        combined['market_context'] = self._assess_market_context(df)

        # Determine trade recommendation
        combined['trade_recommendation'] = self._generate_trade_recommendation(combined, df)

        return combined

    def _assess_market_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall market context"""
        try:
            # Trend analysis
            sma_20 = df['close'].rolling(20).mean()
            current_price = df['close'].iloc[-1]
            trend = 'bullish' if current_price > sma_20.iloc[-1] else 'bearish'

            # Volatility
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()

            # Volume trend
            volume_sma = df['volume'].rolling(10).mean()
            volume_trend = 'increasing' if df['volume'].iloc[-1] > volume_sma.iloc[-1] else 'decreasing'

            return {
                'trend': trend,
                'volatility': 'high' if volatility > 0.02 else 'low',
                'volume_trend': volume_trend,
                'current_price': current_price
            }
        except Exception as e:
            logger.warning(f"[WARNING] Market context assessment failed: {e}")
            return {}

    def _generate_trade_recommendation(self, analysis: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trade recommendation based on analysis"""
        try:
            signal = analysis['signal']
            confidence = analysis['confidence']
            market_context = analysis.get('market_context', {})

            recommendation = {
                'action': 'HOLD',
                'confidence': confidence,
                'reasoning': [],
                'position_size': 0,
                'stop_loss': None,
                'take_profit': None
            }

            # Basic signal check
            if confidence < self.min_combined_confidence:
                recommendation['reasoning'].append("Insufficient confidence for trade")
                return recommendation

            # Signal-based action
            if signal == 'bullish':
                recommendation['action'] = 'BUY'
                current_price = market_context.get('current_price', df['close'].iloc[-1])
                recommendation['position_size'] = 100  # Base position
                recommendation['stop_loss'] = current_price * 0.98  # 2% stop loss
                recommendation['take_profit'] = current_price * 1.04  # 4% take profit
                recommendation['reasoning'].append("Bullish signal detected")

            elif signal == 'bearish':
                recommendation['action'] = 'SELL'
                current_price = market_context.get('current_price', df['close'].iloc[-1])
                recommendation['position_size'] = 100
                recommendation['stop_loss'] = current_price * 1.02  # 2% stop loss
                recommendation['take_profit'] = current_price * 0.96  # 4% take profit
                recommendation['reasoning'].append("Bearish signal detected")

            # Context adjustments
            if market_context.get('trend') == signal:
                recommendation['confidence'] *= 1.2
                recommendation['reasoning'].append("Signal aligns with market trend")
            else:
                recommendation['confidence'] *= 0.8
                recommendation['reasoning'].append("Signal against market trend")

            if market_context.get('volatility') == 'high':
                recommendation['position_size'] = int(recommendation['position_size'] * 0.7)
                recommendation['reasoning'].append("Reduced position size due to high volatility")

            return recommendation

        except Exception as e:
            logger.warning(f"[WARNING] Trade recommendation generation failed: {e}")
            return {'action': 'HOLD', 'reasoning': ['Error in analysis']}

    def _find_similar_patterns(self, analysis: Dict, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find similar historical patterns using FAISS"""
        if not self.faiss_matcher:
            return []

        try:
            # Create feature vector from current analysis
            features = self._create_pattern_features(analysis, df)

            # Search for similar patterns
            similar = self.faiss_matcher.find_similar_patterns(features, k=5)

            return similar

        except Exception as e:
            logger.warning(f"[WARNING] Pattern similarity search failed: {e}")
            return []

    def _create_pattern_features(self, analysis: Dict, df: pd.DataFrame) -> np.ndarray:
        """Create feature vector for pattern matching"""
        try:
            features = []

            # Price action features
            features.extend(df['close'].pct_change().tail(10).fillna(0).values)

            # Volume features
            features.extend(df['volume'].pct_change().tail(10).fillna(0).values)

            # Signal confidence
            features.append(analysis.get('confidence', 0))

            # Pattern type encoding
            pattern_types = ['bullish', 'bearish', 'neutral']
            pattern_encoding = [1 if analysis.get('signal') == pt else 0 for pt in pattern_types]
            features.extend(pattern_encoding)

            return np.array(features)

        except Exception as e:
            logger.warning(f"[WARNING] Feature creation failed: {e}")
            return np.zeros(23)  # Default feature vector

    async def generate_trading_signal(self, symbol: str, timeframe: str = '5min') -> Dict[str, Any]:
        """
        Generate complete trading signal for a symbol

        Args:
            symbol: Stock symbol
            timeframe: Data timeframe

        Returns:
            Trading signal with all analysis components
        """
        # Perform comprehensive analysis
        analysis = await self.analyze_symbol_comprehensive(symbol, timeframe)

        if 'error' in analysis:
            return analysis

        # Extract trade recommendation
        trade_rec = analysis.get('trade_recommendation', {})

        # Create final signal
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'signal': analysis['signal'],
            'confidence': analysis['confidence'],
            'action': trade_rec.get('action', 'HOLD'),
            'position_size': trade_rec.get('position_size', 0),
            'stop_loss': trade_rec.get('stop_loss'),
            'take_profit': trade_rec.get('take_profit'),
            'reasoning': trade_rec.get('reasoning', []),
            'analysis_components': {
                'candlestick': analysis['components']['candlestick'],
                'vpa': analysis['components']['vpa'],
                'multimodal': analysis['components']['multimodal'],
                'market_context': analysis.get('market_context', {})
            },
            'similar_patterns': analysis.get('similar_patterns', [])
        }

        logger.info(f"[SIGNAL] {symbol}: {signal['action']} with {signal['confidence']:.3f} confidence")
        return signal

# Celery task integration

async def process_candlestick_signal(symbol: str, bar_data: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Celery-compatible task for candlestick signal processing

    Args:
        symbol: Stock symbol
        bar_data: Recent bar data

    Returns:
        Trading signal
    """
    try:
        # Initialize integrated system
        system = IntegratedCandlestickSystem()

        # Convert bar data to DataFrame
        df = pd.DataFrame(bar_data)
        df['date'] = pd.to_datetime(df.get('timestamp', df.get('date')))
        df = df.set_index('date').sort_index()

        # Generate signal
        signal = await system.generate_trading_signal(symbol)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'results/candlestick_signal_{symbol}_{timestamp}.json'

        with open(save_path, 'w') as f:
            json.dump(signal, f, indent=2, default=str)

        return signal

    except Exception as e:
        logger.error(f"[ERROR] Candlestick signal processing failed for {symbol}: {e}")
        return {
            'symbol': symbol,
            'error': str(e),
            'timestamp': datetime.now()
        }

# Example usage and testing
async def demo_integrated_system():
    """Demonstrate the integrated candlestick system"""
    logger.info("[DEMO] Integrated Candlestick System Demo")

    # Initialize system
    system = IntegratedCandlestickSystem()

    # Test with sample symbol
    symbol = 'AAPL'
    analysis = await system.analyze_symbol_comprehensive(symbol, '5min', 50)

    print(f"\nComprehensive Analysis for {symbol}:")
    print(f"Signal: {analysis.get('signal', 'unknown')}")
    print(f"Confidence: {analysis.get('confidence', 0):.3f}")
    print(f"Trade Action: {analysis.get('trade_recommendation', {}).get('action', 'HOLD')}")

    if 'trade_recommendation' in analysis:
        rec = analysis['trade_recommendation']
        print(f"Position Size: {rec.get('position_size', 0)}")
        print(f"Stop Loss: {rec.get('stop_loss')}")
        print(f"Take Profit: {rec.get('take_profit')}")
        print(f"Reasoning: {', '.join(rec.get('reasoning', []))}")

if __name__ == "__main__":
    asyncio.run(demo_integrated_system())