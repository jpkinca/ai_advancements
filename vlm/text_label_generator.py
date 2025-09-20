#!/usr/bin/env python3
"""
Text Label Generation Pipeline for VLM Dataset

This module automates the generation of descriptive text labels for chart images,
combining technical pattern detection with LLM augmentation for rich, natural language descriptions.

Features:
- Technical pattern detection (candlesticks, indicators)
- LLM-powered description augmentation
- Manual override and validation hooks
- Batch processing capabilities
- Confidence scoring for generated labels
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextLabelGenerator:
    """
    Generates descriptive text labels for chart images using pattern detection + LLM augmentation
    """

    def __init__(self,
                 llm_provider: str = 'openai',
                 model_name: str = 'gpt-4o-mini',
                 confidence_threshold: float = 0.7):
        """
        Initialize the label generator

        Args:
            llm_provider: 'openai' or 'local' for LLM augmentation
            model_name: Specific model name
            confidence_threshold: Minimum confidence for auto-generated labels
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        # Initialize LLM client
        self.llm_client = None
        self._init_llm_client()

        logger.info(f"[INIT] Text Label Generator ({llm_provider}:{model_name})")

    def _init_llm_client(self):
        """Initialize LLM client based on provider"""
        if self.llm_provider == 'openai':
            try:
                from openai import OpenAI
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    logger.warning("[WARNING] OPENAI_API_KEY not set, falling back to pattern-only")
                    self.llm_provider = 'none'
                else:
                    self.llm_client = OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("[WARNING] OpenAI package not available, falling back to pattern-only")
                self.llm_provider = 'none'
        elif self.llm_provider == 'local':
            # Placeholder for local LLM integration
            logger.warning("[WARNING] Local LLM not implemented yet, falling back to pattern-only")
            self.llm_provider = 'none'

    def generate_label(self,
                      ohlcv_data: pd.DataFrame,
                      symbol: str,
                      timeframe: str,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive text label for a chart

        Args:
            ohlcv_data: OHLCV data used for the chart
            symbol: Trading symbol
            timeframe: Timeframe string
            metadata: Chart metadata from image generator

        Returns:
            Dictionary with label components and confidence scores
        """
        try:
            # Detect technical patterns
            patterns = self._detect_patterns(ohlcv_data)

            # Generate base description
            base_description = self._generate_base_description(ohlcv_data, symbol, timeframe, patterns)

            # Augment with LLM if available
            if self.llm_provider != 'none':
                augmented_description = self._augment_with_llm(base_description, ohlcv_data, metadata)
            else:
                augmented_description = base_description

            # Calculate confidence
            confidence = self._calculate_confidence(patterns, augmented_description)

            # Generate structured label
            label = {
                'symbol': symbol,
                'timeframe': timeframe,
                'patterns_detected': patterns,
                'base_description': base_description,
                'augmented_description': augmented_description,
                'confidence_score': confidence,
                'generated_at': datetime.now().isoformat(),
                'needs_review': confidence < self.confidence_threshold
            }

            logger.info(f"[LABEL] Generated for {symbol} (conf: {confidence:.3f})")
            return label

        except Exception as e:
            logger.error(f"[ERROR] Failed to generate label: {e}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'error': str(e),
                'confidence_score': 0.0,
                'needs_review': True
            }

    def _detect_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect technical patterns in the data"""
        patterns = []

        try:
            # Basic trend patterns
            trend = self._detect_trend(df)
            if trend:
                patterns.append(trend)

            # Candlestick patterns
            candle_patterns = self._detect_candlestick_patterns(df)
            patterns.extend(candle_patterns)

            # Indicator patterns
            indicator_patterns = self._detect_indicator_patterns(df)
            patterns.extend(indicator_patterns)

            # Volume patterns
            volume_pattern = self._detect_volume_pattern(df)
            if volume_pattern:
                patterns.append(volume_pattern)

        except Exception as e:
            logger.warning(f"[WARNING] Pattern detection error: {e}")

        return patterns

    def _detect_trend(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect overall trend"""
        if len(df) < 20:
            return None

        # Simple trend detection using moving averages
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()

        recent_sma_20 = sma_20.iloc[-1]
        recent_sma_50 = sma_50.iloc[-1]
        prev_sma_20 = sma_20.iloc[-2] if len(sma_20) > 1 else recent_sma_20
        prev_sma_50 = sma_50.iloc[-2] if len(sma_50) > 1 else recent_sma_50

        if recent_sma_20 > recent_sma_50 and prev_sma_20 <= prev_sma_50:
            return {'type': 'trend', 'direction': 'bullish', 'strength': 'moderate'}
        elif recent_sma_20 < recent_sma_50 and prev_sma_20 >= prev_sma_50:
            return {'type': 'trend', 'direction': 'bearish', 'strength': 'moderate'}

        return None

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect basic candlestick patterns"""
        patterns = []

        if len(df) < 2:
            return patterns

        # Engulfing patterns
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]

            # Bullish engulfing
            if (current['close'] > current['open'] and
                previous['close'] < previous['open'] and
                current['close'] > previous['open'] and
                current['open'] < previous['close']):
                patterns.append({
                    'type': 'candlestick',
                    'pattern': 'bullish_engulfing',
                    'strength': 'strong'
                })

            # Bearish engulfing
            elif (current['close'] < current['open'] and
                  previous['close'] > previous['open'] and
                  current['close'] < previous['open'] and
                  current['open'] > previous['close']):
                patterns.append({
                    'type': 'candlestick',
                    'pattern': 'bearish_engulfing',
                    'strength': 'strong'
                })

        return patterns

    def _detect_indicator_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect indicator-based patterns"""
        patterns = []

        if len(df) < 50:
            return patterns

        # RSI patterns
        rsi = self._calculate_rsi(df['close'])
        if len(rsi) > 0:
            recent_rsi = rsi.iloc[-1]
            if recent_rsi < 30:
                patterns.append({'type': 'indicator', 'indicator': 'rsi', 'signal': 'oversold'})
            elif recent_rsi > 70:
                patterns.append({'type': 'indicator', 'indicator': 'rsi', 'signal': 'overbought'})

        # MACD patterns
        macd_data = self._calculate_macd(df['close'])
        if macd_data:
            macd, signal = macd_data
            if len(macd) > 1 and len(signal) > 1:
                if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
                    patterns.append({'type': 'indicator', 'indicator': 'macd', 'signal': 'bullish_crossover'})
                elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
                    patterns.append({'type': 'indicator', 'indicator': 'macd', 'signal': 'bearish_crossover'})

        return patterns

    def _detect_volume_pattern(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect volume patterns"""
        if len(df) < 20:
            return None

        avg_volume = df['volume'].rolling(20).mean()
        recent_volume = df['volume'].iloc[-1]
        avg_recent = avg_volume.iloc[-1]

        if recent_volume > avg_recent * 1.5:
            return {'type': 'volume', 'pattern': 'high_volume', 'ratio': recent_volume / avg_recent}
        elif recent_volume < avg_recent * 0.5:
            return {'type': 'volume', 'pattern': 'low_volume', 'ratio': recent_volume / avg_recent}

        return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> Optional[Tuple[pd.Series, pd.Series]]:
        """Calculate MACD indicator"""
        if len(prices) < 26:
            return None

        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()

        return macd, signal

    def _generate_base_description(self, df: pd.DataFrame, symbol: str, timeframe: str,
                                 patterns: List[Dict[str, Any]]) -> str:
        """Generate base description from detected patterns"""
        description_parts = [f"{symbol} {timeframe} chart showing"]

        # Add trend information
        trend_patterns = [p for p in patterns if p.get('type') == 'trend']
        if trend_patterns:
            trend = trend_patterns[0]
            description_parts.append(f"a {trend['direction']} trend")
        else:
            description_parts.append("mixed price action")

        # Add key patterns
        pattern_descriptions = []
        for pattern in patterns:
            if pattern['type'] == 'candlestick':
                pattern_descriptions.append(f"{pattern['pattern'].replace('_', ' ')} pattern")
            elif pattern['type'] == 'indicator':
                pattern_descriptions.append(f"{pattern['indicator']} {pattern['signal']}")

        if pattern_descriptions:
            description_parts.append("with")
            description_parts.append(", ".join(pattern_descriptions[:3]))  # Limit to 3

        # Add price context
        if not df.empty:
            current_price = df['close'].iloc[-1]
            price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            description_parts.append(f"Current price: ${current_price:.2f} ({price_change:+.2f}% change)")
            description_parts.append("over the period")

        return " ".join(description_parts)

    def _augment_with_llm(self, base_description: str, df: pd.DataFrame,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Augment base description with LLM"""
        if not self.llm_client:
            return base_description

        try:
            # Prepare context for LLM
            context = {
                'base_description': base_description,
                'price_stats': {
                    'current': float(df['close'].iloc[-1]) if not df.empty else None,
                    'high': float(df['high'].max()) if not df.empty else None,
                    'low': float(df['low'].min()) if not df.empty else None,
                    'change_pct': float((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if not df.empty else None
                },
                'volume_stats': {
                    'average': float(df['volume'].mean()) if not df.empty else None,
                    'recent': float(df['volume'].iloc[-1]) if not df.empty else None
                }
            }

            prompt = f"""
            Based on this technical analysis context, provide a natural, concise description of the chart:

            Context: {json.dumps(context, indent=2)}

            Generate a 1-2 sentence description that captures the key market dynamics and potential trading implications.
            Focus on price action, momentum, and any notable patterns.
            """

            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )

            augmented = response.choices[0].message.content.strip()
            return f"{base_description}. {augmented}"

        except Exception as e:
            logger.warning(f"[WARNING] LLM augmentation failed: {e}")
            return base_description

    def _calculate_confidence(self, patterns: List[Dict[str, Any]], description: str) -> float:
        """Calculate confidence score for the generated label"""
        confidence = 0.5  # Base confidence

        # Boost for multiple patterns
        confidence += min(len(patterns) * 0.1, 0.3)

        # Boost for strong patterns
        strong_patterns = [p for p in patterns if p.get('strength') == 'strong']
        confidence += min(len(strong_patterns) * 0.1, 0.2)

        # Boost for LLM augmentation
        if self.llm_provider != 'none' and len(description.split('.')) > 1:
            confidence += 0.2

        return min(confidence, 1.0)

    def batch_generate_labels(self,
                            data_dict: Dict[str, pd.DataFrame],
                            symbols: List[str],
                            timeframes: List[str],
                            metadata_dict: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Generate labels for multiple charts in batch

        Args:
            data_dict: Dictionary of {symbol_timeframe: ohlcv_data}
            symbols: List of symbols to process
            timeframes: List of timeframes to process
            metadata_dict: Optional metadata dictionary

        Returns:
            List of label dictionaries
        """
        labels = []

        for symbol in symbols:
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"
                if key not in data_dict:
                    continue

                metadata = metadata_dict.get(key) if metadata_dict else None

                try:
                    label = self.generate_label(
                        data_dict[key], symbol, timeframe, metadata
                    )
                    labels.append(label)
                except Exception as e:
                    logger.error(f"[ERROR] Failed to generate label for {key}: {e}")

        logger.info(f"[BATCH] Generated {len(labels)} labels")
        return labels

    def save_labels_to_file(self, labels: List[Dict[str, Any]], output_path: str):
        """Save generated labels to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(labels, f, indent=2)
            logger.info(f"[SAVE] Labels saved to {output_path}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save labels: {e}")

# Example usage
def example_usage():
    """Example usage of the text label generator"""
    from chart_image_generator import create_sample_data

    generator = TextLabelGenerator()

    # Create sample data
    sample_data = create_sample_data('AAPL', 50)

    # Generate label
    label = generator.generate_label(sample_data, 'AAPL', '1D')

    print("Generated Label:")
    print(json.dumps(label, indent=2))

if __name__ == "__main__":
    example_usage()