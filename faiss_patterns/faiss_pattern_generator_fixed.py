#!/usr/bin/env python3
"""
FAISS Pattern Generator - Fixed Version
Generic pattern generation for FAISS-based trading strategies

Author: AI Assistant
Date: September 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class FAISSPatternGenerator:
    """
    Generic pattern generator for FAISS-based pattern recognition
    """
    
    def __init__(self):
        self.pattern_types = [
            'candlestick_pattern',
            'trend_pattern',
            'volume_pattern',
            'momentum_pattern',
            'support_resistance_pattern'
        ]
        
    def generate_pattern_vector(self, data: pd.DataFrame, pattern_type: str = 'generic') -> Optional[Dict[str, Any]]:
        """Generate a pattern vector from market data"""
        try:
            if len(data)  List[float]:
        """Create comprehensive pattern vector"""
        try:
            features = []
            
            # Price features
            features.extend(self._get_price_features(data))
            
            # Volume features  
            features.extend(self._get_volume_features(data))
            
            # Technical indicator features
            features.extend(self._get_technical_features(data))
            
            # Pattern-specific features
            features.extend(self._get_pattern_features(data))
            
            # Ensure consistent dimension (50 features)
            target_dim = 50
            while len(features)  List[float]:
        """Extract price-based features"""
        try:
            recent_data = data.tail(20)  # Last 20 periods
            
            features = []
            
            # Returns
            returns = recent_data['close'].pct_change().dropna()
            if len(returns) > 0:
                features.extend([
                    returns.mean(),           # Average return
                    returns.std(),            # Volatility
                    returns.iloc[-1],         # Last return
                    returns.iloc[-5:].mean()  # Recent average return
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Price position
            current_price = recent_data['close'].iloc[-1]
            high_20 = recent_data['high'].max()
            low_20 = recent_data['low'].min()
            
            features.extend([
                (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5,  # Position in range
                current_price / recent_data['close'].iloc[0] - 1 if recent_data['close'].iloc[0] > 0 else 0,  # 20-period return
                (high_20 - low_20) / current_price if current_price > 0 else 0  # Range as % of price
            ])
            
            # Trend features
            if len(recent_data) >= 10:
                x = np.arange(len(recent_data))
                slope = np.polyfit(x, recent_data['close'].values, 1)[0]
                trend_strength = slope / current_price if current_price > 0 else 0
                features.append(trend_strength)
            else:
                features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Price features failed: {e}")
            return [0.0] * 8
    
    def _get_volume_features(self, data: pd.DataFrame) -> List[float]:
        """Extract volume-based features"""
        try:
            recent_data = data.tail(20)
            
            features = []
            
            # Volume statistics
            volume_mean = recent_data['volume'].mean()
            volume_std = recent_data['volume'].std()
            current_volume = recent_data['volume'].iloc[-1]
            
            features.extend([
                current_volume / volume_mean if volume_mean > 0 else 1.0,  # Volume ratio
                volume_std / volume_mean if volume_mean > 0 else 0.0,      # Volume volatility
                recent_data['volume'].iloc[-5:].mean() / volume_mean if volume_mean > 0 else 1.0  # Recent volume ratio
            ])
            
            # Volume trend
            if len(recent_data) >= 10:
                x = np.arange(len(recent_data))
                volume_slope = np.polyfit(x, recent_data['volume'].values, 1)[0]
                volume_trend = volume_slope / volume_mean if volume_mean > 0 else 0
                features.append(volume_trend)
            else:
                features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Volume features failed: {e}")
            return [0.0] * 4
    
    def _get_technical_features(self, data: pd.DataFrame) -> List[float]:
        """Extract technical indicator features"""
        try:
            features = []
            
            # Simple moving averages
            if len(data) >= 20:
                sma_5 = data['close'].rolling(5).mean().iloc[-1]
                sma_10 = data['close'].rolling(10).mean().iloc[-1]
                sma_20 = data['close'].rolling(20).mean().iloc[-1]
                current_price = data['close'].iloc[-1]
                
                features.extend([
                    current_price / sma_5 - 1 if sma_5 > 0 else 0,
                    current_price / sma_10 - 1 if sma_10 > 0 else 0,
                    current_price / sma_20 - 1 if sma_20 > 0 else 0,
                    sma_5 / sma_10 - 1 if sma_10 > 0 else 0,
                    sma_10 / sma_20 - 1 if sma_20 > 0 else 0
                ])
            else:
                features.extend([0.0] * 5)
            
            # RSI-like momentum
            if len(data) >= 14:
                rsi = self._calculate_rsi(data['close'])
                features.append(rsi / 100)
            else:
                features.append(0.5)
            
            return features
            
        except Exception as e:
            logger.error(f"Technical features failed: {e}")
            return [0.0] * 6
    
    def _get_pattern_features(self, data: pd.DataFrame) -> List[float]:
        """Extract pattern-specific features"""
        try:
            features = []
            
            recent_data = data.tail(10)
            
            # Candlestick patterns
            body_sizes = []
            wick_ratios = []
            
            for _, row in recent_data.iterrows():
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                
                if open_price > 0:
                    body_size = abs(close_price - open_price) / open_price
                    total_range = (high_price - low_price) / open_price if high_price > low_price else 0
                    
                    body_sizes.append(body_size)
                    wick_ratios.append(total_range - body_size if total_range > body_size else 0)
            
            if body_sizes:
                features.extend([
                    np.mean(body_sizes),
                    np.std(body_sizes),
                    np.mean(wick_ratios),
                    np.std(wick_ratios)
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Support/resistance levels
            if len(data) >= 20:
                highs = data['high'].tail(20)
                lows = data['low'].tail(20)
                
                resistance_level = highs.quantile(0.9)
                support_level = lows.quantile(0.1)
                current_price = data['close'].iloc[-1]
                
                features.extend([
                    (current_price - support_level) / (resistance_level - support_level) if resistance_level > support_level else 0.5,
                    abs(current_price - resistance_level) / current_price if current_price > 0 else 0,
                    abs(current_price - support_level) / current_price if current_price > 0 else 0
                ])
            else:
                features.extend([0.5, 0.0, 0.0])
            
            return features
            
        except Exception as e:
            logger.error(f"Pattern features failed: {e}")
            return [0.0] * 7
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta  float:
        """Calculate confidence score for pattern"""
        try:
            # Base confidence on data quality and pattern strength
            data_quality = self._assess_data_quality(data)
            
            # Pattern strength based on volume and volatility
            recent_data = data.tail(10)
            volume_strength = min(1.0, recent_data['volume'].std() / recent_data['volume'].mean()) if recent_data['volume'].mean() > 0 else 0.5
            
            returns = recent_data['close'].pct_change().dropna()
            volatility_strength = min(1.0, returns.std() * 10) if len(returns) > 0 else 0.5
            
            confidence = (data_quality * 0.4 + volume_strength * 0.3 + volatility_strength * 0.3)
            return max(0.1, min(0.95, confidence))
            
        except:
            return 0.5
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess quality of input data"""
        try:
            quality_score = 1.0
            
            # Check for missing values
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            quality_score -= missing_ratio * 0.5
            
            # Check for zero volume
            zero_volume_ratio = (data['volume'] == 0).sum() / len(data)
            quality_score -= zero_volume_ratio * 0.3
            
            # Check for unrealistic price movements
            returns = data['close'].pct_change().dropna()
            if len(returns) > 0:
                extreme_moves = (abs(returns) > 0.20).sum() / len(returns)  # >20% moves
                quality_score -= extreme_moves * 0.2
            
            return max(0.1, min(1.0, quality_score))
            
        except:
            return 0.5
    
    def create_candlestick_vector(self, ohlc_data: pd.DataFrame, window: int = 5) -> np.ndarray:
        """Create candlestick pattern vector"""
        try:
            if len(ohlc_data)  Dict[str, Any]:
        """Extract features based on type"""
        try:
            if feature_type == 'price_only':
                features = self._get_price_features(data)
            elif feature_type == 'volume_only':
                features = self._get_volume_features(data)
            elif feature_type == 'technical_only':
                features = self._get_technical_features(data)
            else:  # comprehensive
                features = self._create_comprehensive_vector(data)
            
            return {
                'features': np.array(features, dtype=np.float32),
                'feature_type': feature_type,
                'dimension': len(features),
                'quality_score': self._assess_data_quality(data)
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {
                'features': np.array([0.0] * 50, dtype=np.float32),
                'feature_type': feature_type,
                'dimension': 50,
                'quality_score': 0.1
            }

# Test the pattern generator
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    prices = [100]
    for _ in range(99):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [1000000 + np.random.randint(0, 5000000) for _ in prices]
    })
    
    generator = FAISSPatternGenerator()
    pattern = generator.generate_pattern_vector(sample_data)
    
    if pattern:
        print(f"Pattern generated successfully!")
        print(f"Vector dimension: {len(pattern['feature_vector'])}")
        print(f"Confidence: {pattern['confidence']:.2f}")
        print(f"Data quality: {pattern['data_quality']:.2f}")
    else:
        print("Pattern generation failed")
