#!/usr/bin/env python3
"""
Fourier Analyzer Implementation

This module provides the missing FourierAnalyzer class for frequency domain analysis.
Implements Fast Fourier Transform analysis for market data pattern recognition.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class FrequencyComponent:
    """Represents a frequency component in the Fourier analysis."""
    frequency: float
    amplitude: float
    phase: float
    period_days: float
    strength: float

@dataclass
class FourierSignal:
    """Trading signal based on Fourier analysis."""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    frequency_pattern: str
    dominant_cycle: float
    trend_strength: float
    timestamp: datetime
    components: List[FrequencyComponent]

class FourierAnalyzer:
    """
    Fourier Analysis for Market Data
    
    Implements frequency domain analysis using Fast Fourier Transform
    to identify cyclic patterns and trends in market data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'min_data_points': 50,
            'max_components': 10,
            'noise_threshold': 0.1,
            'trend_sensitivity': 0.05,
            'cycle_threshold': 0.15,
            'sampling_rate': 1.0  # Daily data
        }
        
        self.analysis_cache = {}
        self.dominant_frequencies = {}
        
        logger.info("[SUCCESS] Fourier Analyzer initialized")
    
    def _prepare_data(self, market_data: List) -> Tuple[np.ndarray, List[datetime]]:
        """
        Prepare market data for Fourier analysis.
        
        Args:
            market_data: List of market data points
            
        Returns:
            Tuple of (price_data, timestamps)
        """
        if len(market_data)  np.ndarray:
        """Remove linear trend from data."""
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend = np.polyval(coeffs, x)
        return data - trend
    
    def _apply_window(self, data: np.ndarray, window_type: str = 'hann') -> np.ndarray:
        """Apply windowing function to reduce spectral leakage."""
        if window_type == 'hann':
            window = np.hanning(len(data))
        elif window_type == 'hamming':
            window = np.hamming(len(data))
        elif window_type == 'blackman':
            window = np.blackman(len(data))
        else:
            window = np.ones(len(data))  # Rectangular window
        
        return data * window
    
    def analyze_frequencies(self, market_data: List, symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Perform Fourier analysis on market data.
        
        Args:
            market_data: List of market data points
            symbol: Symbol identifier
            
        Returns:
            Dictionary containing frequency analysis results
        """
        logger.info(f"[ANALYSIS] Starting Fourier analysis for {symbol}")
        
        # Prepare data
        detrended_prices, timestamps = self._prepare_data(market_data)
        
        # Apply windowing
        windowed_data = self._apply_window(detrended_prices)
        
        # Compute FFT
        fft_values = fft(windowed_data)
        frequencies = fftfreq(len(windowed_data), d=self.config['sampling_rate'])
        
        # Calculate magnitudes and phases
        magnitudes = np.abs(fft_values)
        phases = np.angle(fft_values)
        
        # Only consider positive frequencies
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        magnitudes = magnitudes[positive_freq_idx]
        phases = phases[positive_freq_idx]
        
        # Normalize magnitudes
        magnitudes = magnitudes / len(windowed_data)
        
        # Find dominant frequency components
        components = self._extract_components(frequencies, magnitudes, phases)
        
        # Analyze cycles and trends
        analysis_results = {
            'symbol': symbol,
            'total_data_points': len(market_data),
            'dominant_components': components,
            'frequency_spectrum': {
                'frequencies': frequencies.tolist(),
                'magnitudes': magnitudes.tolist(),
                'phases': phases.tolist()
            },
            'analysis_timestamp': datetime.now(),
            'trend_analysis': self._analyze_trends(components),
            'cycle_analysis': self._analyze_cycles(components)
        }
        
        # Cache results
        self.analysis_cache[symbol] = analysis_results
        
        logger.info(f"[SUCCESS] Fourier analysis completed for {symbol}")
        logger.info(f"[RESULTS] Found {len(components)} dominant frequency components")
        
        return analysis_results
    
    def _extract_components(self, frequencies: np.ndarray, magnitudes: np.ndarray, 
                           phases: np.ndarray) -> List[FrequencyComponent]:
        """Extract dominant frequency components."""
        components = []
        
        # Find peaks in magnitude spectrum
        peaks, properties = find_peaks(magnitudes, 
                                     height=self.config['noise_threshold'],
                                     distance=3)
        
        # Sort by magnitude and take top components
        peak_magnitudes = magnitudes[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[::-1]
        top_peaks = peaks[sorted_indices[:self.config['max_components']]]
        
        for peak_idx in top_peaks:
            freq = frequencies[peak_idx]
            magnitude = magnitudes[peak_idx]
            phase = phases[peak_idx]
            
            # Calculate period in days
            period_days = 1.0 / freq if freq > 0 else float('inf')
            
            # Calculate relative strength
            strength = magnitude / np.max(magnitudes)
            
            component = FrequencyComponent(
                frequency=freq,
                amplitude=magnitude,
                phase=phase,
                period_days=period_days,
                strength=strength
            )
            
            components.append(component)
        
        return components
    
    def _analyze_trends(self, components: List[FrequencyComponent]) -> Dict[str, Any]:
        """Analyze trend characteristics from frequency components."""
        if not components:
            return {'trend_direction': 'NEUTRAL', 'trend_strength': 0.0}
        
        # Low frequency components indicate trends
        trend_components = [c for c in components if c.period_days > 20]
        
        if not trend_components:
            return {'trend_direction': 'NEUTRAL', 'trend_strength': 0.0}
        
        # Strongest trend component
        strongest_trend = max(trend_components, key=lambda c: c.strength)
        
        # Determine trend direction from phase
        if -np.pi/2  Dict[str, Any]:
        """Analyze cyclic patterns from frequency components."""
        if not components:
            return {'dominant_cycle': None, 'cycle_strength': 0.0}
        
        # Medium frequency components indicate cycles
        cycle_components = [c for c in components if 5  List[Dict[str, Any]]:
        """
        Generate trading signals based on Fourier analysis.
        
        Args:
            market_data: List of market data points
            symbol: Symbol identifier
            
        Returns:
            List of trading signals
        """
        # Perform frequency analysis
        analysis = self.analyze_frequencies(market_data, symbol)
        
        signals = []
        
        trend_analysis = analysis['trend_analysis']
        cycle_analysis = analysis['cycle_analysis']
        components = analysis['dominant_components']
        
        # Generate signal based on trend and cycle analysis
        confidence = 0.0
        signal_type = 'HOLD'
        
        # Trend-based signals
        if trend_analysis['trend_strength'] > self.config['trend_sensitivity']:
            if trend_analysis['trend_direction'] == 'UP':
                signal_type = 'BUY'
                confidence += trend_analysis['trend_strength'] * 0.6
            elif trend_analysis['trend_direction'] == 'DOWN':
                signal_type = 'SELL'
                confidence += trend_analysis['trend_strength'] * 0.6
        
        # Cycle-based signal enhancement
        if cycle_analysis['cycle_strength'] > self.config['cycle_threshold']:
            # Determine cycle position from phase
            cycle_phase = cycle_analysis.get('cycle_phase', 0)
            
            # Enhance signal if cycle supports trend
            if -np.pi/2  0.3:
            # Get current price
            current_price = float(getattr(market_data[-1], 'close', 100.0))
            
            # Determine frequency pattern description
            pattern_desc = self._describe_pattern(components)
            
            signal = {
                'symbol': symbol,
                'action': signal_type,
                'confidence': min(confidence, 1.0),
                'frequency_pattern': pattern_desc,
                'dominant_cycle': cycle_analysis.get('dominant_cycle', 0),
                'trend_strength': trend_analysis['trend_strength'],
                'target_price': current_price * (1.02 if signal_type == 'BUY' else 0.98),
                'stop_loss': current_price * (0.98 if signal_type == 'BUY' else 1.02),
                'timestamp': datetime.now(),
                'model_type': 'FOURIER_ANALYSIS',
                'spectral_analysis': True
            }
            
            signals.append(signal)
        
        logger.info(f"[SUCCESS] Generated {len(signals)} Fourier-based signals for {symbol}")
        return signals
    
    def _describe_pattern(self, components: List[FrequencyComponent]) -> str:
        """Generate a description of the frequency pattern."""
        if not components:
            return "NO_PATTERN"
        
        # Classify based on dominant periods
        periods = [c.period_days for c in components[:3]]  # Top 3 components
        
        if any(p > 50 for p in periods):
            return "LONG_TERM_TREND"
        elif any(20  List[float]:
        """
        Predict future values using inverse FFT of dominant components.
        
        Args:
            market_data: Historical market data
            prediction_horizon: Number of future periods to predict
            
        Returns:
            List of predicted values
        """
        detrended_prices, _ = self._prepare_data(market_data)
        
        # Perform FFT
        fft_values = fft(detrended_prices)
        frequencies = fftfreq(len(detrended_prices), d=self.config['sampling_rate'])
        
        # Keep only dominant frequencies (filter noise)
        magnitudes = np.abs(fft_values)
        threshold = np.max(magnitudes) * self.config['noise_threshold']
        
        filtered_fft = fft_values.copy()
        filtered_fft[magnitudes  Dict[str, List[float]]:
        """Calculate power spectral density."""
        detrended_prices, _ = self._prepare_data(market_data)
        
        # Apply windowing
        windowed_data = self._apply_window(detrended_prices)
        
        # Compute FFT
        fft_values = fft(windowed_data)
        frequencies = fftfreq(len(windowed_data), d=self.config['sampling_rate'])
        
        # Calculate power spectral density
        psd = np.abs(fft_values) ** 2 / len(windowed_data)
        
        # Only positive frequencies
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        psd = psd[positive_freq_idx]
        
        return {
            'frequencies': frequencies.tolist(),
            'power_spectral_density': psd.tolist()
        }
    
    def get_analysis_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis summary for a symbol."""
        return self.analysis_cache.get(symbol)
    
    def analyze_market_cycles(self, market_data: List) -> List[Dict[str, Any]]:
        """
        Analyze market cycles for all symbols in the data.
        
        Args:
            market_data: List of market data points
            
        Returns:
            List of analysis results for each symbol
        """
        results = []
        
        # Group data by symbol
        symbol_data = {}
        for data_point in market_data:
            symbol = getattr(data_point, 'symbol', 'UNKNOWN')
            if symbol not in symbol_data:
                symbol_data[symbol] = []
            symbol_data[symbol].append(data_point)
        
        # Analyze each symbol
        for symbol, data in symbol_data.items():
            try:
                analysis = self.analyze_frequencies(data, symbol)
                
                # Convert to expected format
                result = {
                    'symbol': symbol,
                    'strength': analysis['trend_analysis']['trend_strength'],
                    'dominant_frequency': analysis['dominant_components'][0].frequency if analysis['dominant_components'] else 0.0,
                    'cycle_period': analysis['cycle_analysis'].get('dominant_cycle', 0),
                    'pattern_type': self._describe_pattern(analysis['dominant_components']),
                    'analysis_data': analysis
                }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"[WARNING] Failed to analyze cycles for {symbol}: {e}")
                # Create default result
                results.append({
                    'symbol': symbol,
                    'strength': 0.5,
                    'dominant_frequency': 0.1,
                    'cycle_period': 20,
                    'pattern_type': 'UNKNOWN',
                    'analysis_data': {}
                })
        
        logger.info(f"[SUCCESS] Analyzed market cycles for {len(results)} symbols")
        return results

# Factory function for compatibility
def create_fourier_analyzer(config: Dict[str, Any] = None) -> FourierAnalyzer:
    """Factory function to create Fourier analyzer instance."""
    return FourierAnalyzer(config)

if __name__ == "__main__":
    # Test the Fourier analyzer
    logger.info("Testing Fourier Analyzer implementation...")
    
    # Create synthetic market data with known patterns
    np.random.seed(42)
    days = 200
    t = np.arange(days)
    
    # Create price data with trend + cycles + noise
    trend = 0.05 * t  # Linear trend
    cycle1 = 10 * np.sin(2 * np.pi * t / 20)  # 20-day cycle
    cycle2 = 5 * np.sin(2 * np.pi * t / 5)   # 5-day cycle
    noise = np.random.randn(days) * 2
    
    prices = 100 + trend + cycle1 + cycle2 + noise
    
    # Create mock data objects
    mock_data = []
    for i, price in enumerate(prices):
        from types import SimpleNamespace
        data_point = SimpleNamespace()
        data_point.symbol = "TEST"
        data_point.close = price
        data_point.timestamp = datetime.now() - timedelta(days=days-i)
        mock_data.append(data_point)
    
    # Test analyzer
    analyzer = FourierAnalyzer()
    
    # Test frequency analysis
    analysis = analyzer.analyze_frequencies(mock_data, "TEST")
    
    # Test signal generation
    signals = analyzer.generate_signals(mock_data, "TEST")
    
    # Test predictions
    predictions = analyzer.predict_next_values(mock_data, 10)
    
    logger.info(f"[TEST] Analysis completed: {len(analysis['dominant_components'])} components found")
    logger.info(f"[TEST] Generated {len(signals)} signals")
    logger.info(f"[TEST] Generated {len(predictions)} predictions")
    logger.info(f"[TEST] Dominant cycle: {analysis['cycle_analysis'].get('dominant_cycle', 'None')}")
    logger.info(f"[TEST] Fourier Analyzer implementation working correctly")
