"""
Fourier Analysis for Trading

This module implements Fourier transform-based analysis for market data:
- Market cycle detection using FFT
- Spectral density analysis for volatility prediction
- Harmonic pattern recognition
- Frequency domain filtering for noise reduction
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq, fftshift
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime
import logging

from ..core.data_structures import MarketData, TradingSignal
from ..core.base_classes import BaseTradingModel

logger = logging.getLogger(__name__)

@dataclass
class FrequencyComponent:
    """Represents a frequency component in the spectrum."""
    frequency: float
    amplitude: float
    phase: float
    period_days: float
    strength: float

@dataclass
class SpectralAnalysisConfig:
    """Configuration for spectral analysis."""
    window_size: int = 256
    overlap_ratio: float = 0.5
    min_period_days: float = 2.0
    max_period_days: float = 252.0
    noise_threshold: float = 0.1
    dominant_freq_count: int = 5

class FourierAnalyzer:
    """Fourier analysis for market data."""
    
    def __init__(self, config: SpectralAnalysisConfig = None):
        self.config = config or SpectralAnalysisConfig()
        
        logger.info("[SUCCESS] Fourier Analyzer initialized")
    
    def analyze_price_spectrum(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Analyze price spectrum using FFT."""
        if len(market_data) = self.config.min_period_days) & (periods  List[FrequencyComponent]:
        """Find dominant frequency components."""
        # Sort by power spectral density
        sorted_indices = np.argsort(psd)[::-1]
        
        dominant_components = []
        
        for i in range(min(self.config.dominant_freq_count, len(sorted_indices))):
            idx = sorted_indices[i]
            
            if psd[idx]  List[Dict[str, Any]]:
        """Detect market cycles using spectral analysis."""
        spectral_analysis = self.analyze_price_spectrum(market_data)
        
        if not spectral_analysis.get('dominant_frequencies'):
            return []
        
        cycles = []
        
        for component in spectral_analysis['dominant_frequencies']:
            # Only consider cycles with reasonable periods and strength
            if (component.period_days >= 5 and 
                component.period_days = 0.05):
                
                cycle_info = {
                    'period_days': component.period_days,
                    'strength': component.strength,
                    'current_phase': component.phase,
                    'cycle_type': self._classify_cycle(component.period_days),
                    'next_peak_in_days': self._estimate_next_peak(component),
                    'confidence': min(component.strength * 10, 1.0)  # Scale to 0-1
                }
                
                cycles.append(cycle_info)
        
        # Sort by strength
        cycles.sort(key=lambda x: x['strength'], reverse=True)
        
        logger.info(f"[SUCCESS] Detected {len(cycles)} market cycles")
        return cycles
    
    def _classify_cycle(self, period_days: float) -> str:
        """Classify cycle type based on period."""
        if period_days  float:
        """Estimate days until next cycle peak."""
        # Current phase normalized to 0-1 cycle
        normalized_phase = (component.phase % (2 * np.pi)) / (2 * np.pi)
        
        # Days to next peak (assuming peak at phase = 0)
        if normalized_phase  List[MarketData]:
        """Filter out high-frequency noise using low-pass filter."""
        if len(market_data)  np.ndarray:
        """Reconstruct signal from dominant frequency components."""
        if not spectral_analysis.get('dominant_frequencies'):
            return np.zeros(length)
        
        time_series = np.arange(length)
        reconstructed = np.zeros(length)
        
        for component in spectral_analysis['dominant_frequencies']:
            # Only use significant components
            if component.strength >= 0.02:  # At least 2% of total power
                frequency = component.frequency
                amplitude = component.amplitude
                phase = component.phase
                
                # Add frequency component
                component_signal = amplitude * np.cos(2 * np.pi * frequency * time_series + phase)
                reconstructed += component_signal
        
        return reconstructed

class HarmonicPatternDetector:
    """Detects harmonic patterns in price data using Fourier analysis."""
    
    def __init__(self, min_pattern_strength: float = 0.1):
        self.min_pattern_strength = min_pattern_strength
        
        logger.info("[SUCCESS] Harmonic Pattern Detector initialized")
    
    def detect_patterns(self, market_data: List[MarketData]) -> List[Dict[str, Any]]:
        """Detect harmonic patterns in market data."""
        if len(market_data) = self.min_pattern_strength:
                            pattern = {
                                'type': self._classify_harmonic_pattern(target_ratio),
                                'primary_frequency': freq1.frequency,
                                'harmonic_frequency': freq2.frequency,
                                'ratio': ratio,
                                'strength': pattern_strength,
                                'primary_period_days': freq1.period_days,
                                'harmonic_period_days': freq2.period_days,
                                'phase_relationship': abs(freq1.phase - freq2.phase),
                                'confidence': pattern_strength * (1 - abs(ratio - target_ratio))
                            }
                            
                            patterns.append(pattern)
        
        # Sort by confidence
        patterns.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"[SUCCESS] Detected {len(patterns)} harmonic patterns")
        return patterns
    
    def _classify_harmonic_pattern(self, ratio: float) -> str:
        """Classify harmonic pattern based on frequency ratio."""
        if abs(ratio - 2.0)  None:
        """Prepare data (no preparation needed for spectral analysis)."""
        logger.info(f"[SUCCESS] Data prepared: {len(market_data)} data points")
    
    def train(self, validation_data: Optional[List[MarketData]] = None) -> Dict[str, Any]:
        """Training not required for spectral analysis."""
        return {'message': 'Spectral analysis requires no training'}
    
    def predict(self, market_data: List[MarketData]) -> List[TradingSignal]:
        """Generate trading signals based on spectral analysis."""
        if len(market_data) = 0.3:  # Minimum confidence threshold
                
                signal_type = self._determine_signal_from_cycle(cycle, market_data[-1])
                
                if signal_type != "HOLD":
                    signal = TradingSignal(
                        symbol=market_data[-1].symbol,
                        signal_type=signal_type,
                        confidence=Decimal(str(cycle['confidence'])),
                        price_target=market_data[-1].close,
                        stop_loss=self._calculate_stop_loss(market_data[-1].close, signal_type, cycle),
                        take_profit=self._calculate_take_profit(market_data[-1].close, signal_type, cycle),
                        timestamp=market_data[-1].timestamp,
                        metadata={
                            'cycle_period_days': cycle['period_days'],
                            'cycle_strength': cycle['strength'],
                            'cycle_type': cycle['cycle_type'],
                            'next_peak_days': cycle['next_peak_in_days'],
                            'harmonic_patterns': len(patterns),
                            'spectral_power': spectral_analysis.get('spectral_power', 0),
                            'model_type': 'Spectral_Analysis'
                        }
                    )
                    
                    signals.append(signal)
        
        # Generate signals from harmonic patterns
        for pattern in patterns[:2]:  # Top 2 patterns
            if pattern['confidence'] >= 0.4:
                
                signal_type = self._determine_signal_from_pattern(pattern, market_data[-1])
                
                if signal_type != "HOLD":
                    signal = TradingSignal(
                        symbol=market_data[-1].symbol,
                        signal_type=signal_type,
                        confidence=Decimal(str(pattern['confidence'])),
                        price_target=market_data[-1].close,
                        stop_loss=self._calculate_stop_loss(market_data[-1].close, signal_type, pattern),
                        take_profit=self._calculate_take_profit(market_data[-1].close, signal_type, pattern),
                        timestamp=market_data[-1].timestamp,
                        metadata={
                            'pattern_type': pattern['type'],
                            'pattern_strength': pattern['strength'],
                            'frequency_ratio': pattern['ratio'],
                            'primary_period': pattern['primary_period_days'],
                            'harmonic_period': pattern['harmonic_period_days'],
                            'model_type': 'Harmonic_Pattern'
                        }
                    )
                    
                    signals.append(signal)
        
        logger.info(f"[SUCCESS] Generated {len(signals)} spectral trading signals")
        return signals
    
    def _determine_signal_from_cycle(self, cycle: Dict[str, Any], current_data: MarketData) -> str:
        """Determine trading signal from cycle analysis."""
        days_to_peak = cycle['next_peak_in_days']
        cycle_period = cycle['period_days']
        
        # Simple cycle-based strategy
        cycle_position = (days_to_peak / cycle_period) % 1.0
        
        # Buy near cycle trough, sell near cycle peak
        if 0.7  str:
        """Determine trading signal from harmonic pattern."""
        pattern_type = pattern['type']
        strength = pattern['strength']
        
        # Pattern-based signals
        if pattern_type in ['octave', 'perfect_fifth'] and strength > 0.15:
            return "BUY"  # Strong harmonic convergence
        elif pattern_type in ['sub_octave', 'sub_third'] and strength > 0.12:
            return "SELL"  # Harmonic divergence
        else:
            return "HOLD"
    
    def _calculate_stop_loss(self, current_price: Decimal, signal_type: str, analysis: Dict[str, Any]) -> Decimal:
        """Calculate stop loss based on cycle or pattern analysis."""
        base_stop_pct = 0.02  # 2% base stop loss
        
        # Adjust based on confidence/strength
        confidence = analysis.get('confidence', analysis.get('strength', 0.5))
        adjusted_stop_pct = base_stop_pct * (2 - confidence)  # Higher confidence = tighter stop
        
        if signal_type == "BUY":
            return current_price * Decimal(str(1 - adjusted_stop_pct))
        else:
            return current_price * Decimal(str(1 + adjusted_stop_pct))
    
    def _calculate_take_profit(self, current_price: Decimal, signal_type: str, analysis: Dict[str, Any]) -> Decimal:
        """Calculate take profit based on cycle or pattern analysis."""
        base_profit_pct = 0.04  # 4% base take profit
        
        # Adjust based on cycle period or pattern strength
        if 'period_days' in analysis:
            # Longer cycles = larger profit targets
            period_factor = min(analysis['period_days'] / 30.0, 2.0)
            adjusted_profit_pct = base_profit_pct * period_factor
        else:
            strength = analysis.get('strength', 0.5)
            adjusted_profit_pct = base_profit_pct * (1 + strength)
        
        if signal_type == "BUY":
            return current_price * Decimal(str(1 + adjusted_profit_pct))
        else:
            return current_price * Decimal(str(1 - adjusted_profit_pct))

def create_spectral_trading_model(config: Dict[str, Any] = None) -> SpectralTradingModel:
    """Factory function to create spectral trading model."""
    return SpectralTradingModel(config)
