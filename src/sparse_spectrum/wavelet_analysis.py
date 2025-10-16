"""
Wavelet Analysis for Trading

This module implements wavelet-based analysis for market data:
- Multi-resolution price analysis
- Wavelet-based denoising techniques
- Time-frequency trading signals
- Market microstructure analysis
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pywt
from scipy import signal
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime
import logging

from ..core.data_structures import MarketData, TradingSignal
from ..core.base_classes import BaseTradingModel

logger = logging.getLogger(__name__)

@dataclass
class WaveletConfig:
    """Configuration for wavelet analysis."""
    wavelet_family: str = 'db4'  # Daubechies wavelet
    decomposition_levels: int = 6
    noise_threshold_method: str = 'soft'  # 'soft' or 'hard'
    threshold_mode: str = 'greater'
    denoising_sigma: Optional[float] = None
    min_energy_ratio: float = 0.05

@dataclass
class WaveletComponent:
    """Represents a wavelet decomposition component."""
    level: int
    frequency_band: Tuple[float, float]
    coefficients: np.ndarray
    energy: float
    energy_ratio: float
    period_range_days: Tuple[float, float]

class WaveletAnalyzer:
    """Wavelet analysis for market data."""
    
    def __init__(self, config: WaveletConfig = None):
        self.config = config or WaveletConfig()
        
        # Validate wavelet family
        if self.config.wavelet_family not in pywt.wavelist():
            logger.warning(f"[WARNING] Wavelet {self.config.wavelet_family} not found, using 'db4'")
            self.config.wavelet_family = 'db4'
        
        logger.info(f"[SUCCESS] Wavelet Analyzer initialized with {self.config.wavelet_family}")
    
    def decompose_price_series(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Decompose price series using wavelet transform."""
        significant_coeffs = coeffs[np.abs(coeffs) >= self.config.min_energy_ratio]
        }
        
        logger.info(f"[SUCCESS] Wavelet decomposition completed: {len(components)} components, {len(decomposition_result['significant_components'])} significant")
        return decomposition_result
    
    def denoise_price_series(self, market_data: List[MarketData]) -> List[MarketData]:
        """Denoise price series using wavelet thresholding."""
        if len(market_data)  float:
        """Calculate reconstruction error."""
        reconstructed = pywt.waverec(coeffs, self.config.wavelet_family)
        
        # Handle length mismatch
        min_length = min(len(original), len(reconstructed))
        original_trimmed = original[:min_length]
        reconstructed_trimmed = reconstructed[:min_length]
        
        mse = np.mean((original_trimmed - reconstructed_trimmed) ** 2)
        return float(mse)
    
    def analyze_time_frequency(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Analyze time-frequency characteristics using continuous wavelet transform."""
        if len(market_data)  List[Dict[str, float]]:
        """Find dominant periods in the time-frequency representation."""
        # Average power across time for each period
        avg_power = np.mean(power, axis=1)
        
        # Find peaks in the power spectrum
        peaks, properties = signal.find_peaks(avg_power, height=np.percentile(avg_power, 75))
        
        dominant_periods = []
        for peak_idx in peaks:
            period = periods[peak_idx]
            power_value = avg_power[peak_idx]
            
            if 2  np.ndarray:
        """Calculate time-varying energy across all frequencies."""
        return np.sum(power, axis=0)
    
    def _analyze_frequency_bands(self, power: np.ndarray, periods: np.ndarray) -> Dict[str, float]:
        """Analyze energy in different frequency bands."""
        # Define frequency bands based on periods
        bands = {
            'high_frequency': (2, 10),      # 2-10 days (short-term)
            'medium_frequency': (10, 50),   # 10-50 days (medium-term)
            'low_frequency': (50, 252)      # 50-252 days (long-term)
        }
        
        band_energy = {}
        total_energy = np.sum(power)
        
        for band_name, (min_period, max_period) in bands.items():
            # Find indices corresponding to this period range
            band_indices = (periods >= min_period) & (periods  List[TradingSignal]:
        """Generate trading signals from wavelet analysis."""
        if len(market_data) = 0.3:  # Minimum confidence threshold
                    signal = TradingSignal(
                        symbol=market_data[-1].symbol,
                        signal_type=signal_type,
                        confidence=Decimal(str(confidence)),
                        price_target=market_data[-1].close,
                        stop_loss=self._calculate_stop_loss(market_data[-1].close, signal_type, component),
                        take_profit=self._calculate_take_profit(market_data[-1].close, signal_type, component),
                        timestamp=market_data[-1].timestamp,
                        metadata={
                            'wavelet_level': component.level,
                            'energy_ratio': component.energy_ratio,
                            'period_range': component.period_range_days,
                            'frequency_band': component.frequency_band,
                            'component_type': 'approximation' if component.level == 0 else 'detail',
                            'model_type': 'Wavelet_Analysis'
                        }
                    )
                    
                    signals.append(signal)
        
        # Generate signals from time-frequency analysis
        if time_freq and time_freq.get('dominant_periods'):
            tf_signal = self._generate_time_frequency_signal(market_data, time_freq)
            if tf_signal:
                signals.append(tf_signal)
        
        logger.info(f"[SUCCESS] Generated {len(signals)} wavelet-based trading signals")
        return signals
    
    def _analyze_component_trend(self, component: WaveletComponent, market_data: List[MarketData]) -> str:
        """Analyze trend in wavelet component."""
        coeffs = component.coefficients
        
        if len(coeffs)  0.1:
                return "BUY"
            elif normalized_slope  0.2 and component.energy_ratio > 0.1:
                return "BUY"
            elif normalized_slope  0.1:
                return "SELL"
        
        return "HOLD"
    
    def _generate_time_frequency_signal(self, market_data: List[MarketData], time_freq: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate signal from time-frequency analysis."""
        dominant_periods = time_freq.get('dominant_periods', [])
        
        if not dominant_periods:
            return None
        
        # Focus on the most dominant period
        strongest_period = dominant_periods[0]
        
        if strongest_period['relative_power']  0.5:
            signal_type = "BUY"
        elif normalized_energy_trend  Decimal:
        """Calculate stop loss based on wavelet analysis."""
        if component:
            # Adjust stop loss based on component period range
            avg_period = np.mean(component.period_range_days) if component.period_range_days[1] != np.inf else component.period_range_days[0]
            period_factor = min(avg_period / 20.0, 3.0)  # Scale factor based on period
            stop_pct = 0.015 * period_factor  # Base 1.5% adjusted by period
        else:
            stop_pct = 0.02  # Default 2%
        
        if signal_type == "BUY":
            return current_price * Decimal(str(1 - stop_pct))
        else:
            return current_price * Decimal(str(1 + stop_pct))
    
    def _calculate_take_profit(self, current_price: Decimal, signal_type: str, component: Optional[WaveletComponent]) -> Decimal:
        """Calculate take profit based on wavelet analysis."""
        if component:
            # Adjust take profit based on energy ratio and period
            energy_factor = component.energy_ratio * 2  # Higher energy = larger target
            avg_period = np.mean(component.period_range_days) if component.period_range_days[1] != np.inf else component.period_range_days[0]
            period_factor = min(avg_period / 15.0, 2.5)
            profit_pct = 0.025 * energy_factor * period_factor
        else:
            profit_pct = 0.04  # Default 4%
        
        profit_pct = max(0.02, min(profit_pct, 0.08))  # Clamp between 2% and 8%
        
        if signal_type == "BUY":
            return current_price * Decimal(str(1 + profit_pct))
        else:
            return current_price * Decimal(str(1 - profit_pct))

class WaveletTradingModel(BaseTradingModel):
    """Trading model based on wavelet analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.wavelet_config = WaveletConfig(**self.config.get('wavelet', {}))
        
        self.signal_generator = WaveletTradingSignalGenerator(self.wavelet_config)
        self.is_trained = True  # No training required
        
        logger.info("[SUCCESS] Wavelet Trading Model initialized")
    
    def prepare_data(self, market_data: List[MarketData]) -> None:
        """Prepare data (no preparation needed for wavelet analysis)."""
        logger.info(f"[SUCCESS] Data prepared: {len(market_data)} data points")
    
    def train(self, validation_data: Optional[List[MarketData]] = None) -> Dict[str, Any]:
        """Training not required for wavelet analysis."""
        return {'message': 'Wavelet analysis requires no training'}
    
    def predict(self, market_data: List[MarketData]) -> List[TradingSignal]:
        """Generate trading signals using wavelet analysis."""
        return self.signal_generator.generate_signals(market_data)

def create_wavelet_trading_model(config: Dict[str, Any] = None) -> WaveletTradingModel:
    """Factory function to create wavelet trading model."""
    return WaveletTradingModel(config)
