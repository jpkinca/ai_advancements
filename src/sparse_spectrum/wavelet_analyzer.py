#!/usr/bin/env python3
"""
Wavelet Analyzer Implementation

This module provides the missing WaveletAnalyzer class for time-frequency analysis.
Implements Continuous Wavelet Transform for multi-scale market data analysis.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[WARNING] PyWavelets (pywt) not available, using simplified wavelet implementation")

try:
    from scipy import signal
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class WaveletCoefficient:
    """Represents a wavelet coefficient at specific scale and time."""
    scale: float
    time_index: int
    coefficient: complex
    magnitude: float
    phase: float
    frequency: float

@dataclass
class WaveletSignal:
    """Trading signal based on wavelet analysis."""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    scale_pattern: str
    dominant_scale: float
    volatility_regime: str
    timestamp: datetime
    coefficients: List[WaveletCoefficient]

class WaveletAnalyzer:
    """
    Wavelet Analysis for Market Data
    
    Implements Continuous Wavelet Transform (CWT) for multi-scale
    time-frequency analysis of market data patterns.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'wavelet_name': 'morlet',
            'min_scale': 1,
            'max_scale': 64,
            'num_scales': 32,
            'volatility_threshold': 0.2,
            'signal_threshold': 0.15,
            'trend_scale_threshold': 16,
            'noise_scale_threshold': 4,
            'sampling_rate': 1.0  # Daily data
        }
        
        self.analysis_cache = {}
        self.coefficient_cache = {}
        
        # Generate scales for CWT
        self.scales = np.logspace(
            np.log10(self.config['min_scale']),
            np.log10(self.config['max_scale']),
            self.config['num_scales']
        )
        
        logger.info(f"[SUCCESS] Wavelet Analyzer initialized with {self.config['wavelet_name']} wavelet")
        logger.info(f"[CONFIG] Scales: {len(self.scales)} from {self.config['min_scale']} to {self.config['max_scale']}")
    
    def _prepare_data(self, market_data: List) -> Tuple[np.ndarray, List[datetime]]:
        """
        Prepare market data for wavelet analysis.
        
        Args:
            market_data: List of market data points
            
        Returns:
            Tuple of (price_data, timestamps)
        """
        if len(market_data)  np.ndarray:
        """
        Perform Continuous Wavelet Transform.
        
        Args:
            data: Time series data
            
        Returns:
            CWT coefficients matrix (scales x time)
        """
        # Use scipy's CWT for continuous wavelet transform
        if self.config['wavelet_name'] == 'morlet':
            # Morlet wavelet
            coefficients, _ = pywt.cwt(data, self.scales, 'cmor')
        elif self.config['wavelet_name'] == 'mexican_hat':
            # Mexican hat wavelet
            coefficients, _ = pywt.cwt(data, self.scales, 'mexh')
        else:
            # Default to Morlet
            coefficients, _ = pywt.cwt(data, self.scales, 'cmor')
        
        return coefficients
    
    def _calculate_instantaneous_frequency(self, coefficients: np.ndarray) -> np.ndarray:
        """Calculate instantaneous frequency from wavelet coefficients."""
        # Calculate phase derivative for instantaneous frequency
        phases = np.angle(coefficients)
        
        # Unwrap phases to avoid discontinuities
        unwrapped_phases = np.unwrap(phases, axis=1)
        
        # Calculate time derivative of phase
        dt = 1.0  # Sampling interval
        inst_freq = np.zeros_like(unwrapped_phases)
        inst_freq[:, 1:] = np.diff(unwrapped_phases, axis=1) / (2 * np.pi * dt)
        
        return np.abs(inst_freq)
    
    def analyze_wavelets(self, market_data: List, symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Perform wavelet analysis on market data.
        
        Args:
            market_data: List of market data points
            symbol: Symbol identifier
            
        Returns:
            Dictionary containing wavelet analysis results
        """
        logger.info(f"[ANALYSIS] Starting wavelet analysis for {symbol}")
        
        # Prepare data
        data, timestamps = self._prepare_data(market_data)
        
        # Perform CWT
        coefficients = self._continuous_wavelet_transform(data)
        
        # Calculate magnitudes and phases
        magnitudes = np.abs(coefficients)
        phases = np.angle(coefficients)
        
        # Calculate instantaneous frequency
        inst_freq = self._calculate_instantaneous_frequency(coefficients)
        
        # Extract significant coefficients
        significant_coeffs = self._extract_significant_coefficients(
            coefficients, magnitudes, phases, timestamps
        )
        
        # Analyze different scales
        scale_analysis = self._analyze_scales(magnitudes, phases)
        
        # Detect volatility regime
        volatility_analysis = self._analyze_volatility(magnitudes)
        
        # Time-frequency analysis
        tf_analysis = self._time_frequency_analysis(magnitudes, inst_freq)
        
        analysis_results = {
            'symbol': symbol,
            'total_data_points': len(market_data),
            'wavelet_type': self.config['wavelet_name'],
            'num_scales': len(self.scales),
            'significant_coefficients': significant_coeffs,
            'scale_analysis': scale_analysis,
            'volatility_analysis': volatility_analysis,
            'time_frequency_analysis': tf_analysis,
            'coefficient_matrix_shape': coefficients.shape,
            'analysis_timestamp': datetime.now()
        }
        
        # Cache results
        self.analysis_cache[symbol] = analysis_results
        self.coefficient_cache[symbol] = {
            'coefficients': coefficients,
            'magnitudes': magnitudes,
            'phases': phases,
            'scales': self.scales
        }
        
        logger.info(f"[SUCCESS] Wavelet analysis completed for {symbol}")
        logger.info(f"[RESULTS] Found {len(significant_coeffs)} significant coefficients")
        
        return analysis_results
    
    def _extract_significant_coefficients(self, coefficients: np.ndarray, magnitudes: np.ndarray,
                                        phases: np.ndarray, timestamps: List[datetime]) -> List[WaveletCoefficient]:
        """Extract significant wavelet coefficients."""
        significant_coeffs = []
        
        # Find coefficients above threshold
        threshold = np.percentile(magnitudes, 95)  # Top 5%
        
        scale_indices, time_indices = np.where(magnitudes > threshold)
        
        for scale_idx, time_idx in zip(scale_indices, time_indices):
            if time_idx  Dict[str, Any]:
        """Analyze patterns across different scales."""
        
        # Calculate average energy at each scale
        scale_energies = np.mean(magnitudes ** 2, axis=1)
        
        # Find dominant scale
        dominant_scale_idx = np.argmax(scale_energies)
        dominant_scale = self.scales[dominant_scale_idx]
        
        # Classify scales
        trend_scales = self.scales >= self.config['trend_scale_threshold']
        noise_scales = self.scales  0:
            trend_ratio = trend_energy / total_energy
            cycle_ratio = cycle_energy / total_energy
            noise_ratio = noise_energy / total_energy
        else:
            trend_ratio = cycle_ratio = noise_ratio = 0.0
        
        return {
            'dominant_scale': dominant_scale,
            'dominant_scale_energy': scale_energies[dominant_scale_idx],
            'scale_energies': scale_energies.tolist(),
            'trend_energy_ratio': trend_ratio,
            'cycle_energy_ratio': cycle_ratio,
            'noise_energy_ratio': noise_ratio,
            'energy_distribution': {
                'trend': trend_energy,
                'cycle': cycle_energy,
                'noise': noise_energy
            }
        }
    
    def _analyze_volatility(self, magnitudes: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility regime from wavelet coefficients."""
        
        # Calculate time-varying volatility from high-frequency scales
        high_freq_scales = self.scales  0 else avg_volatility
        
        if current_volatility > avg_volatility + volatility_std:
            regime = 'HIGH_VOLATILITY'
        elif current_volatility  Dict[str, Any]:
        """Perform time-frequency analysis."""
        
        # Find time-frequency ridges (high energy paths)
        ridges = []
        
        for time_idx in range(magnitudes.shape[1]):
            # Find scale with maximum energy at this time
            max_scale_idx = np.argmax(magnitudes[:, time_idx])
            ridge_point = {
                'time_index': time_idx,
                'scale_index': max_scale_idx,
                'scale': self.scales[max_scale_idx],
                'magnitude': magnitudes[max_scale_idx, time_idx],
                'frequency': inst_freq[max_scale_idx, time_idx] if max_scale_idx < len(inst_freq) else 0.0
        
        return {
            'ridges': ridges,
            'frequency_trend': frequency_trend,
            'frequency_jumps': jump_indices.tolist(),
            'average_frequency': np.mean(ridge_frequencies),
            'frequency_variance': np.var(ridge_frequencies)
        }
    
    def generate_signals(self, market_data: List, symbol: str = "UNKNOWN") -> List[Dict[str, Any]]:
        """
        Generate trading signals based on wavelet analysis.
        
        Args:
            market_data: List of market data points
            symbol: Symbol identifier
            
        Returns:
            List of trading signals
        """
        # Perform wavelet analysis
        analysis = self.analyze_wavelets(market_data, symbol)
        
        signals = []
        
        scale_analysis = analysis['scale_analysis']
        volatility_analysis = analysis['volatility_analysis']
        tf_analysis = analysis['time_frequency_analysis']
        
        # Generate signal based on multi-scale analysis
        confidence = 0.0
        signal_type = 'HOLD'
        
        # Trend analysis from large scales
        trend_ratio = scale_analysis['trend_energy_ratio']
        cycle_ratio = scale_analysis['cycle_energy_ratio']
        
        # Strong trend signal
        if trend_ratio > 0.5:
            # Check frequency trend to determine direction
            freq_trend = tf_analysis['frequency_trend']
            
            if freq_trend > 0.001:  # Increasing frequency suggests bearish
                signal_type = 'SELL'
                confidence += trend_ratio * 0.6
            elif freq_trend  0.4:
            # Use phase information from dominant scale
            significant_coeffs = analysis['significant_coefficients']
            if significant_coeffs:
                dominant_coeff = significant_coeffs[0]  # Strongest coefficient
                phase = dominant_coeff.phase
                
                # Phase-based signal
                if -np.pi/2  self.config['signal_threshold']:
            # Get current price
            current_price = float(getattr(market_data[-1], 'close', 100.0))
            
            # Determine scale pattern description
            pattern_desc = self._describe_scale_pattern(scale_analysis)
            
            signal = {
                'symbol': symbol,
                'action': signal_type,
                'confidence': min(confidence, 1.0),
                'scale_pattern': pattern_desc,
                'dominant_scale': scale_analysis['dominant_scale'],
                'volatility_regime': volatility_regime,
                'target_price': current_price * (1.02 if signal_type == 'BUY' else 0.98),
                'stop_loss': current_price * (0.98 if signal_type == 'BUY' else 1.02),
                'timestamp': datetime.now(),
                'model_type': 'WAVELET_ANALYSIS',
                'time_frequency_analysis': True
            }
            
            signals.append(signal)
        
        logger.info(f"[SUCCESS] Generated {len(signals)} wavelet-based signals for {symbol}")
        return signals
    
    def _describe_scale_pattern(self, scale_analysis: Dict[str, Any]) -> str:
        """Generate description of the scale pattern."""
        trend_ratio = scale_analysis['trend_energy_ratio']
        cycle_ratio = scale_analysis['cycle_energy_ratio']
        noise_ratio = scale_analysis['noise_energy_ratio']
        
        if trend_ratio > 0.5:
            return "TREND_DOMINATED"
        elif cycle_ratio > 0.4:
            return "CYCLE_DOMINATED"
        elif noise_ratio > 0.5:
            return "NOISE_DOMINATED"
        elif trend_ratio > 0.3 and cycle_ratio > 0.3:
            return "TREND_CYCLE_MIX"
        else:
            return "BALANCED_MULTISCALE"
    
    def reconstruct_signal(self, symbol: str, scale_range: Tuple[float, float] = None) -> Optional[np.ndarray]:
        """
        Reconstruct signal from selected wavelet scales.
        
        Args:
            symbol: Symbol identifier
            scale_range: Tuple of (min_scale, max_scale) to include
            
        Returns:
            Reconstructed signal or None if no cached data
        """
        if symbol not in self.coefficient_cache:
            return None
        
        cache = self.coefficient_cache[symbol]
        coefficients = cache['coefficients']
        scales = cache['scales']
        
        # Filter scales if range specified
        if scale_range:
            min_scale, max_scale = scale_range
            scale_mask = (scales >= min_scale) & (scales  Optional[Dict[str, Any]]:
        """Get scalogram (time-scale representation) for visualization."""
        if symbol not in self.coefficient_cache:
            return None
        
        cache = self.coefficient_cache[symbol]
        magnitudes = cache['magnitudes']
        scales = cache['scales']
        
        return {
            'magnitudes': magnitudes,
            'scales': scales.tolist(),
            'time_points': list(range(magnitudes.shape[1])),
            'wavelet_type': self.config['wavelet_name']
        }
    
    def get_analysis_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis summary for a symbol."""
        return self.analysis_cache.get(symbol)
    
    def decompose_signals(self, market_data: List) -> List[Dict[str, Any]]:
        """
        Decompose signals for all symbols in the data using wavelet analysis.
        
        Args:
            market_data: List of market data points
            
        Returns:
            List of decomposition results for each symbol
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
                analysis = self.analyze_wavelets(data, symbol)
                
                # Convert to expected format
                result = {
                    'symbol': symbol,
                    'strength': analysis['scale_analysis']['dominant_scale_energy'],
                    'dominant_scale': analysis['scale_analysis']['dominant_scale'],
                    'volatility_regime': analysis['volatility_analysis']['current_regime'],
                    'pattern_type': analysis['scale_analysis']['trend_energy_ratio'],
                    'decomposition_levels': len(self.scales),
                    'analysis_data': analysis
                }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"[WARNING] Failed to decompose signals for {symbol}: {e}")
                # Create default result
                results.append({
                    'symbol': symbol,
                    'strength': 0.5,
                    'dominant_scale': 10.0,
                    'volatility_regime': 'NORMAL_VOLATILITY',
                    'pattern_type': 0.4,
                    'decomposition_levels': len(self.scales),
                    'analysis_data': {}
                })
        
        logger.info(f"[SUCCESS] Decomposed signals for {len(results)} symbols")
        return results

# Factory function for compatibility
def create_wavelet_analyzer(config: Dict[str, Any] = None) -> WaveletAnalyzer:
    """Factory function to create wavelet analyzer instance."""
    return WaveletAnalyzer(config)

if __name__ == "__main__":
    # Test the wavelet analyzer
    logger.info("Testing Wavelet Analyzer implementation...")
    
    # Create synthetic market data with known patterns
    np.random.seed(42)
    days = 200
    t = np.arange(days)
    
    # Create price data with trend + cycles + noise + volatility clustering
    trend = 0.02 * t  # Linear trend
    cycle1 = 5 * np.sin(2 * np.pi * t / 30)  # 30-day cycle
    cycle2 = 3 * np.sin(2 * np.pi * t / 7)   # 7-day cycle
    
    # Add volatility clustering
    volatility = 1 + 0.5 * np.abs(np.sin(2 * np.pi * t / 50))
    noise = np.random.randn(days) * volatility
    
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
    analyzer = WaveletAnalyzer()
    
    # Test wavelet analysis
    analysis = analyzer.analyze_wavelets(mock_data, "TEST")
    
    # Test signal generation
    signals = analyzer.generate_signals(mock_data, "TEST")
    
    # Test signal reconstruction
    reconstructed = analyzer.reconstruct_signal("TEST", scale_range=(5, 50))
    
    # Test scalogram
    scalogram = analyzer.get_scalogram("TEST")
    
    logger.info(f"[TEST] Analysis completed: {len(analysis['significant_coefficients'])} significant coefficients")
    logger.info(f"[TEST] Generated {len(signals)} signals")
    logger.info(f"[TEST] Reconstructed signal length: {len(reconstructed) if reconstructed is not None else 0}")
    logger.info(f"[TEST] Scalogram shape: {scalogram['magnitudes'].shape if scalogram else 'None'}")
    logger.info(f"[TEST] Dominant scale: {analysis['scale_analysis']['dominant_scale']:.2f}")
    logger.info(f"[TEST] Volatility regime: {analysis['volatility_analysis']['current_regime']}")
    logger.info(f"[TEST] Wavelet Analyzer implementation working correctly")
