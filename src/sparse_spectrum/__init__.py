"""
Sparse Spectrum Methods for Trading

This package implements sparse spectrum analysis techniques:
- Fourier analysis for frequency domain trading
- Wavelet analysis for time-frequency decomposition
- Compressed sensing for sparse feature extraction
"""

from .fourier_analyzer import (
    FourierAnalyzer,
    FrequencyComponent,
    create_fourier_analyzer
)

from .wavelet_analyzer import (
    WaveletAnalyzer,
    WaveletCoefficient,
    WaveletSignal,
    create_wavelet_analyzer
)

# from .compressed_sensing import (
#     CompressedSensingAnalyzer,
#     MarketFeatureExtractor,
#     AnomalyDetector,
#     CompressedSensingTradingModel,
#     create_compressed_sensing_model
# )

# Temporary comment out due to import issues
# Will be fixed in next phase

__all__ = [
    'FourierAnalyzer',
    'FrequencyComponent',
    'create_fourier_analyzer',
    'WaveletAnalyzer',
    'WaveletCoefficient',
    'WaveletSignal', 
    'create_wavelet_analyzer'
    # 'CompressedSensingAnalyzer',
    # 'MarketFeatureExtractor', 
    # 'AnomalyDetector',
    # 'CompressedSensingTradingModel',
    # 'create_compressed_sensing_model'
]
