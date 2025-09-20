"""
FAISS Pattern Recognition Module

This module contains pattern generators and FAISS integration for trading pattern recognition.

Components:
- CANSLIMPatternGenerator: William O'Neil CANSLIM methodology
- SEPAPatternGenerator: Mark Minervini SEPA methodology  
- WarriorTradingPatternGenerator: Cameron Ross day trading patterns
- PatternMatchingEngine: Unified pattern matching interface
"""

__version__ = "1.0.0"
__author__ = "GitHub Copilot"

# Import pattern generators for easy access
try:
    from .canslim_sepa_pattern_generator import (
        CANSLIMPatternGenerator,
        SEPAPatternGenerator, 
        PatternMatchingEngine
    )
    from .WT_day_trading_pattern_generator import WarriorTradingPatternGenerator
    
    __all__ = [
        'CANSLIMPatternGenerator',
        'SEPAPatternGenerator',
        'WarriorTradingPatternGenerator',
        'PatternMatchingEngine'
    ]
    
except ImportError as e:
    # Graceful degradation if dependencies are missing
    import warnings
    warnings.warn(f"Some pattern generators could not be imported: {e}")
    __all__ = []
