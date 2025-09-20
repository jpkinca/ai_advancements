"""
AI Trading Advancements - Core Module

This module provides the foundational components for the AI trading system,
including configuration management, data structures, and base classes.

Author: AI Trading Development Team
Date: August 31, 2025
Version: 1.0.0
"""

from .config import AIAdvancementsConfig, get_config
from .data_structures import (
    # Enums
    SignalType, MarketCondition, TimeFrame,
    
    # Data Classes
    MarketData, TechnicalIndicators, SentimentData,
    TradingSignal, PortfolioPosition, BacktestResult,
    ModelMetrics,
    
    # Utilities
    DataValidator
)
from .base_classes import (
    # Abstract Base Classes
    BaseDataProvider, BaseIndicatorCalculator, BaseAIModel,
    BaseSentimentAnalyzer, BaseTradingStrategy, BaseBacktester,
    BaseRiskManager
)
from .timezone_utils import (
    # Timezone utilities for Eastern time (NYSE/NASDAQ timezone)
    now_eastern, to_eastern, EASTERN_TZ, 
    is_market_hours, is_extended_hours, market_session_type,
    next_market_open, next_market_close, format_eastern_time
)

# Package metadata
__version__ = "1.0.0"
__author__ = "AI Trading Development Team"
__description__ = "Core components for AI-driven algorithmic trading system"

# Export all public interfaces
__all__ = [
    # Configuration
    'AIAdvancementsConfig',
    'get_config',
    
    # Enums
    'SignalType',
    'MarketCondition', 
    'TimeFrame',
    
    # Data Structures
    'MarketData',
    'TechnicalIndicators',
    'SentimentData',
    'TradingSignal',
    'PortfolioPosition',
    'BacktestResult',
    'ModelMetrics',
    'DataValidator',
    
    # Base Classes
    'BaseDataProvider',
    'BaseIndicatorCalculator',
    'BaseAIModel',
    'BaseSentimentAnalyzer',
    'BaseTradingStrategy',
    'BaseBacktester',
    'BaseRiskManager',
    
    # Timezone Utilities (Eastern time for NYSE/NASDAQ)
    'now_eastern',
    'to_eastern',
    'EASTERN_TZ',
    'is_market_hours',
    'is_extended_hours',
    'market_session_type',
    'next_market_open',
    'next_market_close',
    'format_eastern_time'
]

# Package initialization
def initialize_core() -> bool:
    """
    Initialize the core AI trading system.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        # Get configuration
        config = get_config()
        
        # Validate configuration
        if not config.validate_configuration():
            return False
        
        # Create necessary directories
        config._create_directories()
        
        # Log successful initialization
        import logging
        logger = logging.getLogger(__name__)
        logger.info("[SUCCESS] AI Trading Core initialized successfully")
        logger.info(f"[DATA] Version: {__version__}")
        logger.info(f"[DATA] Features enabled: {sum(config.features.values())}")
        
        return True
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"[ERROR] Core initialization failed: {e}")
        return False


def get_timezone_utils():
    """Get timezone utility functions."""
    return {
        'now_eastern': now_eastern,
        'to_eastern': to_eastern,
        'EASTERN_TZ': EASTERN_TZ,
        'is_market_hours': is_market_hours,
        'is_extended_hours': is_extended_hours,
        'market_session_type': market_session_type,
        'next_market_open': next_market_open,
        'next_market_close': next_market_close,
        'format_eastern_time': format_eastern_time
    }


# Auto-initialize on import (optional)
if __name__ != "__main__":
    try:
        initialize_core()
    except Exception:
        # Silent failure on import - let the application handle initialization
        pass


if __name__ == "__main__":
    # Module test and demonstration
    print("\n=== AI Trading Core Module Test ===")
    
    # Test initialization
    success = initialize_core()
    print(f"[DATA] Core initialization: {'SUCCESS' if success else 'FAILED'}")
    
    # Test configuration
    config = get_config()
    print(f"[DATA] Configuration loaded: {config is not None}")
    print(f"[DATA] Database configured: {bool(config.database.url)}")
    print(f"[DATA] IBKR configured: {config.ibkr.host}:{config.ibkr.port}")
    
    # Test data structures
    from datetime import datetime, timezone
    from decimal import Decimal
    
    # Create sample market data
    market_data = MarketData(
        symbol="TEST",
        timestamp=datetime.now(timezone.utc),
        open_price=Decimal('100.00'),
        high_price=Decimal('102.00'),
        low_price=Decimal('99.00'),
        close_price=Decimal('101.50'),
        volume=1000000,
        timeframe=TimeFrame.DAY_1
    )
    
    print(f"[SUCCESS] MarketData created: {market_data.symbol}")
    print(f"[DATA] Validation: {DataValidator.validate_market_data(market_data)}")
    
    # Create sample trading signal
    signal = TradingSignal(
        symbol="TEST",
        timestamp=datetime.now(timezone.utc),
        signal_type=SignalType.BUY,
        confidence=Decimal('0.85'),
        entry_price=Decimal('101.50')
    )
    
    print(f"[SUCCESS] TradingSignal created: {signal.signal_type.value}")
    print(f"[DATA] Confidence: {signal.confidence}")
    print(f"[DATA] Validation: {DataValidator.validate_trading_signal(signal)}")
    
    # Display package info
    print(f"\n[DATA] Package: {__description__}")
    print(f"[DATA] Version: {__version__}")
    print(f"[DATA] Author: {__author__}")
    print(f"[DATA] Exported components: {len(__all__)}")
    
    print("\n[SUCCESS] Core module test completed")
