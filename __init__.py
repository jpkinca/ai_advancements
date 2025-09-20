"""
AI Trading Advancements

A comprehensive AI-driven algorithmic trading system that integrates with 
TradeAppComponents infrastructure. Features reinforcement learning, 
sentiment analysis, and advanced predictive analytics.

Author: AI Trading Development Team
Date: August 31, 2025
Version: 1.0.0
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "AI Trading Development Team"
__description__ = "AI-driven algorithmic trading system with reinforcement learning"
__license__ = "MIT"
__url__ = "https://github.com/jpkinca/TradeAppComponents"

def get_version():
    """Get the current version of AI Trading Advancements."""
    return __version__

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import stable_baselines3
    except ImportError:
        missing_deps.append("stable-baselines3")
    
    try:
        import gymnasium
    except ImportError:
        missing_deps.append("gymnasium")
    
    try:
        import yfinance
    except ImportError:
        missing_deps.append("yfinance")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print(f"[WARNING] Missing dependencies: {', '.join(missing_deps)}")
        print("[DATA] Install with: pip install -r requirements.txt")
        return False
    else:
        print("[SUCCESS] All dependencies are available")
        return True

def system_status():
    """Get a quick system status overview."""
    print("\n=== AI Trading Advancements Status ===")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check configuration
    try:
        from .src.core import get_config
        config = get_config()
        config_ok = config.validate_configuration()
    except Exception as e:
        config_ok = False
        print(f"[ERROR] Configuration error: {e}")
    
    # Display status
    print(f"[DATA] Version: {__version__}")
    print(f"[DATA] Dependencies: {'OK' if deps_ok else 'MISSING'}")
    print(f"[DATA] Configuration: {'OK' if config_ok else 'INVALID'}")
    
    if deps_ok and config_ok:
        print("[SUCCESS] System ready for use")
        return True
    else:
        print("[WARNING] System requires attention")
        return False

# Lazy imports to avoid circular dependencies
def get_core_components():
    """Get core components (lazy import)."""
    from .src.core import (
        AIAdvancementsConfig, get_config,
        MarketData, TradingSignal, PortfolioPosition,
        SignalType, TimeFrame, DataValidator
    )
    return {
        'AIAdvancementsConfig': AIAdvancementsConfig,
        'get_config': get_config,
        'MarketData': MarketData,
        'TradingSignal': TradingSignal,
        'PortfolioPosition': PortfolioPosition,
        'SignalType': SignalType,
        'TimeFrame': TimeFrame,
        'DataValidator': DataValidator
    }

def get_ai_components():
    """Get AI components (lazy import)."""
    from .src.ai_predictive import (
        DQNTradingModel, YFinanceDataProvider
    )
    return {
        'DQNTradingModel': DQNTradingModel,
        'YFinanceDataProvider': YFinanceDataProvider
    }

# Convenience exports for direct import
def YFinanceDataProvider():
    """Direct access to YFinanceDataProvider."""
    ai_components = get_ai_components()
    return ai_components['YFinanceDataProvider']

def DQNTradingModel():
    """Direct access to DQNTradingModel.""" 
    ai_components = get_ai_components()
    return ai_components['DQNTradingModel']

def initialize_ai_trading_system():
    """Initialize the complete AI trading system."""
    try:
        from .src import initialize_ai_trading_system as init_func
        return init_func()
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return False

# Export main functions
__all__ = [
    'get_version',
    'check_dependencies', 
    'system_status',
    'get_core_components',
    'get_ai_components',
    'initialize_ai_trading_system',
    
    # Direct exports for convenience
    'YFinanceDataProvider',
    'DQNTradingModel'
]

if __name__ == "__main__":
    # Package demonstration when run directly
    print(f"\n{__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    
    # Run system status check
    status_ok = system_status()
    
    if status_ok:
        print("\n[DATA] To get started:")
        print("  import ai_advancements")
        print("  core = ai_advancements.get_core_components()")
        print("  ai = ai_advancements.get_ai_components()")
    else:
        print("\n[DATA] Please resolve the above issues before using the system")
        print("[DATA] Check the documentation for setup instructions")
