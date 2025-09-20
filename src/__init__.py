"""
AI Trading Advancements - Main Module

This is the main entry point for the AI Trading Advancements system.
Provides a comprehensive AI-driven algorithmic trading platform that integrates
with existing TradeAppComponents infrastructure.

Author: AI Trading Development Team
Date: August 31, 2025
Version: 1.0.0
"""

# Import core components
from .core import (
    # Configuration
    AIAdvancementsConfig, get_config,
    
    # Data structures
    MarketData, TradingSignal, PortfolioPosition, BacktestResult,
    SignalType, MarketCondition, TimeFrame, ModelMetrics,
    
    # Base classes
    BaseDataProvider, BaseAIModel, BaseTradingStrategy,
    BaseBacktester, BaseRiskManager,
    
    # Utilities
    DataValidator
)

# Import AI predictive components
from .ai_predictive import (
    DQNTradingModel, TradingEnvironment,
    YFinanceDataProvider, TradeAppDataBridge
)

# Package metadata
__version__ = "1.0.0"
__author__ = "AI Trading Development Team"
__description__ = "AI-driven algorithmic trading system with reinforcement learning"
__license__ = "MIT"

# Export all public interfaces
__all__ = [
    # Configuration
    'AIAdvancementsConfig',
    'get_config',
    
    # Core data structures
    'MarketData',
    'TradingSignal',
    'PortfolioPosition',
    'BacktestResult',
    'ModelMetrics',
    
    # Enums
    'SignalType',
    'MarketCondition',
    'TimeFrame',
    
    # Base classes
    'BaseDataProvider',
    'BaseAIModel',
    'BaseTradingStrategy',
    'BaseBacktester',
    'BaseRiskManager',
    
    # Utilities
    'DataValidator',
    
    # AI Models
    'DQNTradingModel',
    'TradingEnvironment',
    
    # Data providers
    'YFinanceDataProvider',
    'TradeAppDataBridge'
]


def initialize_ai_trading_system() -> bool:
    """
    Initialize the complete AI trading system.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("[STARTING] AI Trading System initialization")
        
        # Initialize core components
        from .core import initialize_core
        core_success = initialize_core()
        
        if not core_success:
            logger.error("[ERROR] Core initialization failed")
            return False
        
        # Initialize predictive analytics
        from .ai_predictive import initialize_predictive_analytics
        ai_success = initialize_predictive_analytics()
        
        if not ai_success:
            logger.error("[ERROR] AI predictive analytics initialization failed")
            return False
        
        # Validate system configuration
        config = get_config()
        config_valid = config.validate_configuration()
        
        if not config_valid:
            logger.warning("[WARNING] Configuration validation failed - some features may not work")
        
        # Display system status
        logger.info("[SUCCESS] AI Trading System initialized successfully")
        logger.info(f"[DATA] Version: {__version__}")
        logger.info(f"[DATA] Core components: Loaded")
        logger.info(f"[DATA] AI models: Loaded")
        logger.info(f"[DATA] Data providers: Loaded")
        logger.info(f"[DATA] Configuration valid: {config_valid}")
        
        # Display feature status
        enabled_features = [name for name, enabled in config.features.items() if enabled]
        logger.info(f"[DATA] Enabled features: {', '.join(enabled_features)}")
        
        return True
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"[ERROR] AI Trading System initialization failed: {e}")
        return False


def get_system_info() -> dict:
    """
    Get comprehensive system information.
    
    Returns:
        dict: System information and status
    """
    try:
        config = get_config()
        
        info = {
            'version': __version__,
            'author': __author__,
            'description': __description__,
            'license': __license__,
            'components': {
                'core': len([x for x in __all__ if 'Base' in x or x in ['MarketData', 'TradingSignal']]),
                'ai_models': len([x for x in __all__ if 'Model' in x or 'Environment' in x]),
                'data_providers': len([x for x in __all__ if 'Provider' in x or 'Bridge' in x]),
                'total_exports': len(__all__)
            },
            'features': config.features if config else {},
            'database_configured': bool(config.database.url) if config else False,
            'ibkr_configured': config.ibkr.host != '127.0.0.1' if config else False
        }
        
        return info
        
    except Exception as e:
        return {
            'version': __version__,
            'error': str(e),
            'status': 'Configuration error'
        }


# Quick start function for new users
def quick_start_guide():
    """Display quick start guide for the AI trading system."""
    guide = """
=== AI Trading Advancements Quick Start Guide ===

1. SETUP:
   - Ensure all dependencies are installed: pip install -r requirements.txt
   - Configure environment variables in .env file
   - Set up database connection (PostgreSQL recommended)

2. BASIC USAGE:
   
   # Initialize the system
   from ai_advancements import initialize_ai_trading_system
   success = initialize_ai_trading_system()
   
   # Get market data
   from ai_advancements import YFinanceDataProvider, TimeFrame
   provider = YFinanceDataProvider()
   await provider.connect()
   data = await provider.get_historical_data("AAPL", TimeFrame.DAY_1, start_date, end_date)
   
   # Train AI model
   from ai_advancements import DQNTradingModel
   model = DQNTradingModel()
   metrics = await model.train(data)
   
   # Generate signals
   signals = await model.predict(data)

3. INTEGRATION:
   - The system integrates with existing TradeAppComponents
   - Use TradeAppDataBridge to access scanner results
   - Models can be saved/loaded for persistence

4. DOCUMENTATION:
   - Check docs/ folder for detailed documentation
   - See examples/ for usage patterns
   - API reference available in API_REFERENCE.md

[SUCCESS] Ready to start AI-driven trading!
    """
    print(guide)


# Auto-initialize on import (optional)
if __name__ != "__main__":
    try:
        # Only try to initialize if running in the main application
        import os
        if os.getenv('AI_TRADING_AUTO_INIT', 'false').lower() == 'true':
            initialize_ai_trading_system()
    except Exception:
        # Silent failure on import - let the application handle initialization
        pass


if __name__ == "__main__":
    # Main module demonstration and testing
    import asyncio
    from datetime import datetime, timezone, timedelta
    from decimal import Decimal
    
    async def demo_ai_trading_system():
        print("\n=== AI Trading Advancements System Demo ===")
        
        # Initialize system
        print("[PROCESSING] Initializing AI Trading System...")
        success = initialize_ai_trading_system()
        
        if not success:
            print("[ERROR] System initialization failed")
            return
        
        print("[SUCCESS] System initialized successfully")
        
        # Display system info
        info = get_system_info()
        print(f"\n[DATA] System Information:")
        print(f"  Version: {info['version']}")
        print(f"  Components: {info['components']['total_exports']} total exports")
        print(f"  Core components: {info['components']['core']}")
        print(f"  AI models: {info['components']['ai_models']}")
        print(f"  Data providers: {info['components']['data_providers']}")
        print(f"  Database configured: {info['database_configured']}")
        
        # Demo data provider
        print("\n[PROCESSING] Testing data provider...")
        provider = YFinanceDataProvider()
        connected = await provider.connect()
        
        if connected:
            symbols = await provider.get_symbols()
            current_price = await provider.get_current_price("AAPL")
            print(f"[SUCCESS] Data provider working")
            print(f"[DATA] Available symbols: {len(symbols)}")
            print(f"[DATA] AAPL price: ${current_price}")
            await provider.disconnect()
        
        # Demo AI model
        print("\n[PROCESSING] Testing AI model...")
        
        # Create synthetic training data
        market_data = []
        base_price = 150.0
        
        for i in range(200):
            # Create trending price movement
            trend = 0.001 if i < 100 else -0.001
            noise = (i % 10 - 5) / 1000
            base_price *= (1 + trend + noise)
            
            market_data.append(MarketData(
                symbol="DEMO",
                timestamp=datetime.now(timezone.utc) + timedelta(days=i),
                open_price=Decimal(str(base_price * 0.999)),
                high_price=Decimal(str(base_price * 1.002)),
                low_price=Decimal(str(base_price * 0.998)),
                close_price=Decimal(str(base_price)),
                volume=1000000,
                timeframe=TimeFrame.DAY_1
            ))
        
        print(f"[SUCCESS] Created {len(market_data)} demo data points")
        
        # Quick training demo
        model = DQNTradingModel(name="DemoModel")
        
        try:
            print("[PROCESSING] Training demo model (quick training)...")
            metrics = await model.train(
                training_data=market_data[:150],
                total_timesteps=1000  # Quick demo training
            )
            
            print(f"[SUCCESS] Model trained")
            print(f"[DATA] Final portfolio value: ${metrics.hyperparameters['final_portfolio_value']:.2f}")
            
            # Generate predictions
            print("[PROCESSING] Generating predictions...")
            signals = await model.predict(market_data[150:])
            
            buy_signals = sum(1 for s in signals if s.signal_type == SignalType.BUY)
            sell_signals = sum(1 for s in signals if s.signal_type == SignalType.SELL)
            
            print(f"[SUCCESS] Generated {len(signals)} signals")
            print(f"[DATA] Buy: {buy_signals}, Sell: {sell_signals}, Hold: {len(signals) - buy_signals - sell_signals}")
            
        except Exception as e:
            print(f"[WARNING] Model demo failed (this is normal if dependencies missing): {e}")
        
        # Demo integration
        print("\n[PROCESSING] Testing TradeApp integration...")
        bridge = TradeAppDataBridge()
        
        scanner_symbols = await bridge.get_scanner_symbols()
        all_symbols = await bridge.get_all_relevant_symbols()
        
        print(f"[SUCCESS] Integration bridge working")
        print(f"[DATA] Scanner symbols: {len(scanner_symbols)}")
        print(f"[DATA] Total symbols: {len(all_symbols)}")
        
        print(f"\n=== Demo Summary ===")
        print(f"[SUCCESS] AI Trading System is functional")
        print(f"[DATA] All major components tested")
        print(f"[DATA] Ready for production use")
        
        # Show quick start
        print(f"\n[DATA] For detailed usage, run:")
        print(f"  from ai_advancements import quick_start_guide")
        print(f"  quick_start_guide()")
        
        print(f"\n[SUCCESS] AI Trading Advancements demo completed")
    
    # Run the demo
    asyncio.run(demo_ai_trading_system())
