"""
AI Trading Advancements - AI Predictive Module

This module provides reinforcement learning and predictive analytics components
for the AI trading system, including DQN models and market data providers.

Author: AI Trading Development Team
Date: August 31, 2025
Version: 1.0.0
"""

from .dqn_trading_model import DQNTradingModel, TradingEnvironment, TradingCallback
from .market_data_provider import YFinanceDataProvider, TradeAppDataBridge

# Module metadata
__version__ = "1.0.0"
__author__ = "AI Trading Development Team"
__description__ = "AI predictive analytics and reinforcement learning for trading"

# Export all public interfaces
__all__ = [
    # RL Models
    'DQNTradingModel',
    'TradingEnvironment',
    'TradingCallback',
    
    # Data Providers
    'YFinanceDataProvider',
    'TradeAppDataBridge'
]


def initialize_predictive_analytics() -> bool:
    """
    Initialize the AI predictive analytics module.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        # Verify required dependencies
        try:
            import torch
            import stable_baselines3
            import gymnasium
            import yfinance
        except ImportError as e:
            logger.error(f"[ERROR] Missing required dependency: {e}")
            return False
        
        # Check for GPU availability
        if torch.cuda.is_available():
            logger.info(f"[SUCCESS] CUDA available: {torch.cuda.get_device_name()}")
        else:
            logger.info("[DATA] Using CPU for training (CUDA not available)")
        
        logger.info("[SUCCESS] AI Predictive Analytics module initialized")
        logger.info(f"[DATA] PyTorch version: {torch.__version__}")
        logger.info(f"[DATA] Stable-Baselines3 version: {stable_baselines3.__version__}")
        
        return True
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"[ERROR] Predictive analytics initialization failed: {e}")
        return False


# Auto-initialize on import (optional)
if __name__ != "__main__":
    try:
        initialize_predictive_analytics()
    except Exception:
        # Silent failure on import - let the application handle initialization
        pass


if __name__ == "__main__":
    # Module test and demonstration
    import asyncio
    from datetime import datetime, timezone, timedelta
    
    try:
        import sys
        sys.path.append('..')
        from ..core import MarketData, TimeFrame, SignalType
        from decimal import Decimal
    except ImportError:
        print("[ERROR] Cannot import core modules for testing")
        sys.exit(1)
    
    async def test_predictive_module():
        print("\n=== AI Predictive Analytics Module Test ===")
        
        # Test initialization
        success = initialize_predictive_analytics()
        print(f"[DATA] Module initialization: {'SUCCESS' if success else 'FAILED'}")
        
        if not success:
            print("[ERROR] Cannot proceed without proper initialization")
            return
        
        # Test data provider
        print("[PROCESSING] Testing data provider...")
        provider = YFinanceDataProvider()
        
        connected = await provider.connect()
        print(f"[DATA] Data provider connected: {connected}")
        
        if connected:
            # Get some sample data
            symbols = await provider.get_symbols()
            print(f"[DATA] Available symbols: {len(symbols)}")
            
            current_price = await provider.get_current_price("AAPL")
            print(f"[DATA] AAPL current price: ${current_price}")
            
            await provider.disconnect()
        
        # Test trading environment
        print("[PROCESSING] Testing trading environment...")
        
        # Create synthetic data for testing
        market_data = []
        base_price = 100.0
        
        for i in range(100):
            base_price *= (1 + (i % 10 - 5) / 1000)  # Simple pattern
            
            market_data.append(MarketData(
                symbol="TEST",
                timestamp=datetime.now(timezone.utc) + timedelta(days=i),
                open_price=Decimal(str(base_price * 0.999)),
                high_price=Decimal(str(base_price * 1.001)),
                low_price=Decimal(str(base_price * 0.998)),
                close_price=Decimal(str(base_price)),
                volume=100000,
                timeframe=TimeFrame.DAY_1
            ))
        
        print(f"[SUCCESS] Created {len(market_data)} test data points")
        
        # Test environment creation
        try:
            env = TradingEnvironment(
                market_data=market_data,
                initial_balance=10000.0,
                lookback_window=10
            )
            
            obs, _ = env.reset()
            print(f"[SUCCESS] Trading environment created")
            print(f"[DATA] Observation shape: {obs.shape}")
            print(f"[DATA] Action space: {env.action_space}")
            
            # Test a few steps
            for i in range(5):
                action = env.action_space.sample()
                obs, reward, done, _, info = env.step(action)
                
                if done:
                    break
            
            print(f"[DATA] Final portfolio value: ${info.get('portfolio_value', 0):.2f}")
            
        except Exception as e:
            print(f"[ERROR] Trading environment test failed: {e}")
        
        # Test DQN model creation
        print("[PROCESSING] Testing DQN model creation...")
        
        try:
            model = DQNTradingModel(
                name="TestModel",
                learning_rate=0.001
            )
            
            print(f"[SUCCESS] DQN model created: {model.name}")
            print(f"[DATA] Model version: {model.version}")
            print(f"[DATA] Is trained: {model.is_trained}")
            
        except Exception as e:
            print(f"[ERROR] DQN model creation failed: {e}")
        
        # Test data bridge
        print("[PROCESSING] Testing data bridge...")
        
        try:
            bridge = TradeAppDataBridge()
            
            scanner_symbols = await bridge.get_scanner_symbols()
            all_symbols = await bridge.get_all_relevant_symbols()
            
            print(f"[SUCCESS] Data bridge functional")
            print(f"[DATA] Scanner symbols: {len(scanner_symbols)}")
            print(f"[DATA] All symbols: {len(all_symbols)}")
            
        except Exception as e:
            print(f"[ERROR] Data bridge test failed: {e}")
        
        print(f"\n[DATA] Module: {__description__}")
        print(f"[DATA] Version: {__version__}")
        print(f"[DATA] Components: {len(__all__)}")
        
        print("\n[SUCCESS] AI Predictive Analytics module test completed")
    
    # Run the test
    asyncio.run(test_predictive_module())
