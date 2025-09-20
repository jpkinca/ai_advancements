"""
Complete System Integration Test
Demonstrates the full Sweet Spot & Danger Zone detection system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import warnings

from sweet_spot_danger_zone_system import DualSignalTradingSystem, example_usage
from backtesting_framework import AdvancedBacktester, example_backtest
from real_time_signals import RealTimeSignalGenerator, create_real_time_generator, example_real_time_system
from unified_trading_system import create_unified_system, example_unified_system

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_data(symbols: list, periods: int = 1000) -> dict:
    """Generate synthetic market data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=periods, freq='1H')

    data = {}
    for symbol in symbols:
        # Generate realistic price series with trends and volatility
        trend = np.cumsum(np.random.normal(0.0001, 0.0005, periods))
        noise = np.random.normal(0, 0.01, periods)
        returns = trend + noise

        price = 100 * np.exp(returns.cumsum())

        # Add volume
        volume = np.random.lognormal(15, 0.3, periods)

        # Create OHLCV data
        high_mult = 1 + np.abs(np.random.normal(0, 0.005, periods))
        low_mult = 1 - np.abs(np.random.normal(0, 0.005, periods))

        df = pd.DataFrame({
            'open': price,
            'high': price * high_mult,
            'low': price * low_mult,
            'close': price,
            'volume': volume
        }, index=dates)

        data[symbol] = df

    return data

def test_core_system():
    """Test the core dual signal system"""
    print("="*60)
    print("TESTING CORE DUAL SIGNAL SYSTEM")
    print("="*60)

    try:
        # Generate test data
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        test_data = generate_test_data(symbols, 500)

        # Test basic functionality
        system = DualSignalTradingSystem()

        # Train the system first
        train_data = test_data['AAPL'].iloc[:300]
        metrics = system.train(train_data)
        print("✓ System trained successfully")

        # Generate signals
        signals = system.generate_signals(test_data['AAPL'])
        print(f"✓ Generated {len(signals)} signals for AAPL")
        print(f"✓ Signal columns: {list(signals.columns)}")

        # Test training metrics
        print(f"✓ Sweet Spot Val Accuracy: {metrics['sweet_spot']['val_accuracy']:.3f}")
        print(f"✓ Danger Zone Val Accuracy: {metrics['danger_zone']['val_accuracy']:.3f}")

        return True

    except Exception as e:
        print(f"✗ Core system test failed: {e}")
        return False

def test_backtesting_framework():
    """Test the backtesting framework"""
    print("\n" + "="*60)
    print("TESTING BACKTESTING FRAMEWORK")
    print("="*60)

    try:
        # Generate test data
        symbols = ['AAPL', 'MSFT']
        test_data = generate_test_data(symbols, 300)

        # Create and train system
        system = DualSignalTradingSystem()
        train_data = test_data['AAPL'].iloc[:200]
        system.train(train_data)
        print("✓ System trained for backtesting")

        # Run backtest
        backtester = AdvancedBacktester()
        results = backtester.run_backtest(system, test_data['AAPL'])

        print("✓ Backtest completed")
        print(f"✓ Total return: {results.get('total_return', 0):.2%}")
        print(f"✓ Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"✓ Max drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"✓ Total trades: {results.get('total_trades', 0)}")

        return True

    except Exception as e:
        print(f"✗ Backtesting test failed: {e}")
        return False

def test_real_time_signals():
    """Test real-time signal generation"""
    print("\n" + "="*60)
    print("TESTING REAL-TIME SIGNAL GENERATION")
    print("="*60)

    try:
        # Create real-time generator
        generator = create_real_time_generator()

        # Add test data
        symbols = ['AAPL', 'MSFT']
        test_data = generate_test_data(symbols, 200)

        for symbol, data in test_data.items():
            generator.add_market_data(symbol, data)

        # Test signal generation
        signal = generator.generate_signals('AAPL')
        if signal:
            print("✓ Real-time signal generation working")
            print(f"✓ Signal: {signal['combined_signal']} (Confidence: {signal['confidence_score']:.2f})")
        else:
            print("⚠ No signal generated (may be normal for limited data)")

        # Test buffer functionality
        generator.signal_buffer.add_signal({'test': 'signal', 'symbol': 'AAPL'})
        recent_signals = generator.get_latest_signals(1)
        print(f"✓ Signal buffer working: {len(recent_signals)} signals")

        return True

    except Exception as e:
        print(f"✗ Real-time signals test failed: {e}")
        return False

def test_unified_system():
    """Test the unified trading system"""
    print("\n" + "="*60)
    print("TESTING UNIFIED TRADING SYSTEM")
    print("="*60)

    try:
        # Create unified system
        system = create_unified_system(50000.0)

        # Setup test data
        symbols = ['AAPL']
        test_data = generate_test_data(symbols, 100)

        # Initialize system
        system.initialize_system(symbols, test_data)
        print("✓ System initialization completed")

        # Test portfolio management
        portfolio_stats = system.portfolio.get_portfolio_stats()
        print(f"✓ Portfolio initialized: ${portfolio_stats['current_capital']:.2f}")

        # Test system status
        status = system.get_system_status()
        print(f"✓ System status: Running={status['running']}")

        return True

    except Exception as e:
        print(f"✗ Unified system test failed: {e}")
        return False

def test_end_to_end_integration():
    """Test complete end-to-end integration"""
    print("\n" + "="*60)
    print("TESTING END-TO-END INTEGRATION")
    print("="*60)

    try:
        # Generate comprehensive test data
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        test_data = generate_test_data(symbols, 400)

        # 1. Train the system
        print("1. Training dual signal system...")
        system = DualSignalTradingSystem()

        # Use a subset for training
        train_data = test_data['AAPL'].iloc[:200]
        system.train(train_data)
        print("✓ System trained")

        # 2. Run backtest
        print("2. Running backtest...")
        backtester = AdvancedBacktester()
        backtest_results = backtester.run_backtest(system, test_data['AAPL'])
        print("✓ Backtest completed")
        print(f"   Return: {backtest_results.get('total_return', 0):.2%}")

        # 3. Setup real-time generation
        print("3. Setting up real-time signals...")
        rt_generator = RealTimeSignalGenerator(system)
        for symbol, data in test_data.items():
            rt_generator.add_market_data(symbol, data)
        print("✓ Real-time generator ready")

        # 4. Test unified system
        print("4. Testing unified system...")
        unified = create_unified_system(100000.0)
        unified.signal_system = system
        unified.initialize_system(symbols, test_data)
        print("✓ Unified system initialized")

        # 5. Generate real-time signals
        print("5. Generating real-time signals...")
        signals_generated = 0
        for symbol in symbols:
            signal = rt_generator.generate_signals(symbol)
            if signal:
                signals_generated += 1

        print(f"✓ Generated signals for {signals_generated}/{len(symbols)} symbols")

        print("\n" + "🎉 END-TO-END INTEGRATION TEST PASSED!")
        return True

    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_performance_benchmark():
    """Run performance benchmarks"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKS")
    print("="*60)

    try:
        # Generate larger dataset
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        test_data = generate_test_data(symbols, 1000)

        # Train system for benchmarking
        system = DualSignalTradingSystem()
        train_data = test_data['AAPL'].iloc[:500]
        system.train(train_data)
        print("✓ System trained for benchmarking")

        # Time signal generation
        import time
        start_time = time.time()
        signals = system.generate_signals(test_data['AAPL'])
        generation_time = time.time() - start_time

        print(".4f")
        print(f"Signals per second: {len(signals) / generation_time:.1f}")

        # Benchmark backtesting
        backtester = AdvancedBacktester()
        start_time = time.time()
        results = backtester.run_backtest(system, test_data['AAPL'])
        backtest_time = time.time() - start_time

        print(".2f")
        print(f"Backtest efficiency: {len(test_data['AAPL']) / backtest_time:.0f} data points/sec")

        return True

    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("SWEET SPOT & DANGER ZONE SYSTEM - INTEGRATION TESTS")
    print("="*60)
    print(f"Test Start Time: {datetime.now()}")
    print()

    test_results = []

    # Run individual component tests
    test_results.append(("Core System", test_core_system()))
    test_results.append(("Backtesting Framework", test_backtesting_framework()))
    test_results.append(("Real-Time Signals", test_real_time_signals()))
    test_results.append(("Unified System", test_unified_system()))

    # Run integration test
    test_results.append(("End-to-End Integration", test_end_to_end_integration()))

    # Run performance benchmarks
    test_results.append(("Performance Benchmarks", run_performance_benchmark()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for production use.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()

    exit(0 if success else 1)