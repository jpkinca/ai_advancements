#!/usr/bin/env python3
"""
Quick Weekend AI Test - Simplified Version

This script runs a quick test of all AI modules using IBKR Gateway
historical data. Perfect for immediate weekend testing.

Usage: python quick_weekend_test.py
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TradeAppComponents_fresh'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Quick IBKR test
try:
    from ib_insync import IB, Stock, util
    IBKR_OK = True
except ImportError:
    IBKR_OK = False

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def quick_ibkr_test():
    """Quick test of IBKR connection and data fetch"""
    if not IBKR_OK:
        print("[ERROR] ib_insync not installed - pip install ib_insync")
        return False
    
    try:
        ib = IB()
        print("[PROCESSING] Connecting to IBKR Gateway...")
        
        await ib.connectAsync('127.0.0.1', 4002, clientId=9998)
        
        if ib.isConnected():
            print(f"[SUCCESS] Connected to IBKR Gateway")
            print(f"   Server version: {ib.client.serverVersion()}")
            
            # Test data fetch for AAPL
            print("[PROCESSING] Fetching AAPL data...")
            contract = Stock('AAPL', 'SMART', 'USD')
            
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr='1 M',  # 1 month
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if bars:
                df = util.df(bars)
                print(f"[SUCCESS] Fetched {len(df)} bars for AAPL")
                print(f"   Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
                print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                
                ib.disconnect()
                return True
            else:
                print("[ERROR] No data received")
                ib.disconnect()
                return False
        else:
            print("[ERROR] Failed to connect")
            return False
            
    except Exception as e:
        print(f"[ERROR] IBKR test failed: {e}")
        return False

def test_ai_modules():
    """Test if AI modules can be imported"""
    print("\n[TESTING] AI Module Availability:")
    
    modules_status = {}
    
    # Test PPO Trader
    try:
        from src.reinforcement_learning.ppo_trader import PPOTrader
        modules_status['PPOTrader'] = True
        print("   [OK] PPOTrader - Reinforcement Learning")
    except ImportError as e:
        modules_status['PPOTrader'] = False
        print(f"   [ERROR] PPOTrader: {e}")
    
    # Test Portfolio Optimizer
    try:
        from src.genetic_optimization.portfolio_optimizer import PortfolioOptimizer
        modules_status['PortfolioOptimizer'] = True
        print("   [OK] PortfolioOptimizer - Genetic Algorithm")
    except ImportError as e:
        modules_status['PortfolioOptimizer'] = False
        print(f"   [ERROR] PortfolioOptimizer: {e}")
    
    # Test Fourier Analyzer
    try:
        from src.sparse_spectrum.fourier_analyzer import FourierAnalyzer
        modules_status['FourierAnalyzer'] = True
        print("   [OK] FourierAnalyzer - Frequency Domain")
    except ImportError as e:
        modules_status['FourierAnalyzer'] = False
        print(f"   [ERROR] FourierAnalyzer: {e}")
    
    # Test Wavelet Analyzer
    try:
        from src.sparse_spectrum.wavelet_analyzer import WaveletAnalyzer
        modules_status['WaveletAnalyzer'] = True
        print("   [OK] WaveletAnalyzer - Time-Frequency")
    except ImportError as e:
        modules_status['WaveletAnalyzer'] = False
        print(f"   [ERROR] WaveletAnalyzer: {e}")
    
    available_count = sum(modules_status.values())
    print(f"\n[SUMMARY] {available_count}/4 AI modules available")
    
    return modules_status

async def quick_ai_demo():
    """Quick demonstration of available AI modules"""
    print("\n[DEMO] Quick AI Modules Test:")
    
    # Generate sample data
    print("   Generating sample market data...")
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Create realistic price series
    returns = np.random.normal(0.0008, 0.02, 252)  # ~20% annual vol
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(100000, 2000000, 252)
    })
    
    print(f"   Sample data: {len(sample_data)} days, price range ${sample_data['close'].min():.2f}-${sample_data['close'].max():.2f}")
    
    # Test available modules with sample data
    try:
        from src.sparse_spectrum.fourier_analyzer import FourierAnalyzer
        
        analyzer = FourierAnalyzer({'window_size': 252})
        result = analyzer.analyze_frequencies(sample_data['close'].values, 'DEMO')
        
        print(f"   [SUCCESS] Fourier Analysis - Dominant frequency: {result.get('dominant_frequency', 0):.4f}")
        
    except Exception as e:
        print(f"   [ERROR] Fourier demo failed: {e}")
    
    try:
        from src.sparse_spectrum.wavelet_analyzer import WaveletAnalyzer
        
        analyzer = WaveletAnalyzer({'wavelet_type': 'morlet'})
        result = analyzer.analyze(sample_data['close'].values, 'DEMO')
        
        print(f"   [SUCCESS] Wavelet Analysis - Pattern strength: {result.get('pattern_strength', 0):.4f}")
        
    except Exception as e:
        print(f"   [ERROR] Wavelet demo failed: {e}")

def show_next_steps():
    """Show next steps for weekend testing"""
    print("\n" + "=" * 60)
    print("    NEXT STEPS FOR WEEKEND AI TESTING")
    print("=" * 60)
    print()
    print("1. IMMEDIATE TESTING:")
    print("   - Run: python weekend_ai_tester.py")
    print("   - Uses your stock universe (10-50 symbols)")
    print("   - Fetches 1-2 years historical data from IBKR")
    print("   - Runs all 4 AI modules")
    print()
    print("2. CUSTOMIZE STOCK UNIVERSE:")
    print("   Edit weekend_ai_tester.py, line ~400:")
    print("   selected_universe = ['AAPL', 'MSFT', 'GOOGL', ...]")
    print()
    print("3. WEEKEND CAPABILITIES:")
    print("   ✓ Portfolio optimization (perfect for weekends)")
    print("   ✓ Frequency domain analysis")
    print("   ✓ Wavelet pattern analysis")
    print("   ✓ PPO model training on historical data")
    print("   ✓ Full database integration")
    print()
    print("4. NO LIVE TRADING NEEDED:")
    print("   - All analysis uses historical data")
    print("   - IBKR Gateway provides data access")
    print("   - Perfect for strategy development")
    print()
    print("5. EXPECTED RUNTIME:")
    print("   - 10 stocks: ~5-10 minutes")
    print("   - 20 stocks: ~10-20 minutes")
    print("   - 50 stocks: ~20-40 minutes")
    print()
    print("6. OUTPUT:")
    print("   - Comprehensive analysis report")
    print("   - Optimal portfolio allocations")
    print("   - Market cycle detection")
    print("   - Pattern recognition results")
    print("   - AI model performance metrics")

async def main():
    """Main quick test function"""
    print("=" * 60)
    print("    QUICK WEEKEND AI TEST")
    print("=" * 60)
    
    # Test IBKR connection
    ibkr_success = await quick_ibkr_test()
    
    # Test AI modules
    modules_status = test_ai_modules()
    
    # Quick demo
    if any(modules_status.values()):
        await quick_ai_demo()
    
    # Show results
    print("\n" + "=" * 60)
    print("    QUICK TEST RESULTS")
    print("=" * 60)
    print(f"IBKR Gateway Connection: {'✓ WORKING' if ibkr_success else '✗ FAILED'}")
    print(f"AI Modules Available: {sum(modules_status.values())}/4")
    
    if ibkr_success and any(modules_status.values()):
        print("\n🎉 READY FOR WEEKEND AI TESTING!")
        show_next_steps()
    else:
        print("\n❌ SETUP ISSUES DETECTED")
        if not ibkr_success:
            print("   - Check IBKR Gateway is running on port 4002")
            print("   - Ensure paper trading is enabled")
        if not any(modules_status.values()):
            print("   - Check AI module dependencies")
            print("   - Verify src/ directory structure")

if __name__ == "__main__":
    asyncio.run(main())
