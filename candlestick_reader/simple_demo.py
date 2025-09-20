#!/usr/bin/env python3
"""
Simplified Candlestick AI System Demo

This script demonstrates the core candlestick analysis functionality
without requiring complex ML dependencies.
"""

import os
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_basic_analysis():
    """Demonstrate basic candlestick analysis"""
    print("\n" + "="*60)
    print("BASIC CANDLESTICK ANALYSIS DEMO")
    print("="*60)

    try:
        from candlestick_reader.candlestick_analyzer import CandlestickAnalyzer
    except ImportError as e:
        print(f"Core analyzer not available: {e}")
        print("This demo requires the basic candlestick analyzer")
        return

    # Initialize analyzer
    analyzer = CandlestickAnalyzer()

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')

    # Create realistic price movements
    base_price = 150.0
    prices = [base_price]
    for i in range(99):
        change = np.random.normal(0, 0.005)  # 0.5% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.003)))
        low = price * (1 - abs(np.random.normal(0, 0.003)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.randint(1000, 10000)

        data.append({
            'date': dates[i],
            'open': open_price,
            'high': max(high, open_price, price),
            'low': min(low, open_price, price),
            'close': price,
            'volume': volume
        })

    df = pd.DataFrame(data).set_index('date')

    # Perform analysis
    chart_path = analyzer.generate_candlestick_chart(df, 'DEMO', '5min')
    result = analyzer.analyze_candlestick_signal(df, 'DEMO', chart_path)

    print(f"Symbol: DEMO")
    print(f"Signal: {result['signal'].upper()}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Dominant Pattern: {result['patterns'].get('dominant', 'None')}")
    print(f"Patterns Detected: {len(result['patterns'].get('all', []))}")

    if result['patterns'].get('all'):
        print("\nTop Patterns:")
        for pattern in result['patterns']['all'][:5]:
            print(f"  - {pattern['name']}: {pattern['confidence']:.3f}")

    print(f"\nAnalysis Summary: {result['analysis']}")

    # Clean up
    if os.path.exists(chart_path):
        os.remove(chart_path)

    return result

async def demo_integrated_system():
    """Demonstrate the integrated candlestick system"""
    print("\n" + "="*60)
    print("INTEGRATED SYSTEM DEMO")
    print("="*60)

    try:
        from candlestick_reader.integration import IntegratedCandlestickSystem
    except ImportError as e:
        print(f"Integrated system not available: {e}")
        print("This demo requires the integration module")
        return

    # Initialize integrated system
    system = IntegratedCandlestickSystem()

    # Mock data accessor for demo
    class MockDataAccessor:
        async def get_symbol_data(self, symbol, timeframe, start_date, end_date):
            # Generate sample data
            dates = pd.date_range(start_date, end_date, freq='5min')
            data = []

            base_price = 100.0
            for i, date in enumerate(dates[:50]):  # Limit to 50 bars
                price = base_price + np.sin(i * 0.1) * 5 + np.random.normal(0, 1)
                data.append({
                    'date': date,
                    'open': price + np.random.normal(0, 0.5),
                    'high': price + abs(np.random.normal(0, 1)),
                    'low': price - abs(np.random.normal(0, 1)),
                    'close': price,
                    'volume': np.random.randint(1000, 5000)
                })

            return pd.DataFrame(data).set_index('date')

    # Replace data accessor
    system.data_accessor = MockDataAccessor()

    # Perform comprehensive analysis
    result = await system.analyze_symbol_comprehensive('AAPL', '5min', 50)

    print(f"Symbol: {result['symbol']}")
    print(f"Combined Signal: {result['signal'].upper()}")
    print(f"Overall Confidence: {result['confidence']:.3f}")

    # Component breakdown
    components = result.get('components', {})
    print("\nComponent Analysis:")
    print(f"  Candlestick: {components.get('candlestick', {}).get('signal', 'N/A')} "
          f"({components.get('candlestick', {}).get('confidence', 0):.3f})")
    print(f"  VPA: {components.get('vpa', {}).get('signal', 'N/A')} "
          f"({components.get('vpa', {}).get('confidence', 0):.3f})")

    # Trade recommendation
    trade = result.get('trade_recommendation', {})
    print("\nTrade Recommendation:")
    print(f"  Action: {trade.get('action', 'HOLD')}")
    print(f"  Position Size: {trade.get('position_size', 0)}")
    print(f"  Reasoning: {', '.join(trade.get('reasoning', []))}")

    return result

def demo_system_validation():
    """Demonstrate system validation"""
    print("\n" + "="*60)
    print("SYSTEM VALIDATION")
    print("="*60)

    try:
        from candlestick_reader.test_candlestick_system import run_system_validation

        # Run system validation
        validation = run_system_validation()

        print(f"Overall System Status: {validation['overall_status'].upper()}")

        print("\nComponent Status:")
        for component, status in validation['tests'].items():
            status_icon = "✓" if status['status'] == 'passed' else "✗"
            print(f"  {status_icon} {component}: {status['message']}")

        if validation['overall_status'] == 'passed':
            print("\n🎉 System is ready for production use!")
        else:
            print("\n⚠️  System validation failed. Check dependencies and configuration.")

    except ImportError as e:
        print(f"Validation not available: {e}")
        print("System validation requires the test module")

async def main():
    """Run complete system demo"""
    print("🚀 Candlestick AI Analysis System Demo")
    print("="*60)
    print("This demo showcases the core candlestick analysis functionality")
    print("without requiring complex ML dependencies.")
    print("="*60)

    try:
        # Basic analysis demo
        await demo_basic_analysis()

        # Integrated system demo
        await demo_integrated_system()

        # System validation
        demo_system_validation()

        print("\n" + "="*60)
        print("🎯 DEMO COMPLETED!")
        print("="*60)
        print("The Candlestick AI System core functionality is operational!")
        print("\nKey Features Demonstrated:")
        print("• Pattern recognition and analysis")
        print("• VPA integration for enhanced signals")
        print("• Trade recommendation generation")
        print("• System validation and testing")
        print("\nFor full ML capabilities, install additional dependencies:")
        print("  pip install ultralytics transformers torch torchvision")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    # Create results directory for demo outputs
    os.makedirs('results', exist_ok=True)

    # Run the complete demo
    exit_code = asyncio.run(main())