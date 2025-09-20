#!/usr/bin/env python3
"""
Quick test script for Chain-of-Alpha MVP

This script runs a minimal test of the MVP with mock components.
"""

import sys
import os
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from chain_of_alpha_mvp import ChainOfAlphaMVP
from config import CONFIG

def run_quick_test():
    """Run a quick test of the MVP"""

    print("Chain-of-Alpha MVP - Quick Test")
    print("=" * 40)

    # Configure for testing
    test_config = CONFIG.copy()
    test_config.update({
        'llm_model': 'mock',  # Use mock LLM for testing
        'tickers': ['AAPL', 'MSFT'],  # Just 2 stocks for speed
        'start_date': '2023-01-01',
        'end_date': '2024-01-01',
        'num_factors': 3,  # Generate fewer factors
        'optimization_iterations': 1,  # Minimal optimization
        'export_formats': ['json'],  # Only JSON export
    })

    try:
        # Initialize MVP
        print("Initializing MVP...")
        mvp = ChainOfAlphaMVP(test_config)

        # Run pipeline
        print("Running MVP pipeline...")
        results = mvp.run_mvp()

        # Check results
        print("\nTest Results:")
        print(f"✓ Market data acquired: {len(results.get('market_data', {}))} stocks")
        print(f"✓ Factors generated: {len(results.get('candidate_factors', []))}")
        print(f"✓ Factors backtested: {len(results.get('initial_backtest', []))}")
        print(f"✓ Factors optimized: {len(results.get('optimized_factors', []))}")
        print(f"✓ Execution time: {results.get('execution_time_formatted', 'N/A')}")

        # Check exports
        results_dir = test_config.get('output_dir', 'results')
        if os.path.exists(results_dir):
            files = os.listdir(results_dir)
            print(f"✓ Exported files: {len([f for f in files if f.endswith('.json')])} JSON files")

        print("\n✅ Quick test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logging.exception("Test failure details")
        return False

def run_component_tests():
    """Test individual components"""

    print("\nComponent Tests:")
    print("-" * 20)

    try:
        from src.data_acquisition import DataAcquisition
        from src.llm_interface import LLMInterface

        # Test data acquisition
        print("Testing data acquisition...")
        data_config = {'tickers': ['AAPL'], 'start_date': '2024-01-01', 'end_date': '2024-01-15'}
        data_acq = DataAcquisition(data_config)
        data = data_acq.fetch_data()
        print(f"✓ Data acquisition: {len(data)} records")

        # Test LLM interface
        print("Testing LLM interface...")
        llm_config = {'llm_model': 'mock'}
        llm = LLMInterface(llm_config)
        response = llm.generate_response("Test prompt")
        print(f"✓ LLM interface: {len(response)} characters")

        print("✅ Component tests passed!")
        return True

    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise

    success = True

    # Run component tests first
    success &= run_component_tests()

    # Run full MVP test
    success &= run_quick_test()

    if success:
        print("\n🎉 All tests passed! MVP is ready to use.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the logs above.")
        sys.exit(1)