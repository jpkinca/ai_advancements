#!/usr/bin/env python3
"""
Debug script to isolate factor generation issues
"""

import pandas as pd
import numpy as np
from src.llm_interface import LLMInterface
from src.factor_generation import FactorGenerationChain
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    print("=== Factor Generation Debug ===")
    
    # Create test config
    config = {
        'llm_model': 'mock',
        'temperature': 0.7,
        'max_tokens': 1000
    }
    
    # Initialize LLM interface
    llm = LLMInterface(config)
    
    # Initialize factor generation
    factor_gen = FactorGenerationChain(config, llm)
    
    # Create simple test data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'returns': np.random.randn(100) * 0.02,
        'rsi': np.random.uniform(20, 80, 100)
    }, index=dates)
    
    print("Test data shape:", test_data.shape)
    print("Test data columns:", test_data.columns.tolist())
    
    # Test single factor generation
    try:
        factors = factor_gen.generate_factors(test_data, num_factors=1)
        print(f"Generated {len(factors)} factors")
        for factor in factors:
            print(f"Factor: {factor}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()