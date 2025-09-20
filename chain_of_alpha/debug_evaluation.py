#!/usr/bin/env python3
"""
Debug script to check factor evaluation
"""

import pandas as pd
import numpy as np
from src.data_acquisition import DataAcquisition
from src.llm_interface import LLMInterface
from src.factor_generation import FactorGenerationChain
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=== Factor Evaluation Debug ===")
    
    # Create test config for just one ticker
    config = {
        'tickers': ['AAPL'],
        'start_date': '2023-01-01',
        'end_date': '2024-01-01',
        'data_source': 'yfinance',
        'llm_model': 'mock',
        'temperature': 0.7,
        'max_tokens': 1000
    }
    
    # Initialize components
    data_acq = DataAcquisition(config)
    llm = LLMInterface(config)
    factor_gen = FactorGenerationChain(config, llm)
    
    print("1. Fetching data...")
    data = data_acq.fetch_data()
    print(f"Data shape: {data.shape}")
    print(f"Available columns: {data.columns.tolist()}")
    
    print("\n2. Generating factors...")
    factors = factor_gen.generate_factors(data, num_factors=3)
    print(f"Generated {len(factors)} factors")
    
    for i, factor in enumerate(factors):
        print(f"\nFactor {i+1}:")
        print(f"  ID: {factor['id']}")
        print(f"  Expression: {factor['expression']}")
        print(f"  Explanation: {factor['explanation']}")
        
        # Try to compute this factor manually
        try:
            result = eval(factor['expression'], {"__builtins__": {}}, {"df": data, "np": np, "pd": pd})
            print(f"  Computation successful: {type(result)}")
            if hasattr(result, 'shape'):
                print(f"  Result shape: {result.shape}")
            if hasattr(result, 'isnull'):
                print(f"  Non-null values: {(~result.isnull()).sum()}")
                print(f"  Sample values: {result.dropna().head(3).tolist()}")
        except Exception as e:
            print(f"  Computation failed: {e}")
    
    print("\n3. Evaluating factors...")
    evaluated_factors = factor_gen.evaluate_factors(factors, data)
    print(f"Evaluated {len(evaluated_factors)} factors")
    
    for i, factor in enumerate(evaluated_factors):
        print(f"\nEvaluated Factor {i+1}:")
        print(f"  Has values: {'values' in factor}")
        if 'values' in factor and factor['values'] is not None:
            values = factor['values']
            print(f"  Values type: {type(values)}")
            print(f"  Values shape: {values.shape}")
            print(f"  Non-null count: {(~values.isnull()).sum()}")
            print(f"  Stats: {factor.get('stats', {})}")

if __name__ == "__main__":
    main()