#!/usr/bin/env python3
"""
Quick test to debug the FAISS pattern generator issue
"""

import pandas as pd
import numpy as np

# Try to import the module and see what happens
try:
    print("Starting debug test...")
    
    # Create simple test data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    volumes = np.random.randint(1000000, 5000000, 100)
    
    price_data = pd.DataFrame({
        'Date': dates,
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': volumes
    })
    
    volume_data = pd.DataFrame({
        'Date': dates,
        'Volume': volumes
    })
    
    market_data = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    
    print("Test data created successfully")
    print(f"Price data shape: {price_data.shape}")
    print(f"Volume data shape: {volume_data.shape}")
    print(f"Market data shape: {market_data.shape}")
    
    # Try importing the pattern generators
    from canslim_sepa_pattern_generator import PatternMatchingEngine
    print("Pattern matching engine imported successfully")
    
    engine = PatternMatchingEngine()
    print("Engine initialized")
    
    # Test SEPA patterns with our simple data
    print("Testing SEPA pattern generation...")
    sepa_patterns = engine.sepa_generator.generate_sepa_patterns(
        price_data, volume_data, market_data
    )
    print("SEPA patterns generated successfully!")
    print(f"SEPA result keys: {list(sepa_patterns.keys())}")
    
    print("\nDebug test completed successfully!")
    
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
