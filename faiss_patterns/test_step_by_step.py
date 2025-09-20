#!/usr/bin/env python3
"""
Step-by-step test to identify hanging issue
"""

import pandas as pd
import numpy as np

def test_import():
    print("Testing import...")
    try:
        from canslim_sepa_pattern_generator import CANSLIMPatternGenerator
        print("✓ CANSLIMPatternGenerator imported successfully")
        return CANSLIMPatternGenerator()
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return None

def test_simple_data():
    print("Testing simple data creation...")
    try:
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        
        price_data = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices],
            'Close': prices,
            'Volume': [1000] * 10
        })
        
        print(f"✓ Price data created: {price_data.shape}")
        return price_data
    except Exception as e:
        print(f"✗ Data creation failed: {e}")
        return None

def main():
    print("Starting step-by-step test...")
    
    # Test 1: Import
    generator = test_import()
    if generator is None:
        return
    
    # Test 2: Simple data
    price_data = test_simple_data()
    if price_data is None:
        return
    
    print("All basic tests passed!")

if __name__ == "__main__":
    main()
