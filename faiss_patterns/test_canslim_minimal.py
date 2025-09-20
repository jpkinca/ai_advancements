#!/usr/bin/env python3
"""
Minimal test for CANSLIM pattern generator
"""

import pandas as pd
import numpy as np
from canslim_sepa_pattern_generator import PatternMatchingEngine

def create_minimal_data():
    """Create minimal test data"""
    np.random.seed(42)
    
    # Generate basic price data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    prices = 100 + np.cumsum(np.random.randn(50) * 0.02)
    volumes = np.random.randint(100000, 1000000, 50)
    
    price_data = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.randn(50) * 0.001),
        'High': prices * (1 + np.abs(np.random.randn(50)) * 0.01),
        'Low': prices * (1 - np.abs(np.random.randn(50)) * 0.01),
        'Close': prices,
        'Volume': volumes
    })
    
    volume_data = pd.DataFrame({
        'Date': dates,
        'Volume': volumes
    })
    
    # Minimal earnings data (ensure equal-length arrays)
    earnings_data = pd.DataFrame({
        'eps_growth': [15, 22, 28, 35],
        'annual_eps_growth': [20, 25, 30, 32],
        'roe': [18, 19, 21, 22],
        'eps_surprise': [5, 8, 12, 15]
    })
    
    # Minimal institutional data
    institutional_data = pd.DataFrame({
        'institutional_ownership': [0.45, 0.50, 0.55],
        'new_positions': [3, 5, 8]
    })
    
    # Minimal market data
    market_data = pd.DataFrame({
        'Date': dates,
        'Close': prices * (1 + np.random.randn(50) * 0.001)
    })
    
    return price_data, volume_data, earnings_data, institutional_data, market_data

def main():
    print("Creating minimal test data...")
    price_data, volume_data, earnings_data, institutional_data, market_data = create_minimal_data()
    
    print(f"Price data shape: {price_data.shape}")
    print(f"Volume data shape: {volume_data.shape}")
    print(f"Earnings data shape: {earnings_data.shape}")
    print(f"Institutional data shape: {institutional_data.shape}")
    print(f"Market data shape: {market_data.shape}")
    
    print("\nInitializing pattern engine...")
    engine = PatternMatchingEngine()
    
    print("Testing CANSLIM pattern generation...")
    try:
        canslim_patterns = engine.canslim_generator.generate_canslim_patterns(
            price_data, volume_data, earnings_data, institutional_data, market_data
        )
        print("CANSLIM patterns generated successfully!")
        print(f"Pattern keys: {list(canslim_patterns.keys())}")
        
        # Show some results
        if 'base_patterns' in canslim_patterns:
            print(f"Base patterns found: {list(canslim_patterns['base_patterns'].keys())}")
        
    except Exception as e:
        print(f"Error generating CANSLIM patterns: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
