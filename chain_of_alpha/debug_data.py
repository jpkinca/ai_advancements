#!/usr/bin/env python3
"""
Debug script to isolate data acquisition issues
"""

import pandas as pd
import numpy as np
from src.data_acquisition import DataAcquisition
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    print("=== Data Acquisition Debug ===")
    
    # Create test config
    config = {
        'tickers': ['AAPL'],
        'start_date': '2023-01-01',
        'end_date': '2024-01-01',
        'data_source': 'yfinance'
    }
    
    # Initialize data acquisition
    data_acq = DataAcquisition(config)
    
    # Test data acquisition
    try:
        data = data_acq.fetch_data()
        print(f"Data shape: {data.shape}")
        print(f"Data index type: {type(data.index)}")
        print(f"Data index names: {data.index.names}")
        print(f"Data columns: {data.columns.tolist()}")
        print("\nFirst few rows:")
        print(data.head())
        
        # Check for any problematic columns
        for col in data.columns:
            col_data = data[col]
            print(f"\nColumn {col}:")
            print(f"  Type: {type(col_data)}")
            print(f"  Shape: {col_data.shape}")
            print(f"  Dtype: {col_data.dtype}")
            
            # Check for any DataFrame columns (which shouldn't exist)
            if isinstance(col_data.iloc[0], pd.DataFrame):
                print(f"  ERROR: Column {col} contains DataFrame values!")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()