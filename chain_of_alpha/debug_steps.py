#!/usr/bin/env python3
"""
Debug script to check each step of data processing
"""

import pandas as pd
import numpy as np
from src.data_acquisition import DataAcquisition
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=== Data Processing Step by Step Debug ===")
    
    # Create test config for just one ticker
    config = {
        'tickers': ['AAPL'],
        'start_date': '2023-01-01',
        'end_date': '2023-02-01',  # Shorter period for debugging
        'data_source': 'yfinance'
    }
    
    # Initialize data acquisition
    data_acq = DataAcquisition(config)
    
    print("1. Fetching single ticker...")
    raw_data = data_acq._fetch_single_ticker('AAPL')
    if raw_data is not None:
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Raw data columns: {raw_data.columns.tolist()}")
        print(f"Raw data index type: {type(raw_data.index)}")
        
        print("\n2. Preprocessing data...")
        processed_data = data_acq._preprocess_data(raw_data, 'AAPL')
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Processed data columns: {processed_data.columns.tolist()}")
        print(f"Processed data column types: {[type(col) for col in processed_data.columns]}")
        
        # Check if any columns are problematic
        for col in processed_data.columns:
            if isinstance(col, tuple):
                print(f"Found tuple column: {col}")
        
        print("\n3. Testing concatenation manually...")
        # Simulate what happens in _combine_market_data
        df_copy = processed_data.copy()
        df_copy['ticker'] = 'AAPL'
        df_copy.set_index('ticker', append=True, inplace=True)
        df_copy = df_copy.reorder_levels(['ticker', df_copy.index.names[0]])
        
        print(f"After adding ticker index: {df_copy.shape}")
        print(f"Index names: {df_copy.index.names}")
        print(f"Columns: {df_copy.columns.tolist()}")
        print(f"Column types: {[type(col) for col in df_copy.columns]}")

if __name__ == "__main__":
    main()