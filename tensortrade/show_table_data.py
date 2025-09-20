#!/usr/bin/env python3
"""Show first 10 records from each table populated by db_utils.py"""

from src.db_utils import get_engine
from sqlalchemy import text
import pandas as pd

def main():
    db_url = 'postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway'
    
    engine = get_engine(db_url)
    print('✅ Connected to database')
    
    # Show first 10 records from tt_prices (the only table with data)
    with engine.connect() as conn:
        print('\n📊 TT_PRICES TABLE - First 10 Records:')
        print('=' * 80)
        
        df = pd.read_sql(text('SELECT * FROM tt_prices ORDER BY instrument, timestamp LIMIT 10'), conn)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.precision', 2)
        
        print(df.to_string(index=False))
        
        count_result = pd.read_sql(text("SELECT COUNT(*) as count FROM tt_prices"), conn)
        total_count = count_result["count"][0]
        print(f'\nTotal records in tt_prices: {total_count}')
        
        # Show unique symbols
        symbols_df = pd.read_sql(text('SELECT instrument, COUNT(*) as records FROM tt_prices GROUP BY instrument ORDER BY instrument'), conn)
        print(f'\nSymbols in database ({len(symbols_df)} symbols):')
        print(symbols_df.to_string(index=False))

if __name__ == '__main__':
    main()
