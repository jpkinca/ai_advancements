#!/usr/bin/env python3
"""
AI Data Access Utility

This module provides easy access to stored historical market data for AI modules.
Handles data retrieval, filtering, and formatting for different AI module requirements.

Features:
- Easy data queries by symbol, timeframe, date range
- Automatic Eastern Time handling
- Data format conversion for AI modules
- Efficient caching and batch operations
- Market hours filtering
- Data quality validation
"""

import asyncio
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from decimal import Decimal
import json

# Database imports
try:
    import asyncpg
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Eastern Time Zone
EASTERN_TZ = pytz.timezone('America/New_York')

class AIDataAccessor:
    """
    Data access utility for AI modules to query stored historical market data
    with proper Eastern Time handling and efficient caching.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.connection_pool = None
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        logger.info("[SETUP] AI Data Accessor initialized")
        logger.info(f"   Database: {'Configured' if self.database_url else 'Not configured'}")
    
    async def initialize(self) -> bool:
        """Initialize database connection pool"""
        if not DATABASE_AVAILABLE or not self.database_url:
            logger.error("[ERROR] Database not available or URL not configured")
            return False
        
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=30
            )
            logger.info("[SUCCESS] Database connection pool initialized")
            return True
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize database pool: {e}")
            return False
    
    async def close(self):
        """Close database connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("[INFO] Database connection pool closed")
    
    async def get_available_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data in the database"""
        if not self.connection_pool:
            return {}
        
        try:
            async with self.connection_pool.acquire() as conn:
                # Get symbols and timeframes
                query = """
                SELECT symbol, timeframe, 
                       COUNT(*) as data_points,
                       MIN(date_eastern) as earliest_date,
                       MAX(date_eastern) as latest_date,
                       MIN(close_price) as min_price,
                       MAX(close_price) as max_price
                FROM historical_market_data 
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
                """
                
                rows = await conn.fetch(query)
                
                summary = {
                    'total_records': 0,
                    'symbols': set(),
                    'timeframes': set(),
                    'data_by_symbol': {},
                    'data_by_timeframe': {}
                }
                
                for row in rows:
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    data_points = row['data_points']
                    
                    summary['total_records'] += data_points
                    summary['symbols'].add(symbol)
                    summary['timeframes'].add(timeframe)
                    
                    if symbol not in summary['data_by_symbol']:
                        summary['data_by_symbol'][symbol] = {}
                    
                    summary['data_by_symbol'][symbol][timeframe] = {
                        'data_points': data_points,
                        'date_range': {
                            'start': row['earliest_date'].strftime('%Y-%m-%d %H:%M ET'),
                            'end': row['latest_date'].strftime('%Y-%m-%d %H:%M ET')
                        },
                        'price_range': {
                            'min': float(row['min_price']),
                            'max': float(row['max_price'])
                        }
                    }
                    
                    if timeframe not in summary['data_by_timeframe']:
                        summary['data_by_timeframe'][timeframe] = []
                    
                    summary['data_by_timeframe'][timeframe].append({
                        'symbol': symbol,
                        'data_points': data_points
                    })
                
                # Convert sets to sorted lists
                summary['symbols'] = sorted(list(summary['symbols']))
                summary['timeframes'] = sorted(list(summary['timeframes']))
                
                return summary
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to get data summary: {e}")
            return {}
    
    async def get_symbol_data(self, symbol: str, timeframe: str, 
                            start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                            market_hours_only: bool = True, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get historical data for a specific symbol and timeframe
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timeframe: Data timeframe (e.g., '1day', '1hour', '15min')
            start_date: Start date (Eastern Time)
            end_date: End date (Eastern Time)
            market_hours_only: Only return market hours data
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with historical data
        """
        if not self.connection_pool:
            logger.error("[ERROR] Database not initialized")
            return pd.DataFrame()
        
        # Create cache key
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}_{market_hours_only}_{limit}"
        
        # Check cache
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                logger.info(f"[CACHE HIT] {cache_key}")
                return data
        
        try:
            async with self.connection_pool.acquire() as conn:
                # Build query
                where_conditions = [f"symbol = $1", f"timeframe = $2"]
                params = [symbol, timeframe]
                param_count = 2
                
                if start_date:
                    param_count += 1
                    where_conditions.append(f"date_eastern >= ${param_count}")
                    params.append(start_date)
                
                if end_date:
                    param_count += 1
                    where_conditions.append(f"date_eastern <= ${param_count}")
                    params.append(end_date)
                
                if market_hours_only:
                    where_conditions.append("is_market_hours = true")
                
                query = f"""
                SELECT * FROM historical_market_data
                WHERE {' AND '.join(where_conditions)}
                ORDER BY date_eastern
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                # Execute query
                rows = await conn.fetch(query, *params)
                
                # Convert to DataFrame
                if rows:
                    df = pd.DataFrame([dict(row) for row in rows])
                    df = self._process_dataframe(df)
                else:
                    df = pd.DataFrame()
                
                # Cache result
                self.cache[cache_key] = (datetime.now(), df)
                
                logger.info(f"[SUCCESS] Retrieved {len(df)} records for {symbol} {timeframe}")
                return df
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to get symbol data: {e}")
            return pd.DataFrame()
    
    async def get_multiple_symbols_data(self, symbols: List[str], timeframe: str,
                                     start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                                     market_hours_only: bool = True) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols in a single operation"""
        if not self.connection_pool:
            logger.error("[ERROR] Database not initialized")
            return {}
        
        try:
            async with self.connection_pool.acquire() as conn:
                # Build query for multiple symbols
                symbol_placeholders = ', '.join([f'${i+1}' for i in range(len(symbols))])
                where_conditions = [f"symbol IN ({symbol_placeholders})", f"timeframe = ${len(symbols)+1}"]
                params = symbols + [timeframe]
                param_count = len(symbols) + 1
                
                if start_date:
                    param_count += 1
                    where_conditions.append(f"date_eastern >= ${param_count}")
                    params.append(start_date)
                
                if end_date:
                    param_count += 1
                    where_conditions.append(f"date_eastern <= ${param_count}")
                    params.append(end_date)
                
                if market_hours_only:
                    where_conditions.append("is_market_hours = true")
                
                query = f"""
                SELECT * FROM historical_market_data
                WHERE {' AND '.join(where_conditions)}
                ORDER BY symbol, date_eastern
                """
                
                rows = await conn.fetch(query, *params)
                
                # Group by symbol
                data_by_symbol = {}
                if rows:
                    df = pd.DataFrame([dict(row) for row in rows])
                    df = self._process_dataframe(df)
                    
                    for symbol in symbols:
                        symbol_df = df[df['symbol'] == symbol].copy()
                        if not symbol_df.empty:
                            data_by_symbol[symbol] = symbol_df
                
                logger.info(f"[SUCCESS] Retrieved data for {len(data_by_symbol)} symbols")
                return data_by_symbol
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to get multiple symbols data: {e}")
            return {}
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame to proper format for AI modules"""
        if df.empty:
            return df
        
        # Convert price columns to float
        price_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'typical_price']
        for col in price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert volume to integer
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        
        # Convert technical indicators to float
        tech_columns = ['returns', 'log_returns', 'volume_ratio', 'sma_5', 'sma_10', 'sma_20', 'sma_50']
        for col in tech_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure datetime columns are timezone-aware
        if 'date_eastern' in df.columns:
            df['date_eastern'] = pd.to_datetime(df['date_eastern'])
            if df['date_eastern'].dt.tz is None:
                df['date_eastern'] = df['date_eastern'].dt.tz_localize(EASTERN_TZ)
        
        # Rename columns for AI module compatibility
        column_mapping = {
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'date_eastern': 'date'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Set date as index if available
        if 'date' in df.columns:
            df = df.set_index('date').sort_index()
        
        return df
    
    # AI Module-Specific Data Formatters
    
    async def get_ppo_training_data(self, symbol: str, timeframe: str = '15min',
                                  lookback_days: int = 30) -> List[Dict[str, float]]:
        """Get data formatted for PPO training"""
        end_date = datetime.now(EASTERN_TZ)
        start_date = end_date - timedelta(days=lookback_days)
        
        df = await self.get_symbol_data(symbol, timeframe, start_date, end_date)
        
        if df.empty:
            return []
        
        # Format for PPO training
        training_data = []
        for i, row in df.iterrows():
            data_point = {
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
                'low': float(row.get('low', 0)),
                'close': float(row.get('close', 0)),
                'volume': float(row.get('volume', 0)),
                'returns': float(row.get('returns', 0)) if pd.notna(row.get('returns')) else 0.0,
                'timestamp': i.timestamp() if isinstance(i, datetime) else 0
            }
            
            # Add technical indicators if available
            for indicator in ['sma_5', 'sma_10', 'sma_20', 'volume_ratio']:
                if indicator in row and pd.notna(row[indicator]):
                    data_point[indicator] = float(row[indicator])
            
            training_data.append(data_point)
        
        return training_data
    
    async def get_vpa_training_data(self, symbol: str, timeframe: str = '15min',
                                   lookback_days: int = 30) -> List[Dict[str, float]]:
        """Get data formatted for VPA training"""
        end_date = datetime.now(EASTERN_TZ)
        start_date = end_date - timedelta(days=lookback_days)
        
        df = await self.get_symbol_data(symbol, timeframe, start_date, end_date)
        
        if df.empty:
            return []
        
        # Import VPA features
        try:
            from volume.volume_price_action import VPAFeatures
            vpa_computer = VPAFeatures()
            df = vpa_computer.compute_basic_vpa(df)
            df = vpa_computer.compute_advanced_vpa(df)
            df = vpa_computer.detect_vpa_patterns(df)
        except ImportError:
            logger.warning("[WARNING] VPA module not available, using basic features")
        
        # Format for VPA training
        training_data = []
        for i, row in df.iterrows():
            data_point = {
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
                'low': float(row.get('low', 0)),
                'close': float(row.get('close', 0)),
                'volume': float(row.get('volume', 0)),
                'vol_price_ratio': float(row.get('vol_price_ratio', 0)),
                'volume_imbalance': float(row.get('volume_imbalance', 0)),
                'volume_ratio': float(row.get('volume_ratio', 0)),
                'bullish_volume': int(row.get('bullish_volume', 0)),
                'bearish_volume': int(row.get('bearish_volume', 0)),
                'timestamp': i.timestamp() if isinstance(i, datetime) else 0
            }
            
            training_data.append(data_point)
        
        return training_data
    
    async def get_portfolio_optimization_data(self, symbols: List[str], 
                                            timeframe: str = '1day',
                                            lookback_days: int = 252) -> Dict[str, np.ndarray]:
        """Get data formatted for portfolio optimization (returns matrix)"""
        end_date = datetime.now(EASTERN_TZ)
        start_date = end_date - timedelta(days=lookback_days)
        
        data = await self.get_multiple_symbols_data(symbols, timeframe, start_date, end_date)
        
        if not data:
            return {}
        
        # Extract returns for each symbol
        returns_data = {}
        for symbol, df in data.items():
            if not df.empty and 'returns' in df.columns:
                returns = df['returns'].dropna()
                if len(returns) > 20:  # Minimum data requirement
                    returns_data[symbol] = returns.values
        
        return returns_data
    
    async def get_fourier_analysis_data(self, symbol: str, timeframe: str = '1day',
                                      lookback_days: int = 500) -> np.ndarray:
        """Get price data formatted for Fourier analysis"""
        end_date = datetime.now(EASTERN_TZ)
        start_date = end_date - timedelta(days=lookback_days)
        
        df = await self.get_symbol_data(symbol, timeframe, start_date, end_date)
        
        if df.empty or 'close' not in df.columns:
            return np.array([])
        
        return np.array(df['close'].values)
    
    async def get_wavelet_analysis_data(self, symbol: str, timeframe: str = '1hour',
                                      lookback_days: int = 100) -> np.ndarray:
        """Get price data formatted for wavelet analysis"""
        end_date = datetime.now(EASTERN_TZ)
        start_date = end_date - timedelta(days=lookback_days)
        
        df = await self.get_symbol_data(symbol, timeframe, start_date, end_date)
        
        if df.empty or 'close' not in df.columns:
            return np.array([])
        
        return np.array(df['close'].values)
    
    # Utility Methods
    
    async def get_latest_data_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get the latest data timestamp for a symbol/timeframe"""
        if not self.connection_pool:
            return None
        
        try:
            async with self.connection_pool.acquire() as conn:
                query = """
                SELECT MAX(date_eastern) as latest_date
                FROM historical_market_data
                WHERE symbol = $1 AND timeframe = $2
                """
                
                row = await conn.fetchrow(query, symbol, timeframe)
                return row['latest_date'] if row and row['latest_date'] else None
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to get latest timestamp: {e}")
            return None
    
    async def get_data_quality_stats(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get data quality statistics for a symbol/timeframe"""
        if not self.connection_pool:
            return {}
        
        try:
            async with self.connection_pool.acquire() as conn:
                query = """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN volume = 0 THEN 1 END) as zero_volume_count,
                    AVG(CASE WHEN returns IS NOT NULL THEN ABS(returns) END) as avg_abs_return,
                    STDDEV(returns) as return_volatility,
                    COUNT(CASE WHEN is_market_hours = true THEN 1 END) as market_hours_count
                FROM historical_market_data
                WHERE symbol = $1 AND timeframe = $2
                """
                
                row = await conn.fetchrow(query, symbol, timeframe)
                
                if row:
                    total = row['total_records']
                    return {
                        'total_records': total,
                        'zero_volume_ratio': row['zero_volume_count'] / total if total > 0 else 0,
                        'avg_abs_return': float(row['avg_abs_return']) if row['avg_abs_return'] else 0,
                        'return_volatility': float(row['return_volatility']) if row['return_volatility'] else 0,
                        'market_hours_ratio': row['market_hours_count'] / total if total > 0 else 0
                    }
                
                return {}
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to get data quality stats: {e}")
            return {}
    
    def print_data_summary(self, summary: Optional[Dict[str, Any]] = None):
        """Print formatted data summary"""
        if summary is None:
            # This would require an async call, so just return a message
            print("Call await accessor.get_available_data_summary() first")
            return
        
        print("=" * 80)
        print("    AI DATA ACCESS SUMMARY")
        print("=" * 80)
        print()
        print(f"Total Records: {summary['total_records']:,}")
        print(f"Symbols: {len(summary['symbols'])} ({', '.join(summary['symbols'])})")
        print(f"Timeframes: {len(summary['timeframes'])} ({', '.join(summary['timeframes'])})")
        print()
        
        print("DATA BY SYMBOL:")
        print("-" * 60)
        for symbol, timeframes in summary['data_by_symbol'].items():
            total_points = sum(tf_data['data_points'] for tf_data in timeframes.values())
            print(f"{symbol:8}: {total_points:6,} total data points")
            
            for timeframe, tf_data in timeframes.items():
                points = tf_data['data_points']
                date_range = tf_data['date_range']
                price_range = tf_data['price_range']
                print(f"         {timeframe:8}: {points:6,} points, {date_range['start']} to {date_range['end']}")
                print(f"                    Price range: ${price_range['min']:.2f} - ${price_range['max']:.2f}")
        
        print()
        print("=" * 80)

# Factory function for easy usage
async def create_ai_data_accessor(database_url: Optional[str] = None) -> AIDataAccessor:
    """Factory function to create and initialize an AI data accessor"""
    accessor = AIDataAccessor(database_url)
    
    if await accessor.initialize():
        return accessor
    else:
        await accessor.close()
        raise RuntimeError("Failed to initialize AI data accessor")

# Example usage
async def main():
    """Example usage of AI Data Accessor"""
    try:
        # Create accessor
        accessor = await create_ai_data_accessor(os.getenv('DATABASE_URL'))
        
        # Get data summary
        summary = await accessor.get_available_data_summary()
        accessor.print_data_summary(summary)
        
        # Example: Get data for PPO training
        ppo_data = await accessor.get_ppo_training_data('AAPL', '15min', lookback_days=30)
        print(f"\nPPO training data for AAPL: {len(ppo_data)} samples")
        
        # Example: Get data for portfolio optimization
        portfolio_data = await accessor.get_portfolio_optimization_data(['AAPL', 'MSFT', 'GOOGL'])
        print(f"Portfolio optimization data: {len(portfolio_data)} symbols")
        
        # Close accessor
        await accessor.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
