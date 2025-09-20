#!/usr/bin/env python3
"""
Multi-Timeframe Historical Data Manager

This module provides centralized historical data fetching and storage for all AI modules.
Fetches data once for multiple timeframes and stores in PostgreSQL with proper Eastern Time.

Key Features:
- Single fetch operation for all timeframes
- Proper NYSE/NASDAQ Eastern Time handling
- Efficient storage in PostgreSQL
- Data sharing across all AI modules
- Multiple duration and bar size combinations
- Automatic data validation and cleanup

Timeframes Supported:
- 1 min, 5 min, 15 min, 1 hour, 1 day, 1 week, 1 month
- Durations: 1D, 1W, 1M, 3M, 6M, 1Y, 2Y, 5Y

All timestamps are converted to and stored as Eastern Time (NYSE/NASDAQ timezone).
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
import json
import hashlib
from urllib.parse import urlparse

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'TradeAppComponents_fresh'))

# Railway PostgreSQL support
from modules.database.railway_db_manager import RailwayPostgreSQLManager

# IBKR imports
try:
    from ib_insync import IB, Stock, util
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False

# Database imports
try:
    import psycopg2
    import asyncpg
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Eastern Time Zone (NYSE/NASDAQ)
EASTERN_TZ = pytz.timezone('America/New_York')
UTC_TZ = pytz.timezone('UTC')

def get_railway_database_url() -> str:
    """Get Railway PostgreSQL connection URL for asyncpg"""
    try:
        # Create Railway manager instance to get connection URL
        railway_manager = RailwayPostgreSQLManager()
        database_url = railway_manager.database_url
        
        # Parse URL to ensure it has SSL requirements for Railway
        parsed = urlparse(database_url)
        if parsed.hostname and 'rlwy.net' in parsed.hostname:
            # Add SSL requirement for Railway
            if '?' not in database_url:
                database_url += '?sslmode=require&gssencmode=disable'
            else:
                if 'sslmode' not in database_url:
                    database_url += '&sslmode=require'
                if 'gssencmode' not in database_url:
                    database_url += '&gssencmode=disable'
        
        logger.info(f"[SUCCESS] Railway PostgreSQL URL configured: {parsed.hostname}:{parsed.port}")
        return database_url
    except Exception as e:
        logger.error(f"[ERROR] Failed to get Railway database URL: {e}")
        # No hardcoded fallback; require environment configuration
        env_url = os.getenv('DATABASE_URL', '')
        if env_url:
            if 'sslmode' not in env_url:
                env_url += ('&' if '?' in env_url else '?') + 'sslmode=require'
            if 'gssencmode' not in env_url:
                env_url += ('&' if '?' in env_url else '?') + 'gssencmode=disable'
            logger.info("[SUCCESS] Using DATABASE_URL from environment")
            return env_url
        raise

class MultiTimeframeDataManager:
    """
    Centralized data manager for fetching and storing multi-timeframe historical data
    with proper Eastern Time handling for NYSE/NASDAQ requirements.
    """
    
    # Standard timeframe configurations
    TIMEFRAME_CONFIGS = {
        # Intraday timeframes
        '1min': {'bar_size': '1 min', 'max_duration': '1 M', 'typical_points': 390*21},    # 1 month of 1-min bars
        '5min': {'bar_size': '5 mins', 'max_duration': '3 M', 'typical_points': 78*63},    # 3 months of 5-min bars
        '15min': {'bar_size': '15 mins', 'max_duration': '6 M', 'typical_points': 26*126}, # 6 months of 15-min bars
        '1hour': {'bar_size': '1 hour', 'max_duration': '1 Y', 'typical_points': 6.5*252}, # 1 year of hourly bars
        
        # Daily and longer timeframes
        '1day': {'bar_size': '1 day', 'max_duration': '5 Y', 'typical_points': 252*5},     # 5 years of daily bars
        '1week': {'bar_size': '1 week', 'max_duration': '10 Y', 'typical_points': 52*10},  # 10 years of weekly bars
        '1month': {'bar_size': '1 month', 'max_duration': '20 Y', 'typical_points': 12*20} # 20 years of monthly bars
    }
    
    # Duration options for each timeframe
    DURATION_OPTIONS = {
        '1min': ['1 D', '1 W', '2 W', '1 M'],
        '5min': ['1 W', '2 W', '1 M', '2 M', '3 M'],
        '15min': ['1 M', '2 M', '3 M', '6 M'],
        '1hour': ['3 M', '6 M', '1 Y'],
        '1day': ['6 M', '1 Y', '2 Y', '3 Y', '5 Y'],
        '1week': ['2 Y', '5 Y', '10 Y'],
        '1month': ['5 Y', '10 Y', '20 Y']
    }
    
    def __init__(self, database_url: str = None, ibkr_host: str = "127.0.0.1", ibkr_port: int = 4002):
        self.database_url = database_url or os.getenv('DATABASE_URL') or get_railway_database_url()
        self.ibkr_host = ibkr_host
        self.ibkr_port = ibkr_port
        self.ib = None
        self.connected = False
        
        # Data cache
        self.data_cache = {}
        self.fetch_metadata = {}
        
        logger.info("[SETUP] Multi-Timeframe Data Manager initialized")
        logger.info(f"   Database URL: {'Configured' if self.database_url else 'Not set'}")
        logger.info(f"   IBKR Gateway: {self.ibkr_host}:{self.ibkr_port}")
        logger.info(f"   Eastern Time Zone: {EASTERN_TZ}")
    
    async def connect_ibkr(self) -> bool:
        """Connect to IBKR Gateway with retry logic"""
        if not IBKR_AVAILABLE:
            logger.error("[ERROR] IBKR not available - install ib_insync")
            return False
        
        try:
            self.ib = IB()
            logger.info(f"[PROCESSING] Connecting to IBKR Gateway at {self.ibkr_host}:{self.ibkr_port}")
            
            # Connect with timeout
            await asyncio.wait_for(
                self.ib.connectAsync(self.ibkr_host, self.ibkr_port, clientId=9997),
                timeout=30.0
            )
            
            if self.ib.isConnected():
                self.connected = True
                server_time = self.ib.reqCurrentTime()
                logger.info(f"[SUCCESS] Connected to IBKR Gateway")
                logger.info(f"   Server version: {self.ib.client.serverVersion()}")
                logger.info(f"   Server time (UTC): {server_time}")
                
                # Convert server time to Eastern Time
                eastern_time = server_time.replace(tzinfo=UTC_TZ).astimezone(EASTERN_TZ)
                logger.info(f"   Eastern Time: {eastern_time}")
                
                return True
            else:
                logger.error("[ERROR] Failed to establish IBKR connection")
                return False
                
        except asyncio.TimeoutError:
            logger.error("[ERROR] IBKR connection timeout")
            return False
        except Exception as e:
            logger.error(f"[ERROR] IBKR connection failed: {e}")
            return False
    
    def disconnect_ibkr(self):
        """Disconnect from IBKR Gateway"""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("[SUCCESS] Disconnected from IBKR Gateway")
    
    def _convert_to_eastern_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame timestamps to Eastern Time"""
        df = df.copy()
        
        # Ensure datetime column is timezone-aware
        if 'date' in df.columns:
            date_col = df['date']
            
            # If timezone-naive, assume UTC
            if date_col.dt.tz is None:
                date_col = date_col.dt.tz_localize(UTC_TZ)
            
            # Convert to Eastern Time
            df['date'] = date_col.dt.tz_convert(EASTERN_TZ)
            
            # Add additional time columns for analysis
            df['eastern_date'] = df['date'].dt.date
            df['eastern_time'] = df['date'].dt.time
            df['eastern_hour'] = df['date'].dt.hour
            df['eastern_dow'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
            df['is_market_hours'] = self._is_market_hours(df['date'])
        
        return df
    
    def _is_market_hours(self, timestamps: pd.Series) -> pd.Series:
        """Determine if timestamps are during regular market hours (9:30-16:00 ET)"""
        # Market hours: 9:30 AM to 4:00 PM Eastern Time, Monday-Friday
        is_weekday = timestamps.dt.dayofweek = pd.Timestamp('09:30:00').time()) & \
                        (timestamps.dt.time  Dict[str, pd.DataFrame]:
        """
        Fetch historical data for a symbol across multiple timeframes
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timeframes: List of timeframes (e.g., ['1min', '15min', '1day'])
            durations: Optional custom durations per timeframe
            
        Returns:
            Dictionary with timeframe -> DataFrame mapping
        """
        if not self.connected:
            logger.error("[ERROR] Not connected to IBKR")
            return {}
        
        symbol_data = {}
        
        # Use default durations if not provided
        if durations is None:
            durations = {}
            for tf in timeframes:
                if tf in self.DURATION_OPTIONS:
                    # Use the second option as default (usually good balance)
                    default_idx = min(1, len(self.DURATION_OPTIONS[tf]) - 1)
                    durations[tf] = self.DURATION_OPTIONS[tf][default_idx]
                else:
                    durations[tf] = '1 Y'  # Fallback
        
        logger.info(f"[PROCESSING] Fetching {symbol} data for {len(timeframes)} timeframes")
        
        for timeframe in timeframes:
            try:
                if timeframe not in self.TIMEFRAME_CONFIGS:
                    logger.warning(f"[WARNING] Unknown timeframe {timeframe}, skipping")
                    continue
                
                config = self.TIMEFRAME_CONFIGS[timeframe]
                duration = durations.get(timeframe, config['max_duration'])
                
                logger.info(f"[DATA] {symbol} {timeframe}: Fetching {duration} of {config['bar_size']} bars")
                
                # Create contract
                contract = Stock(symbol, 'SMART', 'USD')
                
                # Fetch historical data
                bars = await self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime='',
                    durationStr=duration,
                    barSizeSetting=config['bar_size'],
                    whatToShow='TRADES',
                    useRTH=True,  # Regular trading hours only
                    formatDate=1
                )
                
                if bars:
                    # Convert to DataFrame
                    df = util.df(bars)
                    
                    # Add symbol and timeframe metadata
                    df['symbol'] = symbol
                    df['timeframe'] = timeframe
                    df['bar_size'] = config['bar_size']
                    df['duration_requested'] = duration
                    
                    # Convert to Eastern Time
                    df = self._convert_to_eastern_time(df)
                    
                    # Add technical columns
                    df = self._add_technical_columns(df)
                    
                    # Store in results
                    symbol_data[timeframe] = df
                    
                    # Log results
                    date_range = f"{df['date'].iloc[0].strftime('%Y-%m-%d %H:%M')} to {df['date'].iloc[-1].strftime('%Y-%m-%d %H:%M')}"
                    price_range = f"${df['close'].min():.2f} - ${df['close'].max():.2f}"
                    
                    logger.info(f"[SUCCESS] {symbol} {timeframe}: {len(df)} bars, {date_range}, {price_range}")
                    
                else:
                    logger.warning(f"[WARNING] No data received for {symbol} {timeframe}")
                
                # Rate limiting for IBKR pacing rules
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to fetch {symbol} {timeframe}: {e}")
                continue
        
        # Cache the data
        self.data_cache[symbol] = symbol_data
        self.fetch_metadata[symbol] = {
            'fetch_time': datetime.now(EASTERN_TZ),
            'timeframes': list(symbol_data.keys()),
            'total_bars': sum(len(df) for df in symbol_data.values())
        }
        
        logger.info(f"[SUCCESS] {symbol}: Fetched {len(symbol_data)} timeframes, {sum(len(df) for df in symbol_data.values())} total bars")
        
        return symbol_data
    
    def _add_technical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical analysis columns"""
        df = df.copy()
        
        # Basic price calculations
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volume analysis
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Moving averages (only if enough data)
        if len(df) >= 20:
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
        
        if len(df) >= 50:
            df['sma_50'] = df['close'].rolling(50).mean()
        
        return df
    
    async def fetch_multiple_symbols(self, symbols: List[str], timeframes: List[str], 
                                   custom_durations: Dict[str, Dict[str, str]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data for multiple symbols and timeframes efficiently
        
        Args:
            symbols: List of symbols to fetch
            timeframes: List of timeframes for each symbol
            custom_durations: Optional symbol -> timeframe -> duration mapping
            
        Returns:
            Nested dictionary: symbol -> timeframe -> DataFrame
        """
        if not self.connected:
            logger.error("[ERROR] Not connected to IBKR")
            return {}
        
        all_data = {}
        total_requests = len(symbols) * len(timeframes)
        
        logger.info(f"[PROCESSING] Bulk fetch: {len(symbols)} symbols × {len(timeframes)} timeframes = {total_requests} requests")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                # Get custom durations for this symbol if provided
                symbol_durations = None
                if custom_durations and symbol in custom_durations:
                    symbol_durations = custom_durations[symbol]
                
                # Fetch data for this symbol
                symbol_data = await self.fetch_symbol_data(symbol, timeframes, symbol_durations)
                
                if symbol_data:
                    all_data[symbol] = symbol_data
                    
                logger.info(f"[PROGRESS] Completed {i}/{len(symbols)} symbols ({symbol})")
                
                # Rate limiting between symbols
                if i  bool:
        """Store all fetched data to PostgreSQL database"""
        if not DATABASE_AVAILABLE or not self.database_url:
            logger.warning("[WARNING] Database not available or URL not configured")
            return False
        
        try:
            # Connect to database
            conn = await asyncpg.connect(self.database_url)
            
            # Create table if not exists
            await self._create_historical_data_table(conn)
            
            total_rows = 0
            
            for symbol, timeframe_data in all_data.items():
                for timeframe, df in timeframe_data.items():
                    # Prepare data for insertion
                    records = self._prepare_database_records(df, symbol, timeframe)
                    
                    if records:
                        # Upsert records (insert or update on conflict)
                        await self._upsert_historical_data(conn, records)
                        total_rows += len(records)
                        
                        logger.info(f"[DATABASE] Stored {len(records)} records for {symbol} {timeframe}")
            
            await conn.close()
            
            logger.info(f"[SUCCESS] Stored {total_rows} total records to database")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Database storage failed: {e}")
            return False
    
    async def _create_historical_data_table(self, conn):
        """Create the historical data table with proper Eastern Time handling"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS historical_market_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(20) NOT NULL,
            bar_size VARCHAR(20) NOT NULL,
            
            -- Eastern Time timestamps (NYSE/NASDAQ timezone)
            date_eastern TIMESTAMPTZ NOT NULL,
            eastern_date DATE NOT NULL,
            eastern_time TIME NOT NULL,
            eastern_hour INTEGER NOT NULL,
            eastern_dow INTEGER NOT NULL,
            is_market_hours BOOLEAN NOT NULL,
            
            -- OHLCV data
            open_price DECIMAL(12, 4) NOT NULL,
            high_price DECIMAL(12, 4) NOT NULL,
            low_price DECIMAL(12, 4) NOT NULL,
            close_price DECIMAL(12, 4) NOT NULL,
            volume BIGINT NOT NULL,
            
            -- Technical analysis columns
            typical_price DECIMAL(12, 4),
            price_range DECIMAL(12, 4),
            body_size DECIMAL(12, 4),
            upper_shadow DECIMAL(12, 4),
            lower_shadow DECIMAL(12, 4),
            returns DECIMAL(10, 6),
            log_returns DECIMAL(10, 6),
            
            -- Volume analysis
            volume_ma DECIMAL(15, 2),
            volume_ratio DECIMAL(8, 4),
            
            -- Moving averages
            sma_5 DECIMAL(12, 4),
            sma_10 DECIMAL(12, 4),
            sma_20 DECIMAL(12, 4),
            sma_50 DECIMAL(12, 4),
            
            -- Metadata
            duration_requested VARCHAR(20),
            fetch_timestamp TIMESTAMPTZ DEFAULT NOW(),
            
            -- Unique constraint to prevent duplicates
            UNIQUE(symbol, timeframe, date_eastern)
        );
        
        -- Indexes for efficient querying
        CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_timeframe 
            ON historical_market_data(symbol, timeframe);
        CREATE INDEX IF NOT EXISTS idx_historical_data_date_eastern 
            ON historical_market_data(date_eastern);
        CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_date 
            ON historical_market_data(symbol, date_eastern);
        CREATE INDEX IF NOT EXISTS idx_historical_data_market_hours 
            ON historical_market_data(symbol, is_market_hours, date_eastern);
        """
        
        await conn.execute(create_sql)
        logger.info("[DATABASE] Historical data table created/verified")
    
    def _prepare_database_records(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict]:
        """Prepare DataFrame records for database insertion"""
        records = []
        
        for _, row in df.iterrows():
            record = {
                'symbol': symbol,
                'timeframe': timeframe,
                'bar_size': row.get('bar_size', ''),
                'date_eastern': row['date'],
                'eastern_date': row['eastern_date'],
                'eastern_time': row['eastern_time'],
                'eastern_hour': row['eastern_hour'],
                'eastern_dow': row['eastern_dow'],
                'is_market_hours': row['is_market_hours'],
                'open_price': Decimal(str(row['open'])),
                'high_price': Decimal(str(row['high'])),
                'low_price': Decimal(str(row['low'])),
                'close_price': Decimal(str(row['close'])),
                'volume': int(row['volume']),
                'typical_price': Decimal(str(row.get('typical_price', 0))) if pd.notna(row.get('typical_price')) else None,
                'price_range': Decimal(str(row.get('price_range', 0))) if pd.notna(row.get('price_range')) else None,
                'body_size': Decimal(str(row.get('body_size', 0))) if pd.notna(row.get('body_size')) else None,
                'upper_shadow': Decimal(str(row.get('upper_shadow', 0))) if pd.notna(row.get('upper_shadow')) else None,
                'lower_shadow': Decimal(str(row.get('lower_shadow', 0))) if pd.notna(row.get('lower_shadow')) else None,
                'returns': Decimal(str(row.get('returns', 0))) if pd.notna(row.get('returns')) else None,
                'log_returns': Decimal(str(row.get('log_returns', 0))) if pd.notna(row.get('log_returns')) else None,
                'volume_ma': Decimal(str(row.get('volume_ma', 0))) if pd.notna(row.get('volume_ma')) else None,
                'volume_ratio': Decimal(str(row.get('volume_ratio', 0))) if pd.notna(row.get('volume_ratio')) else None,
                'sma_5': Decimal(str(row.get('sma_5', 0))) if pd.notna(row.get('sma_5')) else None,
                'sma_10': Decimal(str(row.get('sma_10', 0))) if pd.notna(row.get('sma_10')) else None,
                'sma_20': Decimal(str(row.get('sma_20', 0))) if pd.notna(row.get('sma_20')) else None,
                'sma_50': Decimal(str(row.get('sma_50', 0))) if pd.notna(row.get('sma_50')) else None,
                'duration_requested': row.get('duration_requested', '')
            }
            records.append(record)
        
        return records
    
    async def _upsert_historical_data(self, conn, records: List[Dict]):
        """Upsert historical data records (insert or update on conflict)"""
        if not records:
            return
        
        # Build the INSERT ... ON CONFLICT UPDATE query
        insert_sql = """
        INSERT INTO historical_market_data (
            symbol, timeframe, bar_size, date_eastern, eastern_date, eastern_time,
            eastern_hour, eastern_dow, is_market_hours, open_price, high_price,
            low_price, close_price, volume, typical_price, price_range, body_size,
            upper_shadow, lower_shadow, returns, log_returns, volume_ma, volume_ratio,
            sma_5, sma_10, sma_20, sma_50, duration_requested
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17,
            $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28
        )
        ON CONFLICT (symbol, timeframe, date_eastern)
        DO UPDATE SET
            bar_size = EXCLUDED.bar_size,
            eastern_date = EXCLUDED.eastern_date,
            eastern_time = EXCLUDED.eastern_time,
            eastern_hour = EXCLUDED.eastern_hour,
            eastern_dow = EXCLUDED.eastern_dow,
            is_market_hours = EXCLUDED.is_market_hours,
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            volume = EXCLUDED.volume,
            typical_price = EXCLUDED.typical_price,
            price_range = EXCLUDED.price_range,
            body_size = EXCLUDED.body_size,
            upper_shadow = EXCLUDED.upper_shadow,
            lower_shadow = EXCLUDED.lower_shadow,
            returns = EXCLUDED.returns,
            log_returns = EXCLUDED.log_returns,
            volume_ma = EXCLUDED.volume_ma,
            volume_ratio = EXCLUDED.volume_ratio,
            sma_5 = EXCLUDED.sma_5,
            sma_10 = EXCLUDED.sma_10,
            sma_20 = EXCLUDED.sma_20,
            sma_50 = EXCLUDED.sma_50,
            duration_requested = EXCLUDED.duration_requested,
            fetch_timestamp = NOW()
        """
        
        # Prepare data tuples
        data_tuples = []
        for record in records:
            tuple_data = (
                record['symbol'], record['timeframe'], record['bar_size'],
                record['date_eastern'], record['eastern_date'], record['eastern_time'],
                record['eastern_hour'], record['eastern_dow'], record['is_market_hours'],
                record['open_price'], record['high_price'], record['low_price'],
                record['close_price'], record['volume'], record['typical_price'],
                record['price_range'], record['body_size'], record['upper_shadow'],
                record['lower_shadow'], record['returns'], record['log_returns'],
                record['volume_ma'], record['volume_ratio'], record['sma_5'],
                record['sma_10'], record['sma_20'], record['sma_50'],
                record['duration_requested']
            )
            data_tuples.append(tuple_data)
        
        # Execute batch insert
        await conn.executemany(insert_sql, data_tuples)
    
    def get_cached_data(self, symbol: str, timeframe: str = None) -> Dict[str, pd.DataFrame]:
        """Get cached data for a symbol"""
        if symbol not in self.data_cache:
            return {}
        
        if timeframe:
            return {timeframe: self.data_cache[symbol].get(timeframe)} if timeframe in self.data_cache[symbol] else {}
        
        return self.data_cache[symbol]
    
    def get_fetch_summary(self) -> Dict[str, Any]:
        """Get summary of all fetched data"""
        summary = {
            'total_symbols': len(self.data_cache),
            'symbols': list(self.data_cache.keys()),
            'timeframes_available': set(),
            'total_bars': 0,
            'fetch_times': {}
        }
        
        for symbol, timeframe_data in self.data_cache.items():
            summary['timeframes_available'].update(timeframe_data.keys())
            summary['total_bars'] += sum(len(df) for df in timeframe_data.values())
            
            if symbol in self.fetch_metadata:
                summary['fetch_times'][symbol] = self.fetch_metadata[symbol]['fetch_time']
        
        summary['timeframes_available'] = sorted(list(summary['timeframes_available']))
        
        return summary
    
    def print_data_summary(self):
        """Print a formatted summary of fetched data"""
        print("=" * 80)
        print("    MULTI-TIMEFRAME DATA SUMMARY")
        print("=" * 80)
        print()
        
        summary = self.get_fetch_summary()
        
        print(f"Total Symbols: {summary['total_symbols']}")
        print(f"Total Bars: {summary['total_bars']:,}")
        print(f"Available Timeframes: {', '.join(summary['timeframes_available'])}")
        print()
        
        print("SYMBOL BREAKDOWN:")
        print("-" * 60)
        
        for symbol in summary['symbols']:
            symbol_data = self.data_cache[symbol]
            total_bars = sum(len(df) for df in symbol_data.values())
            timeframes = ', '.join(symbol_data.keys())
            
            print(f"{symbol:8}: {total_bars:6,} bars across {len(symbol_data)} timeframes ({timeframes})")
            
            # Show date ranges for each timeframe
            for tf, df in symbol_data.items():
                if len(df) > 0:
                    start_date = df['date'].iloc[0].strftime('%Y-%m-%d %H:%M')
                    end_date = df['date'].iloc[-1].strftime('%Y-%m-%d %H:%M')
                    print(f"         {tf:8}: {len(df):4} bars, {start_date} to {end_date}")
        
        print()
        print("=" * 80)

# Factory functions for easy usage
async def fetch_weekend_data(symbols: List[str], timeframes: List[str] = None, 
                           database_url: str = None) -> MultiTimeframeDataManager:
    """
    Factory function to fetch weekend data efficiently
    
    Args:
        symbols: List of stock symbols
        timeframes: List of timeframes (default: ['15min', '1hour', '1day'])
        database_url: PostgreSQL database URL
        
    Returns:
        Configured data manager with fetched data
    """
    if timeframes is None:
        timeframes = ['15min', '1hour', '1day']  # Good balance for AI modules
    
    # Create data manager
    manager = MultiTimeframeDataManager(database_url=database_url)
    
    # Connect to IBKR
    if not await manager.connect_ibkr():
        logger.error("[ERROR] Failed to connect to IBKR")
        return manager
    
    try:
        # Fetch all data
        all_data = await manager.fetch_multiple_symbols(symbols, timeframes)
        
        # Store to database if configured
        if database_url and all_data:
            await manager.store_to_database(all_data)
        
        # Print summary
        manager.print_data_summary()
        
        return manager
        
    finally:
        manager.disconnect_ibkr()

if __name__ == "__main__":
    # Example usage
    async def main():
        # Test symbols
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        test_timeframes = ['15min', '1hour', '1day']
        
        # Fetch data
        manager = await fetch_weekend_data(
            symbols=test_symbols,
            timeframes=test_timeframes,
            database_url=os.getenv('DATABASE_URL')
        )
        
        print("\n[SUCCESS] Data fetch complete - ready for AI module analysis!")
    
    asyncio.run(main())
