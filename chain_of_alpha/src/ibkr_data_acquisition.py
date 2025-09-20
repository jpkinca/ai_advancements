"""
IBKR Data Acquisition Module for Chain-of-Alpha MVP

Production-grade data acquisition using IBKR Gateway with PostgreSQL persistence.
No fallbacks, no mock data - real-time trading data only.

Follows CO-PILOT INSTRUCTIONS:
- IBKR Gateway for all market data
- PostgreSQL for data persistence  
- TA-LIB for technical analysis
- US Eastern Timezone for all timestamps
- No fallbacks allowed
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pytz

import pandas as pd
import numpy as np
import talib
from ib_insync import IB, Stock, util
import asyncpg

logger = logging.getLogger(__name__)

# US Eastern Timezone - NYSE/NASDAQ timezone
US_EASTERN = pytz.timezone('US/Eastern')

class IBKRDataAcquisition:
    """
    Production IBKR data acquisition for Chain-of-Alpha.
    
    - Real-time data via IBKR Gateway
    - PostgreSQL persistence
    - TA-LIB technical analysis
    - US Eastern timezone compliance
    - No fallbacks or mock data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ib = IB()
        self.database_url = config.get('database_url') or 'postgresql://postgres:TAqEkujnMknVURCcrYTIDOzQXbgBNtSX@turntable.proxy.rlwy.net:10410/railway'
        
        # IBKR Gateway settings
        self.gateway_host = config.get('ibkr_host', '127.0.0.1')
        self.gateway_port = config.get('ibkr_port', 4002)
        self.client_id = config.get('ibkr_client_id', 300)
        
        # Data settings
        self.tickers = config.get('tickers', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        self.start_date = config.get('start_date', '2020-01-01')
        self.end_date = config.get('end_date', datetime.now(US_EASTERN).strftime('%Y-%m-%d'))
        
        logger.info("[INIT] IBKR Data Acquisition initialized")
    
    def connect_to_gateway(self) -> bool:
        """
        Connect to IBKR Gateway.
        No fallbacks - must succeed for production use.
        """
        try:
            logger.info(f"[IBKR] Connecting to Gateway at {self.gateway_host}:{self.gateway_port}")
            
            self.ib.connect(
                host=self.gateway_host,
                port=self.gateway_port,
                clientId=self.client_id,
                timeout=30
            )
            
            if not self.ib.isConnected():
                raise ConnectionError("Failed to connect to IBKR Gateway")
            
            logger.info(f"[SUCCESS] Connected to IBKR Gateway - Server Version: {self.ib.client.serverVersion()}")
            
            # Wait for connection to stabilize
            time.sleep(2)
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] IBKR Gateway connection failed: {e}")
            raise ConnectionError(f"IBKR Gateway connection required for production: {e}")
    
    def disconnect_from_gateway(self):
        """Disconnect from IBKR Gateway."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("[SUCCESS] Disconnected from IBKR Gateway")
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch market data from IBKR Gateway and persist to PostgreSQL.
        
        Returns:
            Combined DataFrame with all ticker data and TA-LIB indicators
        """
        logger.info("[START] Fetching market data from IBKR Gateway")
        
        # Connect to IBKR Gateway
        self.connect_to_gateway()
        
        try:
            # Fetch data for all tickers
            all_data = {}
            total_points = 0
            
            for ticker in self.tickers:
                logger.info(f"[IBKR] Fetching data for {ticker}...")
                ticker_data = self._fetch_single_ticker(ticker)
                
                if ticker_data is not None and not ticker_data.empty:
                    # Add TA-LIB technical indicators
                    ticker_data = self._add_talib_indicators(ticker_data, ticker)
                    
                    # Persist to PostgreSQL
                    self._persist_ticker_data(ticker, ticker_data)
                    
                    all_data[ticker] = ticker_data
                    total_points += len(ticker_data)
                    logger.info(f"[SUCCESS] {ticker}: {len(ticker_data)} data points")
                else:
                    logger.error(f"[ERROR] No data retrieved for {ticker}")
                    raise ValueError(f"Failed to fetch data for {ticker} - production requires all data")
            
            # Combine all ticker data
            combined_data = self._combine_ticker_data(all_data)
            
            logger.info(f"[SUCCESS] Retrieved {total_points} total data points from IBKR")
            return combined_data
            
        except Exception as e:
            logger.error(f"[ERROR] Data acquisition failed: {e}")
            raise
        finally:
            self.disconnect_from_gateway()
    
    def _fetch_single_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single ticker from IBKR.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            DataFrame with OHLCV data in US Eastern timezone
        """
        try:
            # Create contract
            contract = Stock(ticker, 'SMART', 'USD')
            
            # Qualify contract
            self.ib.qualifyContracts(contract)
            
            # Calculate duration for historical data request
            start_dt = datetime.strptime(self.start_date, '%Y-%m-%d').replace(tzinfo=US_EASTERN)
            end_dt = datetime.strptime(self.end_date, '%Y-%m-%d').replace(tzinfo=US_EASTERN)
            
            # IBKR historical data request
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime=end_dt,
                durationStr=self._calculate_duration(start_dt, end_dt),
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,  # Regular trading hours only
                formatDate=1,
                keepUpToDate=False
            )
            
            if not bars:
                logger.warning(f"[WARNING] No historical data returned for {ticker}")
                return None
            
            # Convert to DataFrame
            df = util.df(bars)
            
            if df.empty:
                logger.warning(f"[WARNING] Empty DataFrame for {ticker}")
                return None
            
            # Ensure proper timezone handling
            if df.index.tz is None:
                df.index = df.index.tz_localize(US_EASTERN)
            else:
                df.index = df.index.tz_convert(US_EASTERN)
            
            # Rename columns for consistency
            df = df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Add basic derived features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            return df
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to fetch {ticker}: {e}")
            raise
    
    def _calculate_duration(self, start_dt: datetime, end_dt: datetime) -> str:
        """Calculate IBKR duration string from date range."""
        delta = end_dt - start_dt
        
        if delta.days <= 30:
            return f"{delta.days} D"
        elif delta.days <= 365:
            weeks = delta.days // 7
            return f"{weeks} W"
        else:
            years = delta.days // 365
            return f"{years} Y"
    
    def _add_talib_indicators(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add TA-LIB technical indicators as per CO-PILOT INSTRUCTIONS.
        
        Args:
            df: OHLCV DataFrame
            ticker: Stock symbol for logging
            
        Returns:
            DataFrame with TA-LIB indicators added
        """
        try:
            logger.info(f"[TA-LIB] Adding technical indicators for {ticker}")
            
            # Ensure we have sufficient data for indicators
            if len(df) < 50:
                logger.warning(f"[WARNING] Insufficient data for {ticker}: {len(df)} bars")
                return df
            
            # Price data arrays for TA-LIB
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # Moving Averages
            df['sma_5'] = talib.SMA(close, timeperiod=5)
            df['sma_10'] = talib.SMA(close, timeperiod=10)
            df['sma_20'] = talib.SMA(close, timeperiod=20)
            df['sma_50'] = talib.SMA(close, timeperiod=50)
            df['ema_12'] = talib.EMA(close, timeperiod=12)
            df['ema_26'] = talib.EMA(close, timeperiod=26)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # RSI
            df['rsi'] = talib.RSI(close, timeperiod=14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # Williams %R
            df['williams_r'] = talib.WILLR(high, low, close)
            
            # Average True Range
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma'] = talib.SMA(volume, timeperiod=20)
                df['obv'] = talib.OBV(close, volume)
                df['ad'] = talib.AD(high, low, close, volume)
            
            # Momentum indicators
            df['momentum'] = talib.MOM(close, timeperiod=10)
            df['roc'] = talib.ROC(close, timeperiod=10)
            
            # Volatility
            df['volatility_20'] = df['returns'].rolling(20).std()
            
            # Volume ratios
            df['volume_ma_5'] = df['volume'].rolling(5).mean()
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            
            # Price momentum
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            logger.info(f"[SUCCESS] Added {len(df.columns)} indicators for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to add TA-LIB indicators for {ticker}: {e}")
            raise
    
    async def _persist_ticker_data(self, ticker: str, data: pd.DataFrame):
        """
        Persist ticker data to PostgreSQL database.
        
        Args:
            ticker: Stock symbol
            data: DataFrame with OHLCV and indicators
        """
        try:
            logger.info(f"[DB] Persisting {len(data)} records for {ticker}")
            
            conn = await asyncpg.connect(self.database_url)
            
            try:
                # Create table if not exists
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS chain_of_alpha_market_data (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(20) NOT NULL,
                        date TIMESTAMPTZ NOT NULL,
                        open DECIMAL(15,4) NOT NULL,
                        high DECIMAL(15,4) NOT NULL,
                        low DECIMAL(15,4) NOT NULL,
                        close DECIMAL(15,4) NOT NULL,
                        volume BIGINT NOT NULL,
                        
                        -- Basic derived features
                        returns DECIMAL(10,6),
                        log_returns DECIMAL(10,6),
                        
                        -- TA-LIB indicators
                        sma_5 DECIMAL(15,4),
                        sma_10 DECIMAL(15,4),
                        sma_20 DECIMAL(15,4),
                        sma_50 DECIMAL(15,4),
                        ema_12 DECIMAL(15,4),
                        ema_26 DECIMAL(15,4),
                        
                        macd DECIMAL(10,6),
                        macd_signal DECIMAL(10,6),
                        macd_hist DECIMAL(10,6),
                        
                        rsi DECIMAL(8,4),
                        
                        bb_upper DECIMAL(15,4),
                        bb_middle DECIMAL(15,4),
                        bb_lower DECIMAL(15,4),
                        
                        stoch_k DECIMAL(8,4),
                        stoch_d DECIMAL(8,4),
                        williams_r DECIMAL(8,4),
                        atr DECIMAL(10,6),
                        
                        volume_sma DECIMAL(20,2),
                        obv DECIMAL(20,2),
                        ad DECIMAL(20,2),
                        
                        momentum DECIMAL(10,6),
                        roc DECIMAL(10,6),
                        volatility_20 DECIMAL(10,6),
                        
                        volume_ma_5 DECIMAL(20,2),
                        volume_ma_20 DECIMAL(20,2),
                        volume_ratio DECIMAL(8,4),
                        momentum_5 DECIMAL(10,6),
                        momentum_20 DECIMAL(10,6),
                        
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        
                        UNIQUE(ticker, date)
                    )
                """)
                
                # Create indexes for performance
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date 
                    ON chain_of_alpha_market_data(ticker, date DESC)
                """)
                
                # Prepare data for insertion
                records = []
                for date, row in data.iterrows():
                    # Convert datetime to UTC if needed
                    if hasattr(date, 'tz_convert'):
                        date_utc = date.tz_convert(timezone.utc)
                    else:
                        date_utc = date.replace(tzinfo=US_EASTERN).astimezone(timezone.utc)
                    
                    record = [
                        ticker,
                        date_utc,
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        int(row['volume']),
                        
                        # Optional fields - handle NaN values
                        float(row['returns']) if pd.notna(row['returns']) else None,
                        float(row['log_returns']) if pd.notna(row['log_returns']) else None,
                        
                        # TA-LIB indicators
                        float(row['sma_5']) if pd.notna(row['sma_5']) else None,
                        float(row['sma_10']) if pd.notna(row['sma_10']) else None,
                        float(row['sma_20']) if pd.notna(row['sma_20']) else None,
                        float(row['sma_50']) if pd.notna(row['sma_50']) else None,
                        float(row['ema_12']) if pd.notna(row['ema_12']) else None,
                        float(row['ema_26']) if pd.notna(row['ema_26']) else None,
                        
                        float(row['macd']) if pd.notna(row['macd']) else None,
                        float(row['macd_signal']) if pd.notna(row['macd_signal']) else None,
                        float(row['macd_hist']) if pd.notna(row['macd_hist']) else None,
                        
                        float(row['rsi']) if pd.notna(row['rsi']) else None,
                        
                        float(row['bb_upper']) if pd.notna(row['bb_upper']) else None,
                        float(row['bb_middle']) if pd.notna(row['bb_middle']) else None,
                        float(row['bb_lower']) if pd.notna(row['bb_lower']) else None,
                        
                        float(row['stoch_k']) if pd.notna(row['stoch_k']) else None,
                        float(row['stoch_d']) if pd.notna(row['stoch_d']) else None,
                        float(row['williams_r']) if pd.notna(row['williams_r']) else None,
                        float(row['atr']) if pd.notna(row['atr']) else None,
                        
                        float(row['volume_sma']) if pd.notna(row['volume_sma']) else None,
                        float(row['obv']) if pd.notna(row['obv']) else None,
                        float(row['ad']) if pd.notna(row['ad']) else None,
                        
                        float(row['momentum']) if pd.notna(row['momentum']) else None,
                        float(row['roc']) if pd.notna(row['roc']) else None,
                        float(row['volatility_20']) if pd.notna(row['volatility_20']) else None,
                        
                        float(row['volume_ma_5']) if pd.notna(row['volume_ma_5']) else None,
                        float(row['volume_ma_20']) if pd.notna(row['volume_ma_20']) else None,
                        float(row['volume_ratio']) if pd.notna(row['volume_ratio']) else None,
                        float(row['momentum_5']) if pd.notna(row['momentum_5']) else None,
                        float(row['momentum_20']) if pd.notna(row['momentum_20']) else None,
                    ]
                    records.append(record)
                
                # Batch insert with conflict handling
                await conn.executemany("""
                    INSERT INTO chain_of_alpha_market_data 
                    (ticker, date, open, high, low, close, volume,
                     returns, log_returns,
                     sma_5, sma_10, sma_20, sma_50, ema_12, ema_26,
                     macd, macd_signal, macd_hist, rsi,
                     bb_upper, bb_middle, bb_lower,
                     stoch_k, stoch_d, williams_r, atr,
                     volume_sma, obv, ad,
                     momentum, roc, volatility_20,
                     volume_ma_5, volume_ma_20, volume_ratio,
                     momentum_5, momentum_20)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                            $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29,
                            $30, $31, $32, $33, $34, $35, $36, $37)
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        returns = EXCLUDED.returns,
                        log_returns = EXCLUDED.log_returns,
                        sma_5 = EXCLUDED.sma_5,
                        sma_10 = EXCLUDED.sma_10,
                        sma_20 = EXCLUDED.sma_20,
                        sma_50 = EXCLUDED.sma_50,
                        ema_12 = EXCLUDED.ema_12,
                        ema_26 = EXCLUDED.ema_26,
                        macd = EXCLUDED.macd,
                        macd_signal = EXCLUDED.macd_signal,
                        macd_hist = EXCLUDED.macd_hist,
                        rsi = EXCLUDED.rsi,
                        bb_upper = EXCLUDED.bb_upper,
                        bb_middle = EXCLUDED.bb_middle,
                        bb_lower = EXCLUDED.bb_lower,
                        stoch_k = EXCLUDED.stoch_k,
                        stoch_d = EXCLUDED.stoch_d,
                        williams_r = EXCLUDED.williams_r,
                        atr = EXCLUDED.atr,
                        volume_sma = EXCLUDED.volume_sma,
                        obv = EXCLUDED.obv,
                        ad = EXCLUDED.ad,
                        momentum = EXCLUDED.momentum,
                        roc = EXCLUDED.roc,
                        volatility_20 = EXCLUDED.volatility_20,
                        volume_ma_5 = EXCLUDED.volume_ma_5,
                        volume_ma_20 = EXCLUDED.volume_ma_20,
                        volume_ratio = EXCLUDED.volume_ratio,
                        momentum_5 = EXCLUDED.momentum_5,
                        momentum_20 = EXCLUDED.momentum_20
                """, records)
                
                logger.info(f"[SUCCESS] Persisted {len(records)} records for {ticker}")
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to persist data for {ticker}: {e}")
            raise
    
    def _combine_ticker_data(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine individual ticker DataFrames into unified structure.
        
        Args:
            all_data: Dictionary of ticker -> DataFrame
            
        Returns:
            Combined DataFrame with MultiIndex (ticker, date)
        """
        try:
            logger.info("[COMBINE] Combining ticker data into unified structure")
            
            dfs_with_ticker = []
            for ticker, df in all_data.items():
                df_copy = df.copy()
                df_copy['ticker'] = ticker
                df_copy.set_index('ticker', append=True, inplace=True)
                df_copy = df_copy.reorder_levels(['ticker', df_copy.index.names[0]])
                dfs_with_ticker.append(df_copy)
            
            # Concatenate all DataFrames
            combined = pd.concat(dfs_with_ticker, axis=0, sort=False)
            
            # Sort by ticker and date
            combined = combined.sort_index()
            
            logger.info(f"[SUCCESS] Combined data shape: {combined.shape}")
            return combined
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to combine ticker data: {e}")
            raise
    
    async def load_data_from_db(self, tickers: List[str] = None, 
                              start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load market data from PostgreSQL database.
        
        Args:
            tickers: List of tickers to load (default: all)
            start_date: Start date for data range
            end_date: End date for data range
            
        Returns:
            DataFrame with market data and indicators
        """
        try:
            logger.info("[DB] Loading market data from PostgreSQL")
            
            conn = await asyncpg.connect(self.database_url)
            
            try:
                # Build query
                query = "SELECT * FROM chain_of_alpha_market_data WHERE 1=1"
                params = []
                param_count = 0
                
                if tickers:
                    param_count += 1
                    query += f" AND ticker = ANY(${param_count})"
                    params.append(tickers)
                
                if start_date:
                    param_count += 1
                    query += f" AND date >= ${param_count}"
                    params.append(start_date)
                
                if end_date:
                    param_count += 1
                    query += f" AND date <= ${param_count}"
                    params.append(end_date)
                
                query += " ORDER BY ticker, date"
                
                # Execute query
                rows = await conn.fetch(query, *params)
                
                if not rows:
                    logger.warning("[WARNING] No data found in database")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame([dict(row) for row in rows])
                
                # Set proper index
                df['date'] = pd.to_datetime(df['date'])
                df.set_index(['ticker', 'date'], inplace=True)
                
                # Remove metadata columns
                df = df.drop(columns=['id', 'created_at'], errors='ignore')
                
                logger.info(f"[SUCCESS] Loaded {len(df)} records from database")
                return df
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to load data from database: {e}")
            raise
    
    def get_current_time(self) -> datetime:
        """Get current time in US Eastern timezone."""
        return datetime.now(US_EASTERN)
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'ib') and self.ib.isConnected():
            self.disconnect_from_gateway()