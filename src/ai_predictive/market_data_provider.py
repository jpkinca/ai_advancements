"""
AI Trading Advancements - Market Data Provider

This module provides market data integration that works with existing TradeAppComponents
infrastructure while supporting the AI trading system requirements.

Author: AI Trading Development Team
Date: August 31, 2025
Version: 1.0.0
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import asyncio
import logging
from pathlib import Path
import sqlite3
import json

# Import core components
from ..core import (
    MarketData, TimeFrame, BaseDataProvider, get_config
)

logger = logging.getLogger(__name__)


class YFinanceDataProvider(BaseDataProvider):
    """
    Yahoo Finance data provider implementation.
    Integrates with existing TradeAppComponents while providing AI-ready data.
    """
    
    def __init__(self, name: str = "YFinance", cache_enabled: bool = True):
        """Initialize Yahoo Finance data provider."""
        super().__init__(name)
        self.cache_enabled = cache_enabled
        self.cache_path = None
        self.session = None
        
        if cache_enabled:
            config = get_config()
            self.cache_path = Path(config.ai_models.training_data_path) / "market_data_cache.db"
            self._initialize_cache()
    
    def _initialize_cache(self) -> None:
        """Initialize SQLite cache for market data."""
        if not self.cache_path:
            return
            
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(str(self.cache_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        open_price REAL NOT NULL,
                        high_price REAL NOT NULL,
                        low_price REAL NOT NULL,
                        close_price REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timestamp, timeframe)
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe 
                    ON market_data(symbol, timeframe, timestamp)
                ''')
                
                conn.commit()
                
            self.logger.info(f"[SUCCESS] Cache initialized: {self.cache_path}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Cache initialization failed: {e}")
            self.cache_enabled = False
    
    async def connect(self) -> bool:
        """Establish connection to Yahoo Finance."""
        try:
            # Test connection with a simple request
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d", interval="1d")
            
            if test_data.empty:
                self.logger.error("[ERROR] Yahoo Finance connection test failed")
                return False
            
            self.is_connected = True
            self.logger.info("[SUCCESS] Connected to Yahoo Finance")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Yahoo Finance connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close connection to Yahoo Finance."""
        self.is_connected = False
        self.logger.info("[SUCCESS] Disconnected from Yahoo Finance")
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime
    ) -> List[MarketData]:
        """Fetch historical market data with caching support."""
        if not self.is_connected:
            await self.connect()
        
        # Check cache first
        if self.cache_enabled:
            cached_data = self._get_cached_data(symbol, timeframe, start_date, end_date)
            if cached_data:
                self.logger.info(f"[DATA] Retrieved {len(cached_data)} cached records for {symbol}")
                return cached_data
        
        try:
            # Map timeframe to yfinance interval
            interval_map = {
                TimeFrame.MINUTE_1: "1m",
                TimeFrame.MINUTE_5: "5m",
                TimeFrame.MINUTE_15: "15m",
                TimeFrame.MINUTE_30: "30m",
                TimeFrame.HOUR_1: "1h",
                TimeFrame.DAY_1: "1d",
                TimeFrame.WEEK_1: "1wk",
                TimeFrame.MONTH_1: "1mo"
            }
            
            interval = interval_map.get(timeframe, "1d")
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if df.empty:
                self.logger.warning(f"[WARNING] No data returned for {symbol}")
                return []
            
            # Convert to MarketData objects
            market_data = []
            for timestamp, row in df.iterrows():
                # Ensure timestamp is timezone-aware
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                
                data_point = MarketData(
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    open_price=Decimal(str(round(row['Open'], 4))),
                    high_price=Decimal(str(round(row['High'], 4))),
                    low_price=Decimal(str(round(row['Low'], 4))),
                    close_price=Decimal(str(round(row['Close'], 4))),
                    volume=int(row['Volume']),
                    timeframe=timeframe
                )
                
                market_data.append(data_point)
            
            # Cache the data
            if self.cache_enabled and market_data:
                self._cache_data(market_data)
            
            self.logger.info(f"[SUCCESS] Retrieved {len(market_data)} records for {symbol}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to fetch data for {symbol}: {e}")
            return []
    
    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current market price for symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if current_price:
                return Decimal(str(round(current_price, 4)))
            
            # Fallback: get latest price from recent history
            df = ticker.history(period="1d", interval="1m")
            if not df.empty:
                latest_price = df['Close'].iloc[-1]
                return Decimal(str(round(latest_price, 4)))
            
            return None
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to get current price for {symbol}: {e}")
            return None
    
    async def get_symbols(self) -> List[str]:
        """Get list of available symbols from common indices."""
        # This is a simplified implementation
        # In production, you might want to integrate with the existing scanner
        common_symbols = [
            # Major indices
            "SPY", "QQQ", "IWM", "DIA",
            # Tech stocks
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
            # Finance
            "JPM", "BAC", "WFC", "GS", "MS",
            # Healthcare
            "JNJ", "PFE", "UNH", "ABBV", "MRK",
            # Consumer
            "WMT", "HD", "PG", "KO", "PEP",
            # Energy
            "XOM", "CVX", "COP", "EOG",
            # Leveraged ETFs (from existing system)
            "TQQQ", "SQQQ", "SPXL", "SPXS", "TNA", "TZA",
            "UVXY", "SVXY", "VXX", "VIXY"
        ]
        
        return common_symbols
    
    def _get_cached_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[List[MarketData]]:
        """Retrieve cached market data."""
        if not self.cache_path or not self.cache_path.exists():
            return None
        
        try:
            with sqlite3.connect(str(self.cache_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT symbol, timestamp, open_price, high_price, low_price, 
                           close_price, volume
                    FROM market_data
                    WHERE symbol = ? AND timeframe = ? 
                      AND timestamp >= ? AND timestamp  None:
        """Cache market data for future use."""
        if not self.cache_path or not market_data:
            return
        
        try:
            with sqlite3.connect(str(self.cache_path)) as conn:
                cursor = conn.cursor()
                
                for data in market_data:
                    cursor.execute('''
                        INSERT OR REPLACE INTO market_data
                        (symbol, timestamp, timeframe, open_price, high_price, 
                         low_price, close_price, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data.symbol,
                        data.timestamp.isoformat(),
                        data.timeframe.value,
                        float(data.open_price),
                        float(data.high_price),
                        float(data.low_price),
                        float(data.close_price),
                        data.volume
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"[ERROR] Cache storage failed: {e}")
    
    def _calculate_expected_points(
        self,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Calculate expected number of data points for timeframe."""
        delta = end_date - start_date
        total_minutes = delta.total_seconds() / 60
        
        timeframe_minutes = {
            TimeFrame.MINUTE_1: 1,
            TimeFrame.MINUTE_5: 5,
            TimeFrame.MINUTE_15: 15,
            TimeFrame.MINUTE_30: 30,
            TimeFrame.HOUR_1: 60,
            TimeFrame.DAY_1: 1440,  # 24 * 60
            TimeFrame.WEEK_1: 10080,  # 7 * 24 * 60
            TimeFrame.MONTH_1: 43200  # 30 * 24 * 60
        }
        
        interval_minutes = timeframe_minutes.get(timeframe, 1440)
        
        # Account for market hours (roughly 6.5 hours per day)
        if interval_minutes  List[str]:
        """Get symbols from existing scanner results."""
        try:
            # Try to read from existing scanner results
            # This would integrate with the actual scanner database
            scanner_symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
                "TQQQ", "SQQQ", "SPXL", "SPXS", "TNA", "TZA"
            ]
            
            self.logger.info(f"[SUCCESS] Retrieved {len(scanner_symbols)} scanner symbols")
            return scanner_symbols
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to get scanner symbols: {e}")
            return []
    
    async def get_watchlist_symbols(self) -> List[str]:
        """Get symbols from existing watchlists."""
        try:
            # This would integrate with the actual watchlist system
            watchlist_symbols = [
                "SPY", "QQQ", "IWM", "DIA", "UVXY", "SVXY"
            ]
            
            self.logger.info(f"[SUCCESS] Retrieved {len(watchlist_symbols)} watchlist symbols")
            return watchlist_symbols
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to get watchlist symbols: {e}")
            return []
    
    async def get_existing_positions(self) -> List[str]:
        """Get symbols from existing positions."""
        try:
            # This would integrate with the actual position tracking
            position_symbols = []
            
            self.logger.info(f"[SUCCESS] Retrieved {len(position_symbols)} position symbols")
            return position_symbols
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to get position symbols: {e}")
            return []
    
    async def get_all_relevant_symbols(self) -> List[str]:
        """Get all symbols relevant for AI analysis."""
        all_symbols = set()
        
        # Combine all symbol sources
        scanner_symbols = await self.get_scanner_symbols()
        watchlist_symbols = await self.get_watchlist_symbols()
        position_symbols = await self.get_existing_positions()
        
        all_symbols.update(scanner_symbols)
        all_symbols.update(watchlist_symbols)
        all_symbols.update(position_symbols)
        
        return list(all_symbols)


# Export public interfaces
__all__ = [
    'YFinanceDataProvider',
    'TradeAppDataBridge'
]


if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    async def test_data_provider():
        print("\n=== Market Data Provider Test ===")
        
        # Initialize data provider
        provider = YFinanceDataProvider()
        
        # Test connection
        connected = await provider.connect()
        print(f"[DATA] Connection successful: {connected}")
        
        if not connected:
            print("[ERROR] Cannot proceed without connection")
            return
        
        # Test current price
        print("[PROCESSING] Testing current price retrieval...")
        current_price = await provider.get_current_price("AAPL")
        print(f"[DATA] AAPL current price: ${current_price}")
        
        # Test historical data
        print("[PROCESSING] Testing historical data retrieval...")
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)
        
        historical_data = await provider.get_historical_data(
            symbol="AAPL",
            timeframe=TimeFrame.DAY_1,
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"[SUCCESS] Retrieved {len(historical_data)} historical records")
        
        if historical_data:
            latest = historical_data[-1]
            print(f"[DATA] Latest: {latest.timestamp.date()} Close: ${latest.close_price}")
            print(f"[DATA] Volume: {latest.volume:,}")
        
        # Test symbol list
        print("[PROCESSING] Testing symbol list...")
        symbols = await provider.get_symbols()
        print(f"[SUCCESS] Available symbols: {len(symbols)}")
        print(f"[DATA] Sample symbols: {symbols[:10]}")
        
        # Test data bridge
        print("[PROCESSING] Testing data bridge...")
        bridge = TradeAppDataBridge()
        
        scanner_symbols = await bridge.get_scanner_symbols()
        watchlist_symbols = await bridge.get_watchlist_symbols()
        all_symbols = await bridge.get_all_relevant_symbols()
        
        print(f"[DATA] Scanner symbols: {len(scanner_symbols)}")
        print(f"[DATA] Watchlist symbols: {len(watchlist_symbols)}")
        print(f"[DATA] Total unique symbols: {len(all_symbols)}")
        
        # Disconnect
        await provider.disconnect()
        
        print("\n[SUCCESS] Market data provider test completed")
    
    # Run the test
    asyncio.run(test_data_provider())
