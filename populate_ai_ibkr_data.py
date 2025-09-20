"""
AI Historical Data IBKR Integration

Production script that integrates IBKR data collection from tws_api_project
with the AI advancements database structure. This version uses real IBKR
connections to populate the ai_historical_market_data table.
"""

import asyncio
import asyncpg
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Import IBKR libraries
try:
    from ib_insync import IB, Stock, util
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_ibkr_data_population.log')
    ]
)
logger = logging.getLogger(__name__)

class AIIBKRDataCollector:
    """Production IBKR data collector for AI historical market data"""
    
    def __init__(self):
        self.database_url = "postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway"
        
        # IBKR Connection settings
        self.ibkr_host = '127.0.0.1'
        self.ibkr_port = 4002  # Gateway port (use 7497 for TWS)
        self.client_id = 350   # Unique client ID
        
        # AI Stock Universe for data collection
        self.priority_stocks = {
            1: ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META'],  # Critical AI leaders
            2: ['ACM', 'ADSK', 'AEIS', 'AEM', 'AFRM'],  # Proven successful from requirements
            3: ['PLTR', 'HOOD', 'RKLB', 'IREN', 'ANET', 'AMD', 'INTC'],    # AI/Tech growth
            4: ['NFLX', 'CRM', 'ORCL', 'AVGO', 'QCOM', 'SHOP', 'DDOG'],   # Extended tech
            5: ['SPY', 'QQQ', 'XLK']  # Reference ETFs
        }
        
        # Data collection parameters (based on data_requirements.md)
        self.target_duration = "3 Y"  # 3 years for optimal 157+ weekly bars
        self.min_bars_required = 100
        self.optimal_bars_target = 157
        
        # Timeframe mapping for IBKR
        self.timeframe_mapping = {
            '1week': '1 week',
            '1month': '1 month', 
            '1day': '1 day',
            '1hour': '1 hour',
            '5min': '5 mins',
            '1min': '1 min'
        }
        
        # Rate limiting
        self.request_delay = 0.2  # 200ms between requests
        self.batch_delay = 2.0    # 2s between batches
        
        self.ib = None
        self.connection_active = False
        
    async def connect_to_database(self) -> asyncpg.Connection:
        """Connect to Railway PostgreSQL database"""
        try:
            conn = await asyncpg.connect(self.database_url)
            logger.info("[SUCCESS] Connected to Railway PostgreSQL database")
            return conn
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to database: {e}")
            raise
    
    def connect_to_ibkr(self) -> bool:
        """Connect to IBKR Gateway on port 4002"""
        if not IBKR_AVAILABLE:
            logger.error("[ERROR] ib_insync not available. Install with: pip install ib_insync")
            return False

        try:
            self.ib = IB()

            try:
                logger.info(f"[IBKR] Attempting connection to Gateway on port {self.ibkr_port}...")
                # Use synchronous connection without asyncio
                import nest_asyncio
                nest_asyncio.apply()  # Allow nested event loops

                self.ib.connect(self.ibkr_host, self.ibkr_port, self.client_id, timeout=15)
                self.connection_active = True
                logger.info(f"[SUCCESS] Connected to IBKR Gateway on port {self.ibkr_port}")
                return True
            except Exception as e:
                logger.error(f"[ERROR] Failed to connect to IBKR Gateway on port {self.ibkr_port}: {e}")
                return False

        except Exception as e:
            logger.error(f"[ERROR] IBKR connection setup failed: {e}")
            return False
    
    def disconnect_from_ibkr(self):
        """Disconnect from IBKR"""
        if self.ib and self.connection_active:
            try:
                self.ib.disconnect()
                self.connection_active = False
                logger.info("[DISCONNECTED] IBKR connection closed")
            except Exception as e:
                logger.warning(f"[WARNING] Error during IBKR disconnect: {e}")
    
    async def fetch_symbol_data(self, symbol: str, timeframe: str = '1week') -> Optional[pd.DataFrame]:
        """Fetch historical data for a symbol from IBKR"""
        try:
            logger.info(f"[IBKR] Fetching {timeframe} data for {symbol}")
            
            # Create stock contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Qualify contract to get primary exchange
            try:
                self.ib.qualifyContracts(contract)
            except Exception as e:
                logger.warning(f"[WARNING] Contract qualification failed for {symbol}: {e}")
            
            # Request historical data
            bar_size = self.timeframe_mapping.get(timeframe, '1 week')
            
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime='',  # Empty means current time
                durationStr=self.target_duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,  # Regular trading hours only
                formatDate=1,
                keepUpToDate=False
            )
            
            if not bars:
                logger.warning(f"[NO DATA] No historical data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            df = util.df(bars)
            
            if df.empty:
                logger.warning(f"[EMPTY] Empty DataFrame for {symbol}")
                return None
            
            # Add symbol and timeframe columns
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            # Ensure proper timezone handling (convert to UTC)
            if df.index.tz is None:
                # Assume Eastern timezone if no timezone info
                df.index = df.index.tz_localize('US/Eastern').tz_convert('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            df.rename(columns={'date': 'timestamp'}, inplace=True)
            
            # Ensure numeric types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"[SUCCESS] Retrieved {len(df)} bars for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to fetch data for {symbol}: {e}")
            return None
        finally:
            # Rate limiting
            await asyncio.sleep(self.request_delay)
    
    async def store_data_to_db(self, conn: asyncpg.Connection, df: pd.DataFrame) -> int:
        """Store DataFrame to ai_historical_market_data table"""
        try:
            if df.empty:
                return 0
            
            symbol = df['symbol'].iloc[0]
            timeframe = df['timeframe'].iloc[0]
            
            # Prepare records for insertion
            records = []
            for _, row in df.iterrows():
                records.append((
                    row['symbol'],
                    row['timeframe'],
                    row['timestamp'],
                    Decimal(str(row['open'])),
                    Decimal(str(row['high'])),
                    Decimal(str(row['low'])),
                    Decimal(str(row['close'])),
                    int(row['volume']) if pd.notna(row['volume']) else 0,
                    'IBKR',  # source
                    'live'   # data_quality
                ))
            
            # Insert with UPSERT to handle duplicates
            result = await conn.executemany("""
                INSERT INTO ai_historical_market_data (
                    symbol, timeframe, timestamp, open_price, high_price, 
                    low_price, close_price, volume, source, data_quality
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (symbol, timeframe, timestamp) 
                DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume,
                    updated_at = CURRENT_TIMESTAMP
            """, records)
            
            logger.info(f"[DB] Stored {len(records)} bars for {symbol} ({timeframe})")
            return len(records)
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to store data: {e}")
            return 0
    
    async def process_symbol(self, conn: asyncpg.Connection, symbol: str, priority: int) -> Dict[str, Any]:
        """Process data collection for a single symbol"""
        result = {
            'symbol': symbol,
            'priority': priority,
            'success': False,
            'timeframe_used': None,
            'bars_collected': 0,
            'error': None,
            'duration_seconds': 0
        }
        
        start_time = time.time()
        
        try:
            logger.info(f"[PROCESSING] {symbol} (Priority {priority})")
            
            # Try timeframes in preference order: weekly -> monthly -> daily
            timeframes_to_try = ['1week', '1month', '1day']
            
            for timeframe in timeframes_to_try:
                try:
                    # Fetch data from IBKR
                    df = await self.fetch_symbol_data(symbol, timeframe)
                    
                    if df is not None and not df.empty:
                        # Check if we have sufficient data
                        bar_count = len(df)
                        
                        if bar_count >= self.min_bars_required:
                            # Store to database
                            stored_count = await self.store_data_to_db(conn, df)
                            
                            if stored_count > 0:
                                result.update({
                                    'success': True,
                                    'timeframe_used': timeframe,
                                    'bars_collected': stored_count
                                })
                                
                                status = "OPTIMAL" if bar_count >= self.optimal_bars_target else "ADEQUATE"
                                logger.info(f"[SUCCESS] {symbol}: {stored_count} bars ({timeframe}) - {status}")
                                break
                        else:
                            logger.warning(f"[INSUFFICIENT] {symbol} {timeframe}: {bar_count} bars < {self.min_bars_required} required")
                    
                except Exception as e:
                    logger.error(f"[ERROR] {symbol} {timeframe}: {e}")
                    continue
            
            if not result['success']:
                result['error'] = 'No adequate data in any timeframe'
                logger.error(f"[FAILED] {symbol}: {result['error']}")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"[ERROR] {symbol}: {e}")
        
        result['duration_seconds'] = time.time() - start_time
        return result
    
    async def check_existing_data(self, conn: asyncpg.Connection, symbols: List[str]) -> Dict[str, bool]:
        """Check which symbols already have adequate data"""
        try:
            result = await conn.fetch("""
                SELECT symbol, COUNT(*) as bar_count
                FROM ai_historical_market_data 
                WHERE symbol = ANY($1) AND timeframe IN ('1week', '1month')
                GROUP BY symbol
                HAVING COUNT(*) >= $2
            """, symbols, self.min_bars_required)
            
            adequate_data = {row['symbol']: True for row in result}
            return {symbol: adequate_data.get(symbol, False) for symbol in symbols}
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to check existing data: {e}")
            return {symbol: False for symbol in symbols}
    
    async def run_data_collection(self, max_symbols_per_priority: int = 10, 
                                 priorities_to_process: List[int] = [1, 2]):
        """Run the complete IBKR data collection process"""
        logger.info("[START] AI IBKR Historical Data Collection")
        
        if not IBKR_AVAILABLE:
            logger.error("[ABORT] ib_insync library not available")
            return False
        
        try:
            # Connect to services
            logger.info("[CONNECTING] Services...")
            
            # Database connection
            conn = await self.connect_to_database()
            
            # IBKR connection
            if not self.connect_to_ibkr():
                logger.error("[ABORT] Failed to connect to IBKR")
                return False
            
            # Build processing list
            symbols_to_process = []
            for priority in sorted(priorities_to_process):
                if priority in self.priority_stocks:
                    priority_symbols = self.priority_stocks[priority][:max_symbols_per_priority]
                    for symbol in priority_symbols:
                        symbols_to_process.append((symbol, priority))
            
            logger.info(f"[PLAN] Processing {len(symbols_to_process)} symbols")
            
            # Check existing data
            all_symbols = [s[0] for s in symbols_to_process]
            existing_data_status = await self.check_existing_data(conn, all_symbols)
            
            # Filter symbols that need data
            symbols_needing_data = [
                (symbol, priority) for symbol, priority in symbols_to_process 
                if not existing_data_status.get(symbol, False)
            ]
            
            logger.info(f"[ANALYSIS] {len(symbols_needing_data)} symbols need data collection")
            
            # Process symbols
            results = []
            total_symbols = len(symbols_needing_data)
            
            for i, (symbol, priority) in enumerate(symbols_needing_data, 1):
                logger.info(f"[PROGRESS] {i}/{total_symbols}: {symbol}")
                
                result = await self.process_symbol(conn, symbol, priority)
                results.append(result)
                
                # Batch delay after every 5 symbols
                if i % 5 == 0:
                    logger.info(f"[BATCH] Completed {i} symbols, pausing for {self.batch_delay}s...")
                    await asyncio.sleep(self.batch_delay)
            
            # Generate summary
            successful = [r for r in results if r['success']]
            failed = [r for r in results if not r['success']]
            total_bars = sum(r['bars_collected'] for r in successful)
            
            # Print summary
            print("\n" + "="*80)
            print("AI IBKR DATA COLLECTION SUMMARY")
            print("="*80)
            print(f"Total Symbols Processed: {len(results)}")
            print(f"Successful: {len(successful)}")
            print(f"Failed: {len(failed)}")
            print(f"Success Rate: {len(successful)/len(results)*100:.1f}%" if results else "N/A")
            print(f"Total Bars Collected: {total_bars:,}")
            
            if successful:
                print("\nSuccessful Collections:")
                for result in successful:
                    print(f"  ✅ {result['symbol']}: {result['bars_collected']} bars ({result['timeframe_used']})")
            
            if failed:
                print("\nFailed Collections:")
                for result in failed[:10]:  # Show first 10
                    print(f"  ❌ {result['symbol']}: {result['error']}")
            
            print("="*80)
            
            # Cleanup connections
            await conn.close()
            self.disconnect_from_ibkr()
            
            logger.info("[COMPLETE] Data collection finished")
            return len(successful) > 0
            
        except Exception as e:
            logger.error(f"[ERROR] Data collection failed: {e}")
            return False

async def main():
    """Main execution function"""
    print("🚀 AI IBKR Historical Data Collection")
    print("=====================================")
    print("⚠️  REQUIREMENTS:")
    print("   - IBKR Gateway or TWS must be running")
    print("   - API connections enabled in IBKR")
    print("   - ai_historical_market_data table must exist")
    print("")
    
    # Confirm IBKR availability
    if not IBKR_AVAILABLE:
        print("❌ ib_insync library not installed")
        print("   Install with: pip install ib_insync")
        return 1
    
    collector = AIIBKRDataCollector()
    
    # Run data collection for priority 1 & 2 stocks
    success = await collector.run_data_collection(
        max_symbols_per_priority=7,  # Top 7 from each priority
        priorities_to_process=[1, 2]  # AI leaders and proven stocks
    )
    
    if success:
        print("\n✅ Data collection completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Verify data quality in database")
        print("2. Run AI trading algorithms on collected data")
        print("3. Expand to additional priorities if needed")
        print("4. Set up automated daily updates")
    else:
        print("\n❌ Data collection had issues")
        print("Check IBKR connection and logs for details")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)