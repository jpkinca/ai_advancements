"""
AI Historical Market Data Table Creator

Creates a dedicated table for AI analysis with all required OHLC data and timeframes.
This table is separate from the existing historical_market_data table which lacks
the necessary columns for our AI modules.

Table: ai_historical_market_data
Purpose: Store multi-timeframe OHLC data from IBKR for AI analysis
"""

import asyncio
import asyncpg
import logging
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class AIMarketDataTableCreator:
    """Create AI-optimized historical market data table"""
    
    def __init__(self):
        self.database_url = "postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway"
        self.table_name = "ai_historical_market_data"
    
    async def connect_to_database(self) -> asyncpg.Connection:
        """Connect to Railway PostgreSQL database"""
        try:
            conn = await asyncpg.connect(self.database_url)
            logger.info("[SUCCESS] Connected to Railway PostgreSQL database")
            return conn
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect: {e}")
            raise
    
    def get_table_creation_sql(self) -> str:
        """Generate SQL for creating AI historical market data table"""
        
        return f"""
        -- AI Historical Market Data Table
        -- Optimized for multi-timeframe OHLC analysis
        
        DROP TABLE IF EXISTS {self.table_name};
        
        CREATE TABLE {self.table_name} (
            id BIGSERIAL PRIMARY KEY,
            
            -- Symbol and timeframe identification
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            
            -- OHLC timestamp (Eastern Time for US markets)
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            
            -- OHLC price data (using DECIMAL for precision)
            open_price DECIMAL(15, 6) NOT NULL,
            high_price DECIMAL(15, 6) NOT NULL,
            low_price DECIMAL(15, 6) NOT NULL,
            close_price DECIMAL(15, 6) NOT NULL,
            
            -- Volume data (BIGINT for large volumes)
            volume BIGINT NOT NULL DEFAULT 0,
            
            -- Data source and metadata
            source VARCHAR(20) DEFAULT 'IBKR',
            data_quality VARCHAR(20) DEFAULT 'live',
            
            -- Record management
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            
            -- Ensure no duplicate OHLC bars
            UNIQUE(symbol, timeframe, timestamp)
        );
        
        -- Performance indexes for AI queries
        
        -- Primary lookup index (symbol + timeframe + time range)
        CREATE INDEX idx_ai_market_symbol_timeframe_timestamp 
        ON {self.table_name}(symbol, timeframe, timestamp DESC);
        
        -- Symbol-only index for cross-timeframe analysis
        CREATE INDEX idx_ai_market_symbol_timestamp 
        ON {self.table_name}(symbol, timestamp DESC);
        
        -- Timeframe analysis index
        CREATE INDEX idx_ai_market_timeframe_timestamp 
        ON {self.table_name}(timeframe, timestamp DESC);
        
        -- Recent data index (for real-time AI processing)
        CREATE INDEX idx_ai_market_recent_data 
        ON {self.table_name}(timestamp DESC) 
        WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days';
        
        -- Data quality index (filter for live vs historical)
        CREATE INDEX idx_ai_market_quality_timestamp 
        ON {self.table_name}(data_quality, timestamp DESC);
        
        -- Volume analysis index (for high-volume filtering)
        CREATE INDEX idx_ai_market_volume 
        ON {self.table_name}(volume DESC) 
        WHERE volume > 1000000;
        
        -- Add comments for documentation
        COMMENT ON TABLE {self.table_name} IS 'AI-optimized historical market data with multi-timeframe OHLC data from IBKR';
        COMMENT ON COLUMN {self.table_name}.symbol IS 'Stock symbol (AAPL, NVDA, etc.)';
        COMMENT ON COLUMN {self.table_name}.timeframe IS 'Data timeframe (1min, 5min, 15min, 1hour, 4hour, 1day, 1month)';
        COMMENT ON COLUMN {self.table_name}.timestamp IS 'OHLC bar timestamp in Eastern Time';
        COMMENT ON COLUMN {self.table_name}.open_price IS 'Opening price with 6 decimal precision';
        COMMENT ON COLUMN {self.table_name}.high_price IS 'High price with 6 decimal precision';
        COMMENT ON COLUMN {self.table_name}.low_price IS 'Low price with 6 decimal precision';
        COMMENT ON COLUMN {self.table_name}.close_price IS 'Closing price with 6 decimal precision';
        COMMENT ON COLUMN {self.table_name}.volume IS 'Trading volume (number of shares)';
        COMMENT ON COLUMN {self.table_name}.source IS 'Data source (IBKR, etc.)';
        COMMENT ON COLUMN {self.table_name}.data_quality IS 'Data quality flag (live, historical, backfilled)';
        """
    
    def get_sample_data_sql(self) -> str:
        """Generate SQL for inserting sample data"""
        
        return f"""
        -- Insert sample data for testing
        INSERT INTO {self.table_name} (
            symbol, timeframe, timestamp, 
            open_price, high_price, low_price, close_price, volume,
            source, data_quality
        ) VALUES 
        -- NVDA 1-minute data
        ('NVDA', '1min', '2025-08-31 09:30:00-05:00', 125.50, 126.00, 125.30, 125.80, 1500000, 'IBKR', 'live'),
        ('NVDA', '1min', '2025-08-31 09:31:00-05:00', 125.80, 126.20, 125.60, 126.10, 1200000, 'IBKR', 'live'),
        
        -- NVDA 5-minute data
        ('NVDA', '5min', '2025-08-31 09:30:00-05:00', 125.50, 126.50, 125.20, 126.30, 8500000, 'IBKR', 'live'),
        
        -- NVDA 1-hour data
        ('NVDA', '1hour', '2025-08-31 09:00:00-05:00', 125.00, 127.00, 124.50, 126.50, 45000000, 'IBKR', 'live'),
        
        -- AAPL data
        ('AAPL', '1min', '2025-08-31 09:30:00-05:00', 180.25, 180.75, 180.10, 180.60, 2000000, 'IBKR', 'live'),
        ('AAPL', '5min', '2025-08-31 09:30:00-05:00', 180.25, 181.00, 179.80, 180.90, 12000000, 'IBKR', 'live'),
        
        -- PLTR data
        ('PLTR', '1min', '2025-08-31 09:30:00-05:00', 28.50, 28.80, 28.40, 28.70, 800000, 'IBKR', 'live')
        
        ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING;
        """
    
    async def create_table(self, conn: asyncpg.Connection) -> bool:
        """Create the AI historical market data table"""
        
        try:
            # Create table and indexes
            table_sql = self.get_table_creation_sql()
            await conn.execute(table_sql)
            
            logger.info(f"[SUCCESS] Created table {self.table_name} with indexes")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to create table: {e}")
            return False
    
    async def insert_sample_data(self, conn: asyncpg.Connection) -> bool:
        """Insert sample data for testing"""
        
        try:
            sample_sql = self.get_sample_data_sql()
            await conn.execute(sample_sql)
            
            logger.info("[SUCCESS] Inserted sample OHLC data")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to insert sample data: {e}")
            return False
    
    async def verify_table_creation(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        """Verify table was created correctly"""
        
        try:
            # Check table exists
            table_check = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = $1 AND table_schema = 'public'
            """, self.table_name)
            
            if table_check == 0:
                return {'success': False, 'error': 'Table not found'}
            
            # Get column count
            column_count = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.columns 
                WHERE table_name = $1 AND table_schema = 'public'
            """, self.table_name)
            
            # Get index count
            index_count = await conn.fetchval("""
                SELECT COUNT(*) FROM pg_indexes 
                WHERE tablename = $1 AND schemaname = 'public'
            """, self.table_name)
            
            # Get row count
            row_count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.table_name}")
            
            # Get sample data
            sample_data = await conn.fetch(f"""
                SELECT symbol, timeframe, timestamp, open_price, close_price, volume 
                FROM {self.table_name} 
                ORDER BY timestamp DESC 
                LIMIT 3
            """)
            
            return {
                'success': True,
                'columns': column_count,
                'indexes': index_count,
                'rows': row_count,
                'sample_data': [dict(row) for row in sample_data]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def display_results(self, verification: Dict[str, Any]):
        """Display table creation results"""
        
        logger.info("=" * 80)
        logger.info("    AI HISTORICAL MARKET DATA TABLE CREATION RESULTS")
        logger.info("=" * 80)
        
        if verification['success']:
            logger.info("")
            logger.info(f"[SUCCESS] Table {self.table_name} created successfully!")
            logger.info(f"  Columns: {verification['columns']}")
            logger.info(f"  Indexes: {verification['indexes']}")
            logger.info(f"  Sample Rows: {verification['rows']}")
            
            if verification['sample_data']:
                logger.info("")
                logger.info("[SAMPLE DATA]")
                for i, row in enumerate(verification['sample_data'], 1):
                    logger.info(f"  {i}. {row['symbol']} {row['timeframe']} @ {row['timestamp']}")
                    logger.info(f"     Open: ${row['open_price']}, Close: ${row['close_price']}, Vol: {row['volume']:,}")
            
            logger.info("")
            logger.info("[TABLE FEATURES]")
            logger.info("  ✓ Multi-timeframe support (1min to 1month)")
            logger.info("  ✓ High-precision DECIMAL prices (6 decimal places)")
            logger.info("  ✓ BIGINT volume for large trading volumes")
            logger.info("  ✓ Optimized indexes for AI queries")
            logger.info("  ✓ Eastern Time timestamp support")
            logger.info("  ✓ Data quality tracking")
            logger.info("  ✓ Unique constraint prevents duplicates")
            
            logger.info("")
            logger.info("[NEXT STEPS]")
            logger.info("  1. Update multi_timeframe_data_manager.py to use ai_historical_market_data")
            logger.info("  2. Run watchlist_manager.py to create watchlist_management table")
            logger.info("  3. Test AI pipeline with critical_priority_test.py")
            
        else:
            logger.info("")
            logger.info(f"[ERROR] Failed to create table: {verification['error']}")
        
        logger.info("")
        logger.info("=" * 80)
    
    async def run_creation(self):
        """Run complete table creation process"""
        
        logger.info(f"Creating AI-optimized historical market data table: {self.table_name}")
        
        try:
            conn = await self.connect_to_database()
            
            # Create table
            table_created = await self.create_table(conn)
            
            if not table_created:
                logger.error("[ERROR] Failed to create table")
                return False
            
            # Insert sample data
            sample_inserted = await self.insert_sample_data(conn)
            
            # Verify creation
            verification = await self.verify_table_creation(conn)
            
            # Display results
            self.display_results(verification)
            
            await conn.close()
            
            return verification['success']
            
        except Exception as e:
            logger.error(f"[ERROR] Table creation failed: {e}")
            return False

async def main():
    """Create AI historical market data table"""
    
    creator = AIMarketDataTableCreator()
    success = await creator.run_creation()
    
    if success:
        logger.info("[FINAL RESULT] AI historical market data table ready for use")
        return 0
    else:
        logger.info("[FINAL RESULT] Table creation failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
