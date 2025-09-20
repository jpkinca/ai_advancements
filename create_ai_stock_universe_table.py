"""
AI Stock Universe Table Creator

Creates a dedicated table for AI stock universe with symbols from ai_historical_market_data.
This table serves as the master list of stocks for AI trading models and analysis.

Table: ai_stock_universe
Purpose: Store unique stock symbols with metadata for AI analysis
"""

import asyncio
import asyncpg
import logging
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class AIStockUniverseCreator:
    """Create AI stock universe table from historical market data"""

    def __init__(self):
        self.database_url = "postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway"
        self.table_name = "ai_stock_universe"

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
        """Generate SQL for creating AI stock universe table"""

        return f"""
        -- AI Stock Universe Table
        -- Master list of stocks for AI trading analysis

        DROP TABLE IF EXISTS {self.table_name};

        CREATE TABLE {self.table_name} (
            symbol VARCHAR(20) PRIMARY KEY,
            company_name VARCHAR(100),
            sector VARCHAR(50),
            industry VARCHAR(50),
            market_cap BIGINT,
            is_active BOOLEAN DEFAULT TRUE,
            added_date DATE DEFAULT CURRENT_DATE,
            source VARCHAR(20) DEFAULT 'IBKR',
            last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB -- Additional stock information
        );

        -- Performance indexes
        CREATE INDEX idx_ai_stock_universe_sector ON {self.table_name}(sector);
        CREATE INDEX idx_ai_stock_universe_industry ON {self.table_name}(industry);
        CREATE INDEX idx_ai_stock_universe_active ON {self.table_name}(is_active) WHERE is_active = true;
        CREATE INDEX idx_ai_stock_universe_source ON {self.table_name}(source);

        -- Add comments for documentation
        COMMENT ON TABLE {self.table_name} IS 'AI stock universe - master list of stocks for AI trading models';
        COMMENT ON COLUMN {self.table_name}.symbol IS 'Stock ticker symbol (primary key)';
        COMMENT ON COLUMN {self.table_name}.company_name IS 'Full company name';
        COMMENT ON COLUMN {self.table_name}.sector IS 'GICS sector classification';
        COMMENT ON COLUMN {self.table_name}.industry IS 'GICS industry classification';
        COMMENT ON COLUMN {self.table_name}.market_cap IS 'Market capitalization in USD';
        COMMENT ON COLUMN {self.table_name}.is_active IS 'Whether stock is actively traded and included in universe';
        COMMENT ON COLUMN {self.table_name}.source IS 'Data source (IBKR, manual, etc.)';
        COMMENT ON COLUMN {self.table_name}.metadata IS 'Additional stock metadata as JSON';
        """

    def get_population_sql(self) -> str:
        """Generate SQL to populate table with unique symbols from ai_historical_market_data"""

        return f"""
        -- Populate AI stock universe with unique symbols from historical data
        INSERT INTO {self.table_name} (symbol, source, added_date)
        SELECT DISTINCT
            symbol,
            'IBKR' as source,
            CURRENT_DATE as added_date
        FROM ai_historical_market_data
        WHERE symbol IS NOT NULL
        ON CONFLICT (symbol) DO NOTHING;

        -- Update metadata with basic info
        UPDATE {self.table_name}
        SET metadata = jsonb_build_object(
            'data_available', true,
            'timeframes', (
                SELECT jsonb_agg(DISTINCT timeframe)
                FROM ai_historical_market_data
                WHERE ai_historical_market_data.symbol = {self.table_name}.symbol
            ),
            'date_range', (
                SELECT jsonb_build_object(
                    'start', MIN(timestamp)::date,
                    'end', MAX(timestamp)::date,
                    'total_bars', COUNT(*)
                )
                FROM ai_historical_market_data
                WHERE ai_historical_market_data.symbol = {self.table_name}.symbol
            )
        )
        WHERE metadata IS NULL;
        """

    async def create_table(self, conn: asyncpg.Connection) -> bool:
        """Create the AI stock universe table"""

        try:
            # Create table and indexes
            table_sql = self.get_table_creation_sql()
            await conn.execute(table_sql)

            logger.info(f"[SUCCESS] Created table {self.table_name} with indexes")
            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to create table: {e}")
            return False

    async def populate_table(self, conn: asyncpg.Connection) -> bool:
        """Populate table with symbols from ai_historical_market_data"""

        try:
            # Check if ai_historical_market_data exists and has data
            data_check = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = 'ai_historical_market_data' AND table_schema = 'public'
            """)

            if data_check == 0:
                logger.warning("[WARNING] ai_historical_market_data table does not exist")
                logger.info("[INFO] Creating empty ai_stock_universe table")
                return True

            # Get count of symbols in historical data
            symbol_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT symbol) FROM ai_historical_market_data
                WHERE symbol IS NOT NULL
            """)

            if symbol_count == 0:
                logger.warning("[WARNING] No symbols found in ai_historical_market_data")
                return True

            logger.info(f"[INFO] Found {symbol_count} unique symbols in historical data")

            # Populate table
            population_sql = self.get_population_sql()
            await conn.execute(population_sql)

            # Get final count
            final_count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.table_name}")

            logger.info(f"[SUCCESS] Populated {self.table_name} with {final_count} stocks")
            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to populate table: {e}")
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
                SELECT symbol, company_name, sector, is_active, source,
                       metadata->>'data_available' as data_available,
                       metadata->'timeframes' as timeframes
                FROM {self.table_name}
                ORDER BY symbol
                LIMIT 5
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
        logger.info("    AI STOCK UNIVERSE TABLE CREATION RESULTS")
        logger.info("=" * 80)

        if verification['success']:
            logger.info("")
            logger.info(f"[SUCCESS] Table {self.table_name} created successfully!")
            logger.info(f"  Columns: {verification['columns']}")
            logger.info(f"  Indexes: {verification['indexes']}")
            logger.info(f"  Stocks: {verification['rows']}")

            if verification['sample_data']:
                logger.info("")
                logger.info("[SAMPLE STOCKS]")
                for i, row in enumerate(verification['sample_data'], 1):
                    logger.info(f"  {i}. {row['symbol']} - Active: {row['is_active']}, Source: {row['source']}")
                    if row['timeframes']:
                        logger.info(f"     Timeframes: {row['timeframes']}")

            logger.info("")
            logger.info("[TABLE FEATURES]")
            logger.info("  ✓ Primary key on symbol")
            logger.info("  ✓ Sector and industry classification ready")
            logger.info("  ✓ Active/inactive status tracking")
            logger.info("  ✓ JSON metadata for extensibility")
            logger.info("  ✓ Auto-populated from historical data")
            logger.info("  ✓ Optimized indexes for AI queries")

            logger.info("")
            logger.info("[NEXT STEPS]")
            logger.info("  1. Update company names, sectors, industries manually or via API")
            logger.info("  2. Set market_cap values for portfolio optimization")
            logger.info("  3. Use in AI models: SELECT symbol FROM ai_stock_universe WHERE is_active = true")

        else:
            logger.info("")
            logger.info(f"[ERROR] Failed to create table: {verification['error']}")

        logger.info("")
        logger.info("=" * 80)

    async def run_creation(self):
        """Run complete table creation process"""

        logger.info(f"Creating AI stock universe table: {self.table_name}")

        try:
            conn = await self.connect_to_database()

            # Create table
            table_created = await self.create_table(conn)

            if not table_created:
                logger.error("[ERROR] Failed to create table")
                return False

            # Populate table
            table_populated = await self.populate_table(conn)

            if not table_populated:
                logger.error("[ERROR] Failed to populate table")
                return False

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
    """Create AI stock universe table"""

    creator = AIStockUniverseCreator()
    success = await creator.run_creation()

    if success:
        logger.info("[FINAL RESULT] AI stock universe table ready for use")
        return 0
    else:
        logger.info("[FINAL RESULT] Table creation failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())