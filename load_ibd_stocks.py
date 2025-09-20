"""
Load IBD Stocks into AI Stock Universe

Inserts the 50 IBD stocks into the ai_stock_universe table.
"""

import asyncio
import asyncpg
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class IBDStocksLoader:
    """Load IBD stocks into AI stock universe table"""

    def __init__(self):
        self.database_url = "postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway"
        self.table_name = "ai_stock_universe"

        # IBD stocks list
        self.ibd_stocks = [
            'IREN', 'CLS', 'ALAB', 'FUTU', 'RKLB', 'HOOD', 'RDDT', 'AMSC', 'INOD', 'FIX',
            'STOK', 'PLTR', 'DAVE', 'RIGL', 'MIRM', 'HIMS', 'GFI', 'RMBS', 'WLDN', 'TVTX',
            'ANET', 'RYTM', 'APH', 'WGS', 'APP', 'BZ', 'LIF', 'TEM', 'AVDL', 'ANIP',
            'RIOT', 'KNSA', 'STNE', 'SOFI', 'TFPM', 'AEM', 'KGC', 'EME', 'ONC', 'ATAT',
            'AU', 'TBBK', 'PAHC', 'MEDP', 'ARQT', 'CDE', 'MU', 'NVDA', 'NVMI', 'IBKR'
        ]

    async def connect_to_database(self) -> asyncpg.Connection:
        """Connect to Railway PostgreSQL database"""
        try:
            conn = await asyncpg.connect(self.database_url)
            logger.info("[SUCCESS] Connected to Railway PostgreSQL database")
            return conn
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect: {e}")
            raise

    async def load_ibd_stocks(self, conn: asyncpg.Connection) -> bool:
        """Load IBD stocks into the table"""

        try:
            # Prepare data for insertion
            values_list = []
            for symbol in self.ibd_stocks:
                values_list.append(f"('{symbol}', 'IBD', CURRENT_DATE, true)")

            values_str = ", ".join(values_list)

            # Insert query with conflict handling
            insert_query = f"""
            INSERT INTO {self.table_name} (symbol, source, added_date, is_active)
            VALUES {values_str}
            ON CONFLICT (symbol) DO UPDATE SET
                source = EXCLUDED.source,
                is_active = true,
                last_updated = CURRENT_TIMESTAMP
            """

            await conn.execute(insert_query)

            # Update metadata for IBD stocks
            await conn.execute(f"""
            UPDATE {self.table_name}
            SET metadata = COALESCE(metadata, '{{}}'::jsonb) || jsonb_build_object('ibd_selected', true)
            WHERE symbol = ANY($1)
            """, self.ibd_stocks)

            logger.info(f"[SUCCESS] Loaded {len(self.ibd_stocks)} IBD stocks into {self.table_name}")
            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to load IBD stocks: {e}")
            return False

    async def verify_loading(self, conn: asyncpg.Connection) -> dict:
        """Verify the stocks were loaded correctly"""

        try:
            # Count IBD stocks
            ibd_count = await conn.fetchval(f"""
                SELECT COUNT(*) FROM {self.table_name}
                WHERE source = 'IBD' AND is_active = true
            """)

            # Get sample IBD stocks
            sample_ibd = await conn.fetch(f"""
                SELECT symbol, source, is_active, metadata->>'ibd_selected' as ibd_selected
                FROM {self.table_name}
                WHERE source = 'IBD'
                ORDER BY symbol
                LIMIT 10
            """)

            # Total universe count
            total_count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.table_name}")

            return {
                'success': True,
                'ibd_count': ibd_count,
                'total_count': total_count,
                'sample_ibd': [dict(row) for row in sample_ibd]
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def display_results(self, verification: dict):
        """Display loading results"""

        logger.info("=" * 80)
        logger.info("    IBD STOCKS LOADING RESULTS")
        logger.info("=" * 80)

        if verification['success']:
            logger.info("")
            logger.info(f"[SUCCESS] IBD stocks loaded successfully!")
            logger.info(f"  IBD Stocks: {verification['ibd_count']}")
            logger.info(f"  Total Universe: {verification['total_count']}")

            if verification['sample_ibd']:
                logger.info("")
                logger.info("[SAMPLE IBD STOCKS]")
                for i, row in enumerate(verification['sample_ibd'], 1):
                    logger.info(f"  {i}. {row['symbol']} - Active: {row['is_active']}, IBD: {row['ibd_selected']}")

            logger.info("")
            logger.info("[NEXT STEPS]")
            logger.info("  1. Run Sweet Spot & Danger Zone system on these stocks")
            logger.info("  2. Update company metadata (names, sectors, market caps)")
            logger.info("  3. Backtest AI signals on IBD universe")

        else:
            logger.info("")
            logger.info(f"[ERROR] Failed to load stocks: {verification['error']}")

        logger.info("")
        logger.info("=" * 80)

    async def run_loading(self):
        """Run the complete IBD stocks loading process"""

        logger.info("Loading 50 IBD stocks into AI stock universe...")

        try:
            conn = await self.connect_to_database()

            # Load stocks
            loaded = await self.load_ibd_stocks(conn)

            if not loaded:
                logger.error("[ERROR] Failed to load stocks")
                return False

            # Verify loading
            verification = await self.verify_loading(conn)

            # Display results
            self.display_results(verification)

            await conn.close()

            return verification['success']

        except Exception as e:
            logger.error(f"[ERROR] Loading failed: {e}")
            return False

async def main():
    """Load IBD stocks into AI stock universe"""

    loader = IBDStocksLoader()
    success = await loader.run_loading()

    if success:
        logger.info("[FINAL RESULT] IBD stocks loaded successfully")
        return 0
    else:
        logger.info("[FINAL RESULT] Loading failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())