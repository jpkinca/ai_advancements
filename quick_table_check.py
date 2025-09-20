"""
Quick verification of AI table creation
"""

import asyncio
import asyncpg
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def check_table():
    database_url = "postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway"
    
    try:
        conn = await asyncpg.connect(database_url)
        
        # Check if table exists
        exists = await conn.fetchval("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name = 'ai_historical_market_data' AND table_schema = 'public'
        """)
        
        if exists:
            # Get sample data
            sample = await conn.fetch("""
                SELECT symbol, timeframe, open_price, close_price, volume 
                FROM ai_historical_market_data 
                LIMIT 3
            """)
            
            logger.info(f"[SUCCESS] ai_historical_market_data table exists with {len(sample)} sample rows")
            for row in sample:
                logger.info(f"  {row['symbol']} {row['timeframe']}: ${row['open_price']} -> ${row['close_price']}")
        else:
            logger.info("[INFO] ai_historical_market_data table not found")
        
        await conn.close()
        
    except Exception as e:
        logger.error(f"[ERROR] Check failed: {e}")

if __name__ == "__main__":
    asyncio.run(check_table())
