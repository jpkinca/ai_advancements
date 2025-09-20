"""
Comprehensive AI Module Database Verification

Checks all AI tables and provides summary of the complete setup.
"""

import asyncio
import asyncpg
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def verify_ai_database():
    database_url = "postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway"
    
    ai_tables = [
        'ai_historical_market_data',
        'watchlist_management', 
        'ai_analysis_results'
    ]
    
    try:
        conn = await asyncpg.connect(database_url)
        logger.info("[SUCCESS] Connected to Railway PostgreSQL")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("    AI MODULE DATABASE VERIFICATION")
        logger.info("=" * 60)
        
        all_tables_exist = True
        
        for table_name in ai_tables:
            # Check if table exists
            exists = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = $1 AND table_schema = 'public'
            """, table_name)
            
            if exists:
                # Get table details
                row_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                column_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.columns 
                    WHERE table_name = $1 AND table_schema = 'public'
                """, table_name)
                
                logger.info(f"[SUCCESS] {table_name}")
                logger.info(f"  Columns: {column_count}, Rows: {row_count}")
                
                # Show sample data for each table
                if row_count > 0:
                    if table_name == 'ai_historical_market_data':
                        sample = await conn.fetchrow("""
                            SELECT symbol, timeframe, open_price, close_price, volume 
                            FROM ai_historical_market_data 
                            ORDER BY timestamp DESC LIMIT 1
                        """)
                        logger.info(f"  Sample: {sample['symbol']} {sample['timeframe']} ${sample['open_price']} -> ${sample['close_price']}")
                    
                    elif table_name == 'watchlist_management':
                        sample = await conn.fetchrow("""
                            SELECT symbol, sector, priority, latest_signal, signal_strength 
                            FROM watchlist_management 
                            ORDER BY priority LIMIT 1
                        """)
                        logger.info(f"  Sample: {sample['symbol']} ({sample['sector']}) P{sample['priority']} {sample['latest_signal']} {sample['signal_strength']}")
                    
                    elif table_name == 'ai_analysis_results':
                        sample = await conn.fetchrow("""
                            SELECT symbol, analysis_type, final_signal, final_confidence 
                            FROM ai_analysis_results 
                            ORDER BY analysis_timestamp DESC LIMIT 1
                        """)
                        logger.info(f"  Sample: {sample['symbol']} {sample['analysis_type']} {sample['final_signal']} conf:{sample['final_confidence']}")
                
            else:
                logger.info(f"[MISSING] {table_name} - Table not found")
                all_tables_exist = False
        
        logger.info("")
        logger.info("=" * 60)
        
        if all_tables_exist:
            logger.info("[FINAL STATUS] AI Module Database Setup COMPLETE")
            logger.info("")
            logger.info("Ready for:")
            logger.info("  1. Multi-timeframe data collection from IBKR")
            logger.info("  2. 50-stock watchlist management")
            logger.info("  3. AI analysis pipeline (PPO, Portfolio, Fourier, Wavelet)")
            logger.info("  4. Weekend AI testing with paper trading")
        else:
            logger.info("[FINAL STATUS] AI Module Database Setup INCOMPLETE")
            logger.info("Some tables are missing - review setup errors")
        
        await conn.close()
        return all_tables_exist
        
    except Exception as e:
        logger.error(f"[ERROR] Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_ai_database())
    logger.info(f"[EXIT] Database ready: {success}")
