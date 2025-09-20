"""
Simplified PostgreSQL Schema Setup for Chain-of-Alpha

Creates essential tables for production deployment.
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE_URL = 'postgresql://postgres:TAqEkujnMknVURCcrYTIDOzQXbgBNtSX@turntable.proxy.rlwy.net:10410/railway'

async def create_simple_schema():
    """Create simplified schema for Chain-of-Alpha."""
    
    logger.info("Creating simplified PostgreSQL schema...")
    
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # 1. Market Data Table
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
                returns DECIMAL(10,6),
                rsi DECIMAL(8,4),
                macd DECIMAL(10,6),
                sma_20 DECIMAL(15,4),
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)
        logger.info("✅ Market data table")
        
        # 2. AI Factors Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chain_of_alpha_factors (
                id SERIAL PRIMARY KEY,
                factor_id INTEGER NOT NULL,
                factor_name VARCHAR(100) NOT NULL,
                formula TEXT NOT NULL,
                rationale TEXT,
                information_coefficient DECIMAL(8,6),
                generated_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("✅ Factors table")
        
        # 3. Pipeline Results Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chain_of_alpha_results (
                id SERIAL PRIMARY KEY,
                execution_id VARCHAR(50) NOT NULL,
                total_return DECIMAL(10,6),
                sharpe_ratio DECIMAL(8,4),
                factors_generated INTEGER,
                execution_time INTERVAL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("✅ Results table")
        
        # Test data insertion
        test_time = datetime.now()
        await conn.execute("""
            INSERT INTO chain_of_alpha_market_data (ticker, date, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (ticker, date) DO NOTHING
        """, 'TEST', test_time, 100, 101, 99, 100.5, 1000000)
        
        count = await conn.fetchval("SELECT COUNT(*) FROM chain_of_alpha_market_data WHERE ticker = 'TEST'")
        logger.info(f"✅ Test insertion: {count} record")
        
        # Cleanup
        await conn.execute("DELETE FROM chain_of_alpha_market_data WHERE ticker = 'TEST'")
        
        await conn.close()
        
        logger.info("🎉 Schema created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Schema creation failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(create_simple_schema())