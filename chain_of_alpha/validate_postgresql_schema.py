"""
PostgreSQL Schema Validation for Chain-of-Alpha Production

Validates and creates required database schema for:
- Market data storage (OHLCV + TA-LIB indicators)
- AI factor storage
- Performance tracking
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE_URL = 'postgresql://postgres:TAqEkujnMknVURCcrYTIDOzQXbgBNtSX@turntable.proxy.rlwy.net:10410/railway'

async def validate_postgresql_schema():
    """Validate and create PostgreSQL schema for Chain-of-Alpha."""
    
    logger.info("="*60)
    logger.info("POSTGRESQL SCHEMA VALIDATION")
    logger.info("="*60)
    
    try:
        # Connect to database
        logger.info("Connecting to PostgreSQL database...")
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Get database version
        version = await conn.fetchval('SELECT version()')
        logger.info(f"✅ Connected: {version.split(',')[0]}")
        
        # Create schema for Chain-of-Alpha
        logger.info("\nCreating Chain-of-Alpha schema...")
        
        # 1. Market Data Table (IBKR + TA-LIB)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chain_of_alpha_market_data (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) NOT NULL,
                date TIMESTAMPTZ NOT NULL,
                
                -- OHLCV Data from IBKR
                open DECIMAL(15,4) NOT NULL,
                high DECIMAL(15,4) NOT NULL,
                low DECIMAL(15,4) NOT NULL,
                close DECIMAL(15,4) NOT NULL,
                volume BIGINT NOT NULL,
                
                -- Basic derived features
                returns DECIMAL(10,6),
                log_returns DECIMAL(10,6),
                
                -- TA-LIB Moving Averages
                sma_5 DECIMAL(15,4),
                sma_10 DECIMAL(15,4),
                sma_20 DECIMAL(15,4),
                sma_50 DECIMAL(15,4),
                ema_12 DECIMAL(15,4),
                ema_26 DECIMAL(15,4),
                
                -- TA-LIB Momentum Indicators
                macd DECIMAL(10,6),
                macd_signal DECIMAL(10,6),
                macd_hist DECIMAL(10,6),
                rsi DECIMAL(8,4),
                
                -- TA-LIB Volatility Indicators  
                bb_upper DECIMAL(15,4),
                bb_middle DECIMAL(15,4),
                bb_lower DECIMAL(15,4),
                atr DECIMAL(10,6),
                
                -- TA-LIB Oscillators
                stoch_k DECIMAL(8,4),
                stoch_d DECIMAL(8,4),
                williams_r DECIMAL(8,4),
                
                -- TA-LIB Volume Indicators
                volume_sma DECIMAL(20,2),
                obv DECIMAL(20,2),
                ad DECIMAL(20,2),
                
                -- Additional momentum
                momentum DECIMAL(10,6),
                roc DECIMAL(10,6),
                volatility_20 DECIMAL(10,6),
                
                -- Volume analysis
                volume_ma_5 DECIMAL(20,2),
                volume_ma_20 DECIMAL(20,2),
                volume_ratio DECIMAL(8,4),
                momentum_5 DECIMAL(10,6),
                momentum_20 DECIMAL(10,6),
                
                -- Metadata
                data_source VARCHAR(20) DEFAULT 'IBKR_GATEWAY',
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                
                -- Constraints
                UNIQUE(ticker, date),
                CHECK (open > 0),
                CHECK (high >= open),
                CHECK (low <= open),
                CHECK (close > 0),
                CHECK (volume >= 0)
            )
        """)
        logger.info("✅ Market data table created")
        
        # 2. AI Generated Factors Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chain_of_alpha_factors (
                id SERIAL PRIMARY KEY,
                factor_id INTEGER NOT NULL,
                factor_name VARCHAR(100) NOT NULL,
                formula TEXT NOT NULL,
                rationale TEXT,
                signal_direction VARCHAR(20),
                implementation_notes TEXT,
                raw_response TEXT,
                
                -- Performance metrics
                information_coefficient DECIMAL(8,6),
                evaluation_score DECIMAL(8,6),
                
                -- Model info
                model_name VARCHAR(100) DEFAULT 'meta-llama/Llama-3.2-3B-Instruct',
                generation_parameters JSONB,
                
                -- Metadata
                generated_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            )
        """)
        logger.info("✅ AI factors table created")
        
        # 3. Factor Performance Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chain_of_alpha_factor_performance (
                id SERIAL PRIMARY KEY,
                factor_id INTEGER NOT NULL,
                evaluation_date TIMESTAMPTZ NOT NULL,
                
                -- Performance metrics
                information_coefficient DECIMAL(8,6),
                rank_correlation DECIMAL(8,6),
                return_correlation DECIMAL(8,6),
                
                -- Statistical measures
                mean_factor_value DECIMAL(10,6),
                std_factor_value DECIMAL(10,6),
                skewness DECIMAL(8,6),
                kurtosis DECIMAL(8,6),
                
                -- Coverage
                coverage_ratio DECIMAL(6,4),
                observations_count INTEGER,
                
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            )
        """)
        logger.info("✅ Factor performance table created")
        
        # 4. Portfolio Weights Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chain_of_alpha_portfolio (
                id SERIAL PRIMARY KEY,
                rebalance_date TIMESTAMPTZ NOT NULL,
                ticker VARCHAR(20) NOT NULL,
                weight DECIMAL(8,6) NOT NULL,
                
                -- Factor contributions
                factor_score DECIMAL(8,6),
                rank_score INTEGER,
                
                -- Position info
                target_weight DECIMAL(8,6),
                actual_weight DECIMAL(8,6),
                
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(rebalance_date, ticker),
                CHECK (weight >= 0),
                CHECK (weight <= 1)
            )
        """)
        logger.info("✅ Portfolio weights table created")
        
        # 5. Backtest Results Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chain_of_alpha_backtest (
                id SERIAL PRIMARY KEY,
                backtest_date TIMESTAMPTZ NOT NULL,
                
                -- Performance metrics
                total_return DECIMAL(10,6),
                annualized_return DECIMAL(10,6),
                volatility DECIMAL(10,6),
                sharpe_ratio DECIMAL(8,4),
                max_drawdown DECIMAL(10,6),
                
                -- Risk metrics
                beta DECIMAL(8,4),
                alpha DECIMAL(8,4),
                information_ratio DECIMAL(8,4),
                tracking_error DECIMAL(8,4),
                
                -- Portfolio stats
                num_positions INTEGER,
                turnover DECIMAL(8,4),
                
                -- Configuration
                config_snapshot JSONB,
                factor_ids INTEGER[],
                
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("✅ Backtest results table created")
        
        # 6. Pipeline Execution Log
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chain_of_alpha_pipeline_log (
                id SERIAL PRIMARY KEY,
                execution_id UUID DEFAULT gen_random_uuid(),
                pipeline_version VARCHAR(50) NOT NULL,
                
                -- Execution info
                start_time TIMESTAMPTZ NOT NULL,
                end_time TIMESTAMPTZ,
                status VARCHAR(20) NOT NULL, -- running, completed, failed
                
                -- Data info
                tickers VARCHAR(500),
                date_range_start DATE,
                date_range_end DATE,
                data_points INTEGER,
                
                -- Results summary
                factors_generated INTEGER,
                top_factor_score DECIMAL(8,6),
                portfolio_return DECIMAL(10,6),
                
                -- Error info
                error_message TEXT,
                error_traceback TEXT,
                
                -- Compliance
                data_source VARCHAR(50) DEFAULT 'IBKR_GATEWAY',
                technical_library VARCHAR(50) DEFAULT 'TA-LIB',
                timezone VARCHAR(50) DEFAULT 'US/Eastern',
                
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("✅ Pipeline execution log table created")
        
        # Create indexes for performance
        logger.info("\nCreating database indexes...")
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date ON chain_of_alpha_market_data(ticker, date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_market_data_date ON chain_of_alpha_market_data(date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_factors_generated_at ON chain_of_alpha_factors(generated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_factor_performance_date ON chain_of_alpha_factor_performance(evaluation_date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_rebalance_date ON chain_of_alpha_portfolio(rebalance_date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_backtest_date ON chain_of_alpha_backtest(backtest_date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_pipeline_log_start_time ON chain_of_alpha_pipeline_log(start_time DESC)"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
        
        logger.info("✅ Performance indexes created")
        
        # Test data insertion
        logger.info("\nTesting data insertion...")
        
        # Test market data insert
        await conn.execute("""
            INSERT INTO chain_of_alpha_market_data 
            (ticker, date, open, high, low, close, volume, returns)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (ticker, date) DO NOTHING
        """, 'TEST', datetime.now(), 100.0, 101.0, 99.0, 100.5, 1000000, 0.005)
        
        # Test factor insert
        await conn.execute("""
            INSERT INTO chain_of_alpha_factors 
            (factor_id, factor_name, formula, generated_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT DO NOTHING
        """, 1, 'Test Factor', 'rank(close/sma_20)', datetime.now())
        
        # Verify data
        market_count = await conn.fetchval("SELECT COUNT(*) FROM chain_of_alpha_market_data")
        factor_count = await conn.fetchval("SELECT COUNT(*) FROM chain_of_alpha_factors")
        
        logger.info(f"✅ Data insertion test: {market_count} market records, {factor_count} factors")
        
        # Clean up test data
        await conn.execute("DELETE FROM chain_of_alpha_market_data WHERE ticker = 'TEST'")
        await conn.execute("DELETE FROM chain_of_alpha_factors WHERE factor_name = 'Test Factor'")
        
        # Get table sizes
        logger.info("\nDatabase schema summary:")
        tables = [
            'chain_of_alpha_market_data',
            'chain_of_alpha_factors', 
            'chain_of_alpha_factor_performance',
            'chain_of_alpha_portfolio',
            'chain_of_alpha_backtest',
            'chain_of_alpha_pipeline_log'
        ]
        
        for table in tables:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            logger.info(f"  {table}: {count} records")
        
        await conn.close()
        
        logger.info("\n" + "="*60)
        logger.info("🎉 POSTGRESQL SCHEMA VALIDATION SUCCESSFUL")
        logger.info("✅ All tables created with proper constraints")
        logger.info("✅ Performance indexes in place") 
        logger.info("✅ Data insertion/retrieval tested")
        logger.info("✅ Ready for production Chain-of-Alpha pipeline")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Schema validation failed: {e}")
        return False


async def main():
    """Main execution."""
    success = await validate_postgresql_schema()
    
    if success:
        logger.info("\n🚀 Database is ready for production!")
        logger.info("Next step: Run the Chain-of-Alpha production pipeline")
    else:
        logger.info("\n🔧 Please fix database issues before proceeding")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())