"""
Complete AI Module Database Setup

Creates all necessary tables for the AI trading module:
1. ai_historical_market_data - OHLC data storage
2. watchlist_management - Track stocks through AI pipeline
3. ai_analysis_results - Store AI predictions and signals

Fixed version with proper PostgreSQL syntax.
"""

import asyncio
import asyncpg
import logging
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class AIModuleDatabaseSetup:
    """Complete database setup for AI trading module"""
    
    def __init__(self):
        self.database_url = "postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway"
        
        self.tables_to_create = [
            'ai_historical_market_data',
            'watchlist_management', 
            'ai_analysis_results'
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
    
    def get_ai_historical_market_data_sql(self) -> str:
        """SQL for AI historical market data table (fixed version)"""
        
        return """
        -- AI Historical Market Data Table
        -- Stores multi-timeframe OHLC data from IBKR for AI analysis
        
        DROP TABLE IF EXISTS ai_historical_market_data CASCADE;
        
        CREATE TABLE ai_historical_market_data (
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
        
        -- Performance indexes for AI queries (fixed - no CURRENT_TIMESTAMP in predicates)
        
        -- Primary lookup index (symbol + timeframe + time range)
        CREATE INDEX idx_ai_market_symbol_timeframe_timestamp 
        ON ai_historical_market_data(symbol, timeframe, timestamp DESC);
        
        -- Symbol-only index for cross-timeframe analysis
        CREATE INDEX idx_ai_market_symbol_timestamp 
        ON ai_historical_market_data(symbol, timestamp DESC);
        
        -- Timeframe analysis index
        CREATE INDEX idx_ai_market_timeframe_timestamp 
        ON ai_historical_market_data(timeframe, timestamp DESC);
        
        -- Data quality index (filter for live vs historical)
        CREATE INDEX idx_ai_market_quality_timestamp 
        ON ai_historical_market_data(data_quality, timestamp DESC);
        
        -- Volume analysis index (for high-volume filtering)
        CREATE INDEX idx_ai_market_high_volume 
        ON ai_historical_market_data(volume DESC) 
        WHERE volume > 1000000;
        
        -- Recent data composite index (most common AI query pattern)
        CREATE INDEX idx_ai_market_recent_composite 
        ON ai_historical_market_data(symbol, timeframe, timestamp DESC, close_price);
        
        -- Add table and column comments
        COMMENT ON TABLE ai_historical_market_data IS 'AI-optimized historical market data with multi-timeframe OHLC data from IBKR';
        COMMENT ON COLUMN ai_historical_market_data.symbol IS 'Stock symbol (AAPL, NVDA, etc.)';
        COMMENT ON COLUMN ai_historical_market_data.timeframe IS 'Data timeframe (1min, 5min, 15min, 1hour, 4hour, 1day, 1month)';
        COMMENT ON COLUMN ai_historical_market_data.timestamp IS 'OHLC bar timestamp in Eastern Time';
        COMMENT ON COLUMN ai_historical_market_data.open_price IS 'Opening price with 6 decimal precision';
        COMMENT ON COLUMN ai_historical_market_data.volume IS 'Trading volume (number of shares)';
        """
    
    def get_watchlist_management_sql(self) -> str:
        """SQL for watchlist management table"""
        
        return """
        -- Watchlist Management Table
        -- Tracks stocks through the AI analysis pipeline
        
        DROP TABLE IF EXISTS watchlist_management CASCADE;
        
        CREATE TABLE watchlist_management (
            id SERIAL PRIMARY KEY,
            
            -- Stock identification
            symbol VARCHAR(20) NOT NULL UNIQUE,
            company_name VARCHAR(100),
            sector VARCHAR(50),
            
            -- Priority and scheduling
            priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
            is_active BOOLEAN DEFAULT true,
            
            -- AI Analysis tracking
            last_analysis_run TIMESTAMP WITH TIME ZONE,
            next_analysis_due TIMESTAMP WITH TIME ZONE,
            analysis_frequency_minutes INTEGER DEFAULT 60,
            
            -- AI Results summary
            latest_signal VARCHAR(20), -- 'BUY', 'SELL', 'HOLD', 'NEUTRAL'
            signal_strength DECIMAL(3, 2), -- 0.00 to 1.00
            confidence_score DECIMAL(3, 2), -- 0.00 to 1.00
            
            -- Performance tracking
            analysis_count INTEGER DEFAULT 0,
            successful_predictions INTEGER DEFAULT 0,
            failed_predictions INTEGER DEFAULT 0,
            
            -- Data collection status
            last_data_fetch TIMESTAMP WITH TIME ZONE,
            data_completeness_score DECIMAL(3, 2) DEFAULT 0.00,
            
            -- Record management
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes for watchlist management
        
        -- Priority-based scheduling index
        CREATE INDEX idx_watchlist_priority_active 
        ON watchlist_management(priority DESC, next_analysis_due) 
        WHERE is_active = true;
        
        -- Sector analysis index
        CREATE INDEX idx_watchlist_sector_active 
        ON watchlist_management(sector, is_active);
        
        -- Analysis scheduling index
        CREATE INDEX idx_watchlist_analysis_due 
        ON watchlist_management(next_analysis_due) 
        WHERE is_active = true AND next_analysis_due IS NOT NULL;
        
        -- Performance tracking index
        CREATE INDEX idx_watchlist_performance 
        ON watchlist_management(signal_strength DESC, confidence_score DESC) 
        WHERE is_active = true;
        
        -- Data fetch status index
        CREATE INDEX idx_watchlist_data_status 
        ON watchlist_management(last_data_fetch, data_completeness_score);
        
        -- Add comments
        COMMENT ON TABLE watchlist_management IS 'Manages 50-stock watchlist for AI analysis pipeline';
        COMMENT ON COLUMN watchlist_management.priority IS 'Analysis priority: 1=highest, 10=lowest';
        COMMENT ON COLUMN watchlist_management.signal_strength IS 'Strength of latest AI signal (0.0-1.0)';
        COMMENT ON COLUMN watchlist_management.confidence_score IS 'AI confidence in prediction (0.0-1.0)';
        """
    
    def get_ai_analysis_results_sql(self) -> str:
        """SQL for AI analysis results table"""
        
        return """
        -- AI Analysis Results Table
        -- Stores detailed AI predictions and analysis results
        
        DROP TABLE IF EXISTS ai_analysis_results CASCADE;
        
        CREATE TABLE ai_analysis_results (
            id BIGSERIAL PRIMARY KEY,
            
            -- Analysis identification
            symbol VARCHAR(20) NOT NULL,
            analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            analysis_type VARCHAR(30) NOT NULL, -- 'PPO', 'Portfolio', 'Fourier', 'Wavelet'
            timeframe VARCHAR(10) NOT NULL,
            
            -- AI Module results
            ppo_signal VARCHAR(20), -- PPOTrader results
            ppo_confidence DECIMAL(5, 4),
            ppo_entry_price DECIMAL(15, 6),
            ppo_exit_price DECIMAL(15, 6),
            ppo_stop_loss DECIMAL(15, 6),
            
            portfolio_weight DECIMAL(5, 4), -- PortfolioOptimizer results
            portfolio_risk_score DECIMAL(5, 4),
            portfolio_expected_return DECIMAL(8, 6),
            
            fourier_trend VARCHAR(20), -- FourierAnalyzer results
            fourier_cycle_strength DECIMAL(5, 4),
            fourier_dominant_period INTEGER,
            
            wavelet_volatility DECIMAL(8, 6), -- WaveletAnalyzer results
            wavelet_trend_strength DECIMAL(5, 4),
            wavelet_noise_level DECIMAL(5, 4),
            
            -- Combined AI decision
            final_signal VARCHAR(20) NOT NULL, -- 'BUY', 'SELL', 'HOLD', 'NEUTRAL'
            final_confidence DECIMAL(5, 4) NOT NULL,
            signal_strength DECIMAL(5, 4) NOT NULL,
            
            -- Market context
            market_price DECIMAL(15, 6),
            market_volume BIGINT,
            market_volatility DECIMAL(8, 6),
            
            -- Analysis metadata
            data_points_used INTEGER,
            computation_time_ms INTEGER,
            error_messages TEXT,
            
            -- Record management
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes for AI analysis results
        
        -- Primary query index (symbol + time)
        CREATE INDEX idx_ai_results_symbol_timestamp 
        ON ai_analysis_results(symbol, analysis_timestamp DESC);
        
        -- Analysis type index
        CREATE INDEX idx_ai_results_type_timestamp 
        ON ai_analysis_results(analysis_type, analysis_timestamp DESC);
        
        -- Signal strength index (for filtering strong signals)
        CREATE INDEX idx_ai_results_strong_signals 
        ON ai_analysis_results(final_signal, signal_strength DESC, final_confidence DESC) 
        WHERE signal_strength > 0.7;
        
        -- Timeframe analysis index
        CREATE INDEX idx_ai_results_timeframe 
        ON ai_analysis_results(timeframe, analysis_timestamp DESC);
        
        -- Performance tracking index
        CREATE INDEX idx_ai_results_performance 
        ON ai_analysis_results(final_confidence DESC, signal_strength DESC);
        
        -- Error tracking index
        CREATE INDEX idx_ai_results_errors 
        ON ai_analysis_results(analysis_timestamp DESC) 
        WHERE error_messages IS NOT NULL;
        
        -- Add comments
        COMMENT ON TABLE ai_analysis_results IS 'Detailed results from all AI analysis modules';
        COMMENT ON COLUMN ai_analysis_results.final_signal IS 'Combined signal from all AI modules';
        COMMENT ON COLUMN ai_analysis_results.signal_strength IS 'Overall strength of the trading signal (0.0-1.0)';
        COMMENT ON COLUMN ai_analysis_results.final_confidence IS 'AI confidence in the final decision (0.0-1.0)';
        """
    
    def get_sample_data_sql(self) -> List[str]:
        """Generate sample data for all tables"""
        
        return [
            # Sample OHLC data
            """
            INSERT INTO ai_historical_market_data (
                symbol, timeframe, timestamp, 
                open_price, high_price, low_price, close_price, volume,
                source, data_quality
            ) VALUES 
            ('NVDA', '1min', '2025-08-31 09:30:00-05:00', 125.50, 126.00, 125.30, 125.80, 1500000, 'IBKR', 'live'),
            ('NVDA', '1min', '2025-08-31 09:31:00-05:00', 125.80, 126.20, 125.60, 126.10, 1200000, 'IBKR', 'live'),
            ('NVDA', '5min', '2025-08-31 09:30:00-05:00', 125.50, 126.50, 125.20, 126.30, 8500000, 'IBKR', 'live'),
            ('AAPL', '1min', '2025-08-31 09:30:00-05:00', 180.25, 180.75, 180.10, 180.60, 2000000, 'IBKR', 'live'),
            ('PLTR', '1min', '2025-08-31 09:30:00-05:00', 28.50, 28.80, 28.40, 28.70, 800000, 'IBKR', 'live')
            ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING;
            """,
            
            # Sample watchlist data
            """
            INSERT INTO watchlist_management (
                symbol, company_name, sector, priority, 
                analysis_frequency_minutes, latest_signal, signal_strength, confidence_score
            ) VALUES 
            ('NVDA', 'NVIDIA Corporation', 'Technology', 1, 15, 'BUY', 0.85, 0.92),
            ('PLTR', 'Palantir Technologies', 'Technology', 1, 15, 'HOLD', 0.65, 0.78),
            ('HOOD', 'Robinhood Markets', 'Financial', 2, 30, 'NEUTRAL', 0.45, 0.60),
            ('RKLB', 'Rocket Lab USA', 'Aerospace', 2, 30, 'BUY', 0.75, 0.82),
            ('IREN', 'Iris Energy Limited', 'Energy', 3, 60, 'SELL', 0.70, 0.75),
            ('ANET', 'Arista Networks', 'Technology', 1, 15, 'BUY', 0.80, 0.88)
            ON CONFLICT (symbol) DO NOTHING;
            """,
            
            # Sample AI analysis results
            """
            INSERT INTO ai_analysis_results (
                symbol, analysis_type, timeframe, 
                ppo_signal, ppo_confidence, 
                final_signal, final_confidence, signal_strength,
                market_price, market_volume
            ) VALUES 
            ('NVDA', 'PPO', '5min', 'BUY', 0.8500, 'BUY', 0.9200, 0.8500, 125.80, 8500000),
            ('PLTR', 'Portfolio', '1hour', 'HOLD', 0.6500, 'HOLD', 0.7800, 0.6500, 28.70, 15000000),
            ('AAPL', 'Fourier', '15min', 'BUY', 0.7200, 'BUY', 0.7800, 0.7200, 180.60, 25000000)
            ON CONFLICT DO NOTHING;
            """
        ]
    
    async def create_table(self, conn: asyncpg.Connection, table_name: str, sql: str) -> bool:
        """Create a single table"""
        try:
            await conn.execute(sql)
            logger.info(f"[SUCCESS] Created table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to create {table_name}: {e}")
            return False
    
    async def insert_sample_data(self, conn: asyncpg.Connection, sample_queries: List[str]) -> int:
        """Insert sample data into all tables"""
        success_count = 0
        
        for i, query in enumerate(sample_queries, 1):
            try:
                await conn.execute(query)
                logger.info(f"[SUCCESS] Inserted sample data set {i}")
                success_count += 1
            except Exception as e:
                logger.error(f"[ERROR] Failed to insert sample data set {i}: {e}")
        
        return success_count
    
    async def verify_setup(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        """Verify all tables were created successfully"""
        
        verification = {
            'tables_created': 0,
            'total_tables': len(self.tables_to_create),
            'table_details': {},
            'success': False
        }
        
        for table_name in self.tables_to_create:
            try:
                # Check table exists
                exists = await conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name = $1 AND table_schema = 'public'
                """, table_name)
                
                if exists:
                    # Get table stats
                    row_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                    column_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM information_schema.columns 
                        WHERE table_name = $1 AND table_schema = 'public'
                    """, table_name)
                    
                    verification['table_details'][table_name] = {
                        'exists': True,
                        'rows': row_count,
                        'columns': column_count
                    }
                    verification['tables_created'] += 1
                else:
                    verification['table_details'][table_name] = {'exists': False}
                    
            except Exception as e:
                verification['table_details'][table_name] = {'exists': False, 'error': str(e)}
        
        verification['success'] = verification['tables_created'] == verification['total_tables']
        return verification
    
    def display_results(self, verification: Dict[str, Any]):
        """Display setup results"""
        
        logger.info("=" * 80)
        logger.info("    AI MODULE DATABASE SETUP RESULTS")
        logger.info("=" * 80)
        
        logger.info("")
        logger.info(f"[OVERALL] Created {verification['tables_created']}/{verification['total_tables']} tables")
        
        logger.info("")
        logger.info("[TABLE STATUS]")
        for table_name, details in verification['table_details'].items():
            if details['exists']:
                logger.info(f"  ✓ {table_name}: {details['columns']} columns, {details['rows']} rows")
            else:
                error_msg = details.get('error', 'Unknown error')
                logger.info(f"  ✗ {table_name}: Failed - {error_msg}")
        
        if verification['success']:
            logger.info("")
            logger.info("[SUCCESS] Complete AI module database setup!")
            logger.info("")
            logger.info("[AVAILABLE TABLES]")
            logger.info("  1. ai_historical_market_data - Multi-timeframe OHLC data storage")
            logger.info("  2. watchlist_management - 50-stock AI pipeline tracking")
            logger.info("  3. ai_analysis_results - Detailed AI predictions and signals")
            
            logger.info("")
            logger.info("[NEXT STEPS]")
            logger.info("  1. Update multi_timeframe_data_manager.py to use ai_historical_market_data")
            logger.info("  2. Update watchlist_manager.py to use watchlist_management")
            logger.info("  3. Run critical_priority_test.py to test complete AI pipeline")
            logger.info("  4. Begin weekend AI testing with 6 priority stocks")
        else:
            logger.info("")
            logger.info("[PARTIAL SUCCESS] Some tables failed to create")
            logger.info("Review errors above and retry failed table creation")
        
        logger.info("")
        logger.info("=" * 80)
    
    async def run_complete_setup(self):
        """Run complete AI module database setup"""
        
        logger.info("Starting complete AI module database setup...")
        
        try:
            conn = await self.connect_to_database()
            
            # Create all tables
            table_sql_map = {
                'ai_historical_market_data': self.get_ai_historical_market_data_sql(),
                'watchlist_management': self.get_watchlist_management_sql(),
                'ai_analysis_results': self.get_ai_analysis_results_sql()
            }
            
            tables_created = 0
            for table_name, sql in table_sql_map.items():
                if await self.create_table(conn, table_name, sql):
                    tables_created += 1
            
            # Insert sample data
            sample_queries = self.get_sample_data_sql()
            sample_sets_inserted = await self.insert_sample_data(conn, sample_queries)
            
            logger.info(f"[SUCCESS] Inserted {sample_sets_inserted}/{len(sample_queries)} sample data sets")
            
            # Verify setup
            verification = await self.verify_setup(conn)
            
            # Display results
            self.display_results(verification)
            
            await conn.close()
            
            return verification['success']
            
        except Exception as e:
            logger.error(f"[ERROR] Complete setup failed: {e}")
            return False

async def main():
    """Run complete AI module database setup"""
    
    setup = AIModuleDatabaseSetup()
    success = await setup.run_complete_setup()
    
    if success:
        logger.info("[FINAL RESULT] AI module database setup completed successfully")
        return 0
    else:
        logger.info("[FINAL RESULT] AI module database setup failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
