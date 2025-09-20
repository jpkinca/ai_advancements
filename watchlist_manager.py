"""
Watchlist Management System for AI Trading Pipeline

This module creates and manages a PostgreSQL table for tracking stocks
in the AI analysis pipeline, including status, priority, and analysis history.

Features:
- PostgreSQL table for watchlist management
- Status tracking (active, paused, analyzing, completed)
- Priority levels (critical, high, medium, low) 
- Analysis metadata and performance tracking
- Integration with multi_timeframe_data_manager
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import asyncpg
import json
import pytz
from decimal import Decimal
import sys
import os
from urllib.parse import urlparse

# Add TradeAppComponents_fresh to path for Railway database manager
sys.path.append(r'c:\Users\nzcon\VSPython\TradeAppComponents_fresh')
from modules.database.railway_db_manager import RailwayPostgreSQLManager

# Configure logging for ASCII-only output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Eastern Time setup for NYSE/NASDAQ compliance
EASTERN_TZ = pytz.timezone('America/New_York')

def get_railway_database_url() -> str:
    """Get Railway PostgreSQL connection URL for asyncpg"""
    try:
        # Create Railway manager instance to get connection URL
        railway_manager = RailwayPostgreSQLManager()
        database_url = railway_manager.database_url
        
        # Parse URL to ensure it has SSL requirements for Railway
        parsed = urlparse(database_url)
        if parsed.hostname and 'rlwy.net' in parsed.hostname:
            # Add SSL requirement for Railway
            if '?' not in database_url:
                database_url += '?sslmode=require&gssencmode=disable'
            else:
                if 'sslmode' not in database_url:
                    database_url += '&sslmode=require'
                if 'gssencmode' not in database_url:
                    database_url += '&gssencmode=disable'
        
        logger.info(f"[SUCCESS] Railway PostgreSQL URL configured: {parsed.hostname}:{parsed.port}")
        return database_url
    except Exception as e:
        logger.error(f"[ERROR] Failed to get Railway database URL: {e}")
        # No hardcoded fallback; require environment configuration
        env_url = os.getenv('DATABASE_URL', '')
        if env_url:
            if 'sslmode' not in env_url:
                env_url += ('&' if '?' in env_url else '?') + 'sslmode=require'
            if 'gssencmode' not in env_url:
                env_url += ('&' if '?' in env_url else '?') + 'gssencmode=disable'
            logger.info("[SUCCESS] Using DATABASE_URL from environment")
            return env_url
        raise

class WatchlistManager:
    """Manages watchlist symbols for AI trading pipeline"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or get_railway_database_url()
        self.pool = None
        self.logger = logger
        
        # Your 50-symbol watchlist with sectors and priorities
        self.default_watchlist = [
            # Tech & Growth (High Priority)
            {'symbol': 'NVDA', 'sector': 'Technology', 'priority': 'critical', 'market_cap': 'large'},
            {'symbol': 'PLTR', 'sector': 'Technology', 'priority': 'critical', 'market_cap': 'large'},
            {'symbol': 'ANET', 'sector': 'Technology', 'priority': 'high', 'market_cap': 'large'},
            {'symbol': 'FUTU', 'sector': 'Technology', 'priority': 'high', 'market_cap': 'mid'},
            {'symbol': 'RDDT', 'sector': 'Technology', 'priority': 'high', 'market_cap': 'large'},
            {'symbol': 'DOCS', 'sector': 'Technology', 'priority': 'high', 'market_cap': 'mid'},
            
            # Financial Services (High Priority)
            {'symbol': 'HOOD', 'sector': 'Financial', 'priority': 'critical', 'market_cap': 'mid'},
            {'symbol': 'SOFI', 'sector': 'Financial', 'priority': 'high', 'market_cap': 'mid'},
            {'symbol': 'IBKR', 'sector': 'Financial', 'priority': 'high', 'market_cap': 'large'},
            {'symbol': 'STNE', 'sector': 'Financial', 'priority': 'high', 'market_cap': 'mid'},
            
            # Aerospace & Defense (High Priority)
            {'symbol': 'RKLB', 'sector': 'Aerospace', 'priority': 'critical', 'market_cap': 'small'},
            {'symbol': 'TARS', 'sector': 'Aerospace', 'priority': 'high', 'market_cap': 'small'},
            
            # Clean Energy & Infrastructure (Critical)
            {'symbol': 'IREN', 'sector': 'Energy', 'priority': 'critical', 'market_cap': 'small'},
            {'symbol': 'AMSC', 'sector': 'Energy', 'priority': 'high', 'market_cap': 'small'},
            
            # Healthcare & Biotech (High Priority)
            {'symbol': 'ALAB', 'sector': 'Healthcare', 'priority': 'high', 'market_cap': 'small'},
            {'symbol': 'MEDP', 'sector': 'Healthcare', 'priority': 'high', 'market_cap': 'mid'},
            {'symbol': 'PODD', 'sector': 'Healthcare', 'priority': 'high', 'market_cap': 'large'},
            {'symbol': 'ONC', 'sector': 'Healthcare', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'ANIP', 'sector': 'Healthcare', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'RMBS', 'sector': 'Healthcare', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'RYTM', 'sector': 'Healthcare', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'MIRM', 'sector': 'Healthcare', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'LIF', 'sector': 'Healthcare', 'priority': 'medium', 'market_cap': 'small'},
            
            # Mining & Resources (Medium Priority)
            {'symbol': 'GFI', 'sector': 'Mining', 'priority': 'medium', 'market_cap': 'mid'},
            {'symbol': 'AEM', 'sector': 'Mining', 'priority': 'medium', 'market_cap': 'large'},
            {'symbol': 'KGC', 'sector': 'Mining', 'priority': 'medium', 'market_cap': 'large'},
            {'symbol': 'AU', 'sector': 'Mining', 'priority': 'medium', 'market_cap': 'mid'},
            {'symbol': 'WPM', 'sector': 'Mining', 'priority': 'medium', 'market_cap': 'large'},
            {'symbol': 'EGO', 'sector': 'Mining', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'CCJ', 'sector': 'Mining', 'priority': 'high', 'market_cap': 'large'},
            {'symbol': 'EME', 'sector': 'Industrial', 'priority': 'medium', 'market_cap': 'large'},
            
            # Consumer & Retail (Medium Priority)
            {'symbol': 'AFRM', 'sector': 'Financial', 'priority': 'medium', 'market_cap': 'mid'},
            {'symbol': 'CVNA', 'sector': 'Consumer', 'priority': 'medium', 'market_cap': 'mid'},
            {'symbol': 'BROS', 'sector': 'Consumer', 'priority': 'medium', 'market_cap': 'small'},
            
            # International & Emerging (Medium Priority)
            {'symbol': 'BAP', 'sector': 'Financial', 'priority': 'medium', 'market_cap': 'large'},
            {'symbol': 'XPEV', 'sector': 'Automotive', 'priority': 'medium', 'market_cap': 'mid'},
            
            # Specialized Tech & Services (Medium Priority)
            {'symbol': 'CLS', 'sector': 'Technology', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'FIX', 'sector': 'Technology', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'AGX', 'sector': 'Technology', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'OUST', 'sector': 'Technology', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'WLDN', 'sector': 'Technology', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'BZ', 'sector': 'Technology', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'WGS', 'sector': 'Technology', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'TFPM', 'sector': 'Industrial', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'APH', 'sector': 'Technology', 'priority': 'medium', 'market_cap': 'large'},
            {'symbol': 'ATAT', 'sector': 'Technology', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'GH', 'sector': 'Technology', 'priority': 'medium', 'market_cap': 'small'},
            {'symbol': 'TBBK', 'sector': 'Financial', 'priority': 'low', 'market_cap': 'small'},
            {'symbol': 'KNSA', 'sector': 'Technology', 'priority': 'low', 'market_cap': 'small'},
            {'symbol': 'TEM', 'sector': 'Industrial', 'priority': 'low', 'market_cap': 'small'},
        ]
    
    async def connect(self):
        """Create database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            self.logger.info("[SUCCESS] Connected to PostgreSQL database")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to connect to database: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.logger.info("[SUCCESS] Database connection pool closed")
    
    async def create_watchlist_table(self):
        """Create the watchlist management table with all necessary fields"""
        
        create_table_sql = """
        -- Drop existing table if it exists (for clean setup)
        DROP TABLE IF EXISTS watchlist_management CASCADE;
        
        -- Create watchlist management table
        CREATE TABLE watchlist_management (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL UNIQUE,
            
            -- Classification
            sector VARCHAR(50) NOT NULL,
            market_cap VARCHAR(10) CHECK (market_cap IN ('small', 'mid', 'large')),
            priority VARCHAR(10) CHECK (priority IN ('critical', 'high', 'medium', 'low')),
            
            -- Status tracking
            status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'analyzing', 'completed', 'error')),
            is_enabled BOOLEAN DEFAULT true,
            
            -- Analysis tracking
            last_data_fetch TIMESTAMPTZ,
            last_analysis_run TIMESTAMPTZ,
            next_scheduled_analysis TIMESTAMPTZ,
            
            -- Data availability
            has_1min_data BOOLEAN DEFAULT false,
            has_5min_data BOOLEAN DEFAULT false,
            has_15min_data BOOLEAN DEFAULT false,
            has_1hour_data BOOLEAN DEFAULT false,
            has_1day_data BOOLEAN DEFAULT false,
            has_1week_data BOOLEAN DEFAULT false,
            has_1month_data BOOLEAN DEFAULT false,
            
            -- Analysis results metadata
            total_analysis_runs INTEGER DEFAULT 0,
            successful_runs INTEGER DEFAULT 0,
            failed_runs INTEGER DEFAULT 0,
            last_error_message TEXT,
            
            -- AI module specific tracking
            ppo_trader_score DECIMAL(5,4),
            portfolio_optimizer_weight DECIMAL(6,5),
            fourier_cycles_detected INTEGER,
            wavelet_patterns_found INTEGER,
            
            -- Performance tracking
            analysis_runtime_seconds DECIMAL(8,3),
            data_quality_score DECIMAL(4,3),
            volatility_score DECIMAL(5,4),
            
            -- Configuration
            fetch_priority INTEGER DEFAULT 50,  -- 1-100 scale for fetch ordering
            ai_modules_enabled TEXT[] DEFAULT ARRAY['ppo_trader', 'portfolio_optimizer', 'fourier_analyzer', 'wavelet_analyzer'],
            custom_parameters JSONB,
            
            -- Timestamps
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            
            -- Indexes for performance
            CONSTRAINT unique_symbol UNIQUE (symbol)
        );
        
        -- Create indexes for efficient queries
        CREATE INDEX idx_watchlist_priority ON watchlist_management (priority, status);
        CREATE INDEX idx_watchlist_sector ON watchlist_management (sector);
        CREATE INDEX idx_watchlist_status ON watchlist_management (status);
        CREATE INDEX idx_watchlist_last_analysis ON watchlist_management (last_analysis_run);
        CREATE INDEX idx_watchlist_fetch_priority ON watchlist_management (fetch_priority DESC);
        CREATE INDEX idx_watchlist_enabled ON watchlist_management (is_enabled, status);
        
        -- Create updated_at trigger
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        CREATE TRIGGER update_watchlist_updated_at 
            BEFORE UPDATE ON watchlist_management 
            FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)
            self.logger.info("[SUCCESS] Created watchlist_management table with indexes and triggers")
    
    async def populate_default_watchlist(self):
        """Populate the table with your 50-symbol watchlist"""
        
        insert_sql = """
        INSERT INTO watchlist_management 
        (symbol, sector, market_cap, priority, fetch_priority, custom_parameters)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (symbol) DO UPDATE SET
            sector = EXCLUDED.sector,
            market_cap = EXCLUDED.market_cap,
            priority = EXCLUDED.priority,
            fetch_priority = EXCLUDED.fetch_priority,
            custom_parameters = EXCLUDED.custom_parameters,
            updated_at = NOW()
        """
        
        async with self.pool.acquire() as conn:
            for i, stock in enumerate(self.default_watchlist):
                # Calculate fetch priority: critical=90-100, high=70-89, medium=40-69, low=10-39
                priority_map = {'critical': 95, 'high': 80, 'medium': 55, 'low': 25}
                base_priority = priority_map[stock['priority']]
                fetch_priority = base_priority + (5 - (i % 5))  # Add small variation
                
                custom_params = {
                    'market_cap': stock['market_cap'],
                    'watchlist_position': i + 1,
                    'date_added': datetime.now(EASTERN_TZ).isoformat()
                }
                
                await conn.execute(
                    insert_sql,
                    stock['symbol'],
                    stock['sector'],
                    stock['market_cap'],
                    stock['priority'],
                    fetch_priority,
                    json.dumps(custom_params)
                )
        
        self.logger.info(f"[SUCCESS] Populated watchlist with {len(self.default_watchlist)} symbols")
    
    async def get_watchlist_for_analysis(self, 
                                        priority_filter: List[str] = None,
                                        limit: int = None,
                                        status_filter: str = 'active') -> List[Dict[str, Any]]:
        """Get symbols for analysis based on priority and status"""
        
        query = """
        SELECT symbol, sector, market_cap, priority, fetch_priority,
               status, is_enabled, last_analysis_run,
               has_1min_data, has_5min_data, has_15min_data, 
               has_1hour_data, has_1day_data, has_1week_data, has_1month_data,
               ai_modules_enabled, custom_parameters
        FROM watchlist_management
        WHERE is_enabled = true 
        """
        
        params = []
        param_count = 0
        
        if status_filter:
            param_count += 1
            query += f" AND status = ${param_count}"
            params.append(status_filter)
        
        if priority_filter:
            param_count += 1
            placeholders = ','.join([f"${i}" for i in range(param_count, param_count + len(priority_filter))])
            query += f" AND priority = ANY(ARRAY[{placeholders}])"
            params.extend(priority_filter)
            param_count += len(priority_filter)
        
        query += " ORDER BY fetch_priority DESC, priority DESC, symbol ASC"
        
        if limit:
            param_count += 1
            query += f" LIMIT ${param_count}"
            params.append(limit)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            return [dict(row) for row in rows]
    
    async def update_data_availability(self, symbol: str, timeframes: Dict[str, bool]):
        """Update which timeframes have data for a symbol"""
        
        update_sql = """
        UPDATE watchlist_management 
        SET has_1min_data = $2,
            has_5min_data = $3,
            has_15min_data = $4,
            has_1hour_data = $5,
            has_1day_data = $6,
            has_1week_data = $7,
            has_1month_data = $8,
            last_data_fetch = $9
        WHERE symbol = $1
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                update_sql,
                symbol,
                timeframes.get('1 min', False),
                timeframes.get('5 mins', False),
                timeframes.get('15 mins', False),
                timeframes.get('1 hour', False),
                timeframes.get('1 day', False),
                timeframes.get('1 week', False),
                timeframes.get('1 month', False),
                datetime.now(EASTERN_TZ)
            )
    
    async def update_analysis_status(self, symbol: str, 
                                   status: str = 'completed',
                                   runtime_seconds: float = None,
                                   error_message: str = None,
                                   ai_results: Dict[str, Any] = None):
        """Update analysis status and results for a symbol"""
        
        # Prepare AI results if provided
        ppo_score = ai_results.get('ppo_trader_score') if ai_results else None
        portfolio_weight = ai_results.get('portfolio_weight') if ai_results else None
        fourier_cycles = ai_results.get('fourier_cycles') if ai_results else None
        wavelet_patterns = ai_results.get('wavelet_patterns') if ai_results else None
        
        update_sql = """
        UPDATE watchlist_management 
        SET status = $2,
            last_analysis_run = $3,
            total_analysis_runs = total_analysis_runs + 1,
            successful_runs = CASE WHEN $2 = 'completed' THEN successful_runs + 1 ELSE successful_runs END,
            failed_runs = CASE WHEN $2 = 'error' THEN failed_runs + 1 ELSE failed_runs END,
            analysis_runtime_seconds = COALESCE($4, analysis_runtime_seconds),
            last_error_message = $5,
            ppo_trader_score = COALESCE($6, ppo_trader_score),
            portfolio_optimizer_weight = COALESCE($7, portfolio_optimizer_weight),
            fourier_cycles_detected = COALESCE($8, fourier_cycles_detected),
            wavelet_patterns_found = COALESCE($9, wavelet_patterns_found)
        WHERE symbol = $1
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                update_sql,
                symbol,
                status,
                datetime.now(EASTERN_TZ),
                runtime_seconds,
                error_message,
                ppo_score,
                portfolio_weight,
                fourier_cycles,
                wavelet_patterns
            )
    
    async def get_watchlist_summary(self) -> Dict[str, Any]:
        """Get comprehensive watchlist summary"""
        
        summary_sql = """
        SELECT 
            COUNT(*) as total_symbols,
            COUNT(*) FILTER (WHERE is_enabled = true) as enabled_symbols,
            COUNT(*) FILTER (WHERE status = 'active') as active_symbols,
            COUNT(*) FILTER (WHERE priority = 'critical') as critical_priority,
            COUNT(*) FILTER (WHERE priority = 'high') as high_priority,
            COUNT(*) FILTER (WHERE priority = 'medium') as medium_priority,
            COUNT(*) FILTER (WHERE priority = 'low') as low_priority,
            COUNT(*) FILTER (WHERE has_1day_data = true) as has_daily_data,
            COUNT(*) FILTER (WHERE has_15min_data = true) as has_intraday_data,
            COUNT(*) FILTER (WHERE last_analysis_run IS NOT NULL) as analyzed_symbols,
            AVG(analysis_runtime_seconds) as avg_analysis_runtime,
            MAX(last_analysis_run) as last_analysis_time
        FROM watchlist_management
        """
        
        sector_sql = """
        SELECT sector, COUNT(*) as count,
               COUNT(*) FILTER (WHERE priority IN ('critical', 'high')) as high_priority_count
        FROM watchlist_management
        WHERE is_enabled = true
        GROUP BY sector
        ORDER BY count DESC
        """
        
        async with self.pool.acquire() as conn:
            summary_row = await conn.fetchrow(summary_sql)
            sector_rows = await conn.fetch(sector_sql)
            
            summary = dict(summary_row)
            summary['sector_breakdown'] = [dict(row) for row in sector_rows]
            
            return summary
    
    async def run_setup(self):
        """Complete watchlist setup process"""
        
        self.logger.info("=" * 80)
        self.logger.info("    WATCHLIST MANAGEMENT SYSTEM SETUP")
        self.logger.info("=" * 80)
        
        try:
            await self.connect()
            await self.create_watchlist_table()
            await self.populate_default_watchlist()
            
            # Get and display summary
            summary = await self.get_watchlist_summary()
            
            self.logger.info("")
            self.logger.info("[WATCHLIST SETUP COMPLETE]")
            self.logger.info(f"  Total Symbols: {summary['total_symbols']}")
            self.logger.info(f"  Enabled Symbols: {summary['enabled_symbols']}")
            self.logger.info(f"  Critical Priority: {summary['critical_priority']}")
            self.logger.info(f"  High Priority: {summary['high_priority']}")
            self.logger.info(f"  Medium Priority: {summary['medium_priority']}")
            self.logger.info(f"  Low Priority: {summary['low_priority']}")
            
            self.logger.info("")
            self.logger.info("[SECTOR BREAKDOWN]")
            for sector in summary['sector_breakdown']:
                self.logger.info(f"  {sector['sector']:>12}: {sector['count']:>2} symbols "
                               f"({sector['high_priority_count']} high priority)")
            
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("    WATCHLIST READY FOR AI ANALYSIS PIPELINE")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Watchlist setup failed: {e}")
            raise
        finally:
            await self.close()

# Usage example and main execution
async def main():
    """Set up the watchlist management system"""
    manager = WatchlistManager()
    await manager.run_setup()
    
    return manager.default_watchlist

if __name__ == "__main__":
    watchlist = asyncio.run(main())
