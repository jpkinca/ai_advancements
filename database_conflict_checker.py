"""
Database Table Conflict Checker

This script checks the existing Railway PostgreSQL database for table conflicts
before creating new AI trading tables.

It will:
1. Connect to Railway PostgreSQL
2. List all existing tables
3. Check for potential conflicts with our new tables:
   - watchlist_management
   - historical_market_data (from multi_timeframe_data_manager)
4. Provide recommendations for avoiding conflicts
"""

import asyncio
import asyncpg
import logging
from typing import List, Dict, Any
import sys
import os

# Add TradeAppComponents_fresh to path
sys.path.append(r'c:\Users\nzcon\VSPython\TradeAppComponents_fresh')
from modules.database.railway_db_manager import RailwayPostgreSQLManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class DatabaseConflictChecker:
    """Check for table conflicts in Railway PostgreSQL database"""
    
    def __init__(self):
        self.logger = logger
        self.railway_manager = RailwayPostgreSQLManager()
        self.database_url = self.railway_manager.database_url
        
        # Tables we plan to create for AI trading
        self.planned_tables = {
            'watchlist_management': {
                'purpose': 'Track 50 stocks through AI analysis pipeline',
                'columns': ['id', 'symbol', 'sector', 'priority', 'status', 'last_analysis_run'],
                'indexes': ['idx_watchlist_priority', 'idx_watchlist_sector', 'idx_watchlist_status']
            },
            'historical_market_data': {
                'purpose': 'Store multi-timeframe OHLCV data from IBKR',
                'columns': ['id', 'symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume'],
                'indexes': ['idx_historical_symbol_timeframe', 'idx_historical_timestamp']
            }
        }
    
    async def connect_to_database(self) -> asyncpg.Connection:
        """Connect to Railway PostgreSQL database"""
        try:
            # Add SSL requirement for Railway
            if '?sslmode=' not in self.database_url:
                self.database_url += '?sslmode=require'
            
            conn = await asyncpg.connect(self.database_url)
            self.logger.info("[SUCCESS] Connected to Railway PostgreSQL database")
            return conn
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to connect to Railway database: {e}")
            raise
    
    async def get_existing_tables(self, conn: asyncpg.Connection) -> List[Dict[str, Any]]:
        """Get list of all existing tables in the database"""
        
        query = """
        SELECT 
            schemaname,
            tablename,
            tableowner,
            tablespace,
            hasindexes,
            hasrules,
            hastriggers
        FROM pg_tables 
        WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
        ORDER BY schemaname, tablename;
        """
        
        try:
            rows = await conn.fetch(query)
            tables = [dict(row) for row in rows]
            self.logger.info(f"[SUCCESS] Found {len(tables)} existing tables")
            return tables
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to get existing tables: {e}")
            return []
    
    async def get_table_details(self, conn: asyncpg.Connection, table_name: str, schema_name: str = 'public') -> Dict[str, Any]:
        """Get detailed information about a specific table"""
        
        # Get column information
        columns_query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = $1 AND table_schema = $2
        ORDER BY ordinal_position;
        """
        
        # Get index information
        indexes_query = """
        SELECT 
            indexname,
            indexdef
        FROM pg_indexes 
        WHERE tablename = $1 AND schemaname = $2;
        """
        
        try:
            columns = await conn.fetch(columns_query, table_name, schema_name)
            indexes = await conn.fetch(indexes_query, table_name, schema_name)
            
            return {
                'columns': [dict(row) for row in columns],
                'indexes': [dict(row) for row in indexes],
                'column_count': len(columns),
                'index_count': len(indexes)
            }
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to get details for table {table_name}: {e}")
            return {'columns': [], 'indexes': [], 'column_count': 0, 'index_count': 0}
    
    async def check_for_conflicts(self, existing_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if our planned tables conflict with existing ones"""
        
        conflicts = {}
        existing_table_names = [table['tablename'] for table in existing_tables]
        
        for planned_table, details in self.planned_tables.items():
            if planned_table in existing_table_names:
                conflicts[planned_table] = {
                    'conflict_type': 'table_name_exists',
                    'severity': 'high',
                    'recommendation': f'Table {planned_table} already exists - need to check structure or use different name'
                }
            else:
                conflicts[planned_table] = {
                    'conflict_type': 'none',
                    'severity': 'none',
                    'recommendation': f'Safe to create {planned_table}'
                }
        
        return conflicts
    
    async def analyze_existing_schema(self, conn: asyncpg.Connection, existing_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the existing database schema for context"""
        
        schema_analysis = {
            'total_tables': len(existing_tables),
            'schemas': {},
            'potential_trading_tables': [],
            'large_tables': []
        }
        
        # Group by schema
        for table in existing_tables:
            schema = table['schemaname']
            if schema not in schema_analysis['schemas']:
                schema_analysis['schemas'][schema] = []
            schema_analysis['schemas'][schema].append(table['tablename'])
        
        # Look for existing trading-related tables
        trading_keywords = ['trade', 'stock', 'market', 'price', 'ohlc', 'portfolio', 'order', 'position']
        for table in existing_tables:
            table_name = table['tablename'].lower()
            if any(keyword in table_name for keyword in trading_keywords):
                schema_analysis['potential_trading_tables'].append({
                    'name': table['tablename'],
                    'schema': table['schemaname']
                })
        
        return schema_analysis
    
    def display_results(self, existing_tables: List[Dict[str, Any]], conflicts: Dict[str, Any], schema_analysis: Dict[str, Any]):
        """Display comprehensive analysis results"""
        
        self.logger.info("=" * 80)
        self.logger.info("    DATABASE CONFLICT ANALYSIS RESULTS")
        self.logger.info("=" * 80)
        
        # Existing tables summary
        self.logger.info("")
        self.logger.info("[EXISTING DATABASE STRUCTURE]")
        self.logger.info(f"  Total Tables: {schema_analysis['total_tables']}")
        
        for schema, tables in schema_analysis['schemas'].items():
            self.logger.info(f"  Schema '{schema}': {len(tables)} tables")
            # Show first few table names
            if tables:
                sample_tables = ', '.join(tables[:5])
                if len(tables) > 5:
                    sample_tables += f", ... (+{len(tables) - 5} more)"
                self.logger.info(f"    Tables: {sample_tables}")
        
        # Potential trading tables
        if schema_analysis['potential_trading_tables']:
            self.logger.info("")
            self.logger.info("[EXISTING TRADING-RELATED TABLES]")
            for table in schema_analysis['potential_trading_tables']:
                self.logger.info(f"  {table['schema']}.{table['name']}")
        else:
            self.logger.info("")
            self.logger.info("[EXISTING TRADING-RELATED TABLES]")
            self.logger.info("  None found - safe to create new trading tables")
        
        # Conflict analysis
        self.logger.info("")
        self.logger.info("[CONFLICT ANALYSIS FOR PLANNED TABLES]")
        
        safe_tables = []
        conflict_tables = []
        
        for table_name, conflict_info in conflicts.items():
            purpose = self.planned_tables[table_name]['purpose']
            
            if conflict_info['conflict_type'] == 'none':
                safe_tables.append(table_name)
                self.logger.info(f"  [SAFE] {table_name}: {purpose}")
            else:
                conflict_tables.append(table_name)
                self.logger.info(f"  [CONFLICT] {table_name}: {conflict_info['recommendation']}")
        
        # Summary
        self.logger.info("")
        self.logger.info("[SUMMARY]")
        self.logger.info(f"  Safe to create: {len(safe_tables)} tables")
        self.logger.info(f"  Conflicts found: {len(conflict_tables)} tables")
        
        if safe_tables:
            self.logger.info(f"  Safe tables: {', '.join(safe_tables)}")
        
        if conflict_tables:
            self.logger.info(f"  Conflict tables: {', '.join(conflict_tables)}")
            self.logger.info("")
            self.logger.info("[RECOMMENDATIONS]")
            for table in conflict_tables:
                conflict_info = conflicts[table]
                self.logger.info(f"  {table}: {conflict_info['recommendation']}")
        
        # Next steps
        self.logger.info("")
        self.logger.info("[NEXT STEPS]")
        if not conflict_tables:
            self.logger.info("  [SUCCESS] No conflicts found - safe to proceed with table creation")
            self.logger.info("  1. Run watchlist_manager.py to create watchlist_management table")
            self.logger.info("  2. Run data manager to create historical_market_data table")
            self.logger.info("  3. Start AI analysis pipeline testing")
        else:
            self.logger.info("  [ACTION NEEDED] Resolve table conflicts before proceeding")
            self.logger.info("  1. Check existing table structures")
            self.logger.info("  2. Consider renaming new tables (e.g., ai_watchlist_management)")
            self.logger.info("  3. Or drop existing conflicting tables if safe to do so")
        
        self.logger.info("")
        self.logger.info("=" * 80)
    
    async def run_analysis(self):
        """Run complete database conflict analysis"""
        
        self.logger.info("Starting database conflict analysis for AI trading tables...")
        
        try:
            # Connect to database
            conn = await self.connect_to_database()
            
            # Get existing tables
            existing_tables = await self.get_existing_tables(conn)
            
            # Check for conflicts
            conflicts = await self.check_for_conflicts(existing_tables)
            
            # Analyze schema
            schema_analysis = await self.analyze_existing_schema(conn, existing_tables)
            
            # If we found conflicts, get detailed info
            conflict_details = {}
            for table_name, conflict_info in conflicts.items():
                if conflict_info['conflict_type'] != 'none':
                    details = await self.get_table_details(conn, table_name)
                    conflict_details[table_name] = details
            
            # Display results
            self.display_results(existing_tables, conflicts, schema_analysis)
            
            # Show conflict details if any
            if conflict_details:
                self.logger.info("")
                self.logger.info("[EXISTING TABLE DETAILS]")
                for table_name, details in conflict_details.items():
                    self.logger.info(f"  Table: {table_name}")
                    self.logger.info(f"    Columns: {details['column_count']}")
                    self.logger.info(f"    Indexes: {details['index_count']}")
                    if details['columns']:
                        column_names = [col['column_name'] for col in details['columns'][:5]]
                        self.logger.info(f"    Sample columns: {', '.join(column_names)}")
            
            await conn.close()
            
        except Exception as e:
            self.logger.error(f"[ERROR] Analysis failed: {e}")
            raise

async def main():
    """Run the database conflict analysis"""
    
    checker = DatabaseConflictChecker()
    await checker.run_analysis()

if __name__ == "__main__":
    asyncio.run(main())
