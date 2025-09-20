"""
Simple Database Table Conflict Checker for Railway PostgreSQL

Direct connection to Railway PostgreSQL to check for table conflicts
before creating watchlist_management table.
"""

import asyncio
import asyncpg
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class SimpleConflictChecker:
    """Simple checker for Railway PostgreSQL table conflicts"""
    
    def __init__(self):
        # Direct Railway PostgreSQL connection
        self.database_url = "postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway"
        
        # Tables we plan to create
        self.planned_tables = ['watchlist_management', 'historical_market_data']
    
    async def connect_to_database(self) -> asyncpg.Connection:
        """Connect to Railway PostgreSQL database"""
        try:
            conn = await asyncpg.connect(self.database_url)
            logger.info("[SUCCESS] Connected to Railway PostgreSQL database")
            return conn
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to Railway database: {e}")
            raise
    
    async def get_all_tables(self, conn: asyncpg.Connection) -> List[str]:
        """Get list of all table names in public schema"""
        
        query = """
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY tablename;
        """
        
        try:
            rows = await conn.fetch(query)
            tables = [row['tablename'] for row in rows]
            logger.info(f"[SUCCESS] Found {len(tables)} existing tables in public schema")
            return tables
        except Exception as e:
            logger.error(f"[ERROR] Failed to get table list: {e}")
            return []
    
    async def check_table_conflicts(self, existing_tables: List[str]) -> Dict[str, bool]:
        """Check if our planned tables conflict with existing ones"""
        
        conflicts = {}
        for planned_table in self.planned_tables:
            conflicts[planned_table] = planned_table in existing_tables
        
        return conflicts
    
    async def get_table_info(self, conn: asyncpg.Connection, table_name: str) -> Dict[str, Any]:
        """Get basic info about an existing table"""
        
        column_query = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = $1 AND table_schema = 'public'
        ORDER BY ordinal_position;
        """
        
        index_query = """
        SELECT indexname 
        FROM pg_indexes 
        WHERE tablename = $1 AND schemaname = 'public';
        """
        
        try:
            columns = await conn.fetch(column_query, table_name)
            indexes = await conn.fetch(index_query, table_name)
            
            return {
                'columns': [f"{row['column_name']} ({row['data_type']})" for row in columns],
                'indexes': [row['indexname'] for row in indexes],
                'column_count': len(columns),
                'index_count': len(indexes)
            }
        except Exception as e:
            logger.error(f"[ERROR] Failed to get info for table {table_name}: {e}")
            return {'columns': [], 'indexes': [], 'column_count': 0, 'index_count': 0}
    
    def display_results(self, existing_tables: List[str], conflicts: Dict[str, bool]):
        """Display analysis results"""
        
        logger.info("=" * 70)
        logger.info("    RAILWAY POSTGRESQL TABLE CONFLICT ANALYSIS")
        logger.info("=" * 70)
        
        # Show existing tables
        logger.info("")
        logger.info(f"[EXISTING TABLES] Found {len(existing_tables)} tables in public schema:")
        if existing_tables:
            for table in existing_tables:
                logger.info(f"  - {table}")
        else:
            logger.info("  No existing tables found")
        
        # Show conflict analysis
        logger.info("")
        logger.info("[CONFLICT ANALYSIS]")
        
        safe_tables = []
        conflict_tables = []
        
        for table_name, has_conflict in conflicts.items():
            if has_conflict:
                conflict_tables.append(table_name)
                logger.info(f"  [CONFLICT] {table_name} - Table already exists!")
            else:
                safe_tables.append(table_name)
                logger.info(f"  [SAFE] {table_name} - No conflict, safe to create")
        
        # Summary and recommendations
        logger.info("")
        logger.info("[SUMMARY]")
        logger.info(f"  Safe to create: {len(safe_tables)} tables")
        logger.info(f"  Conflicts found: {len(conflict_tables)} tables")
        
        if safe_tables:
            logger.info(f"  Safe tables: {', '.join(safe_tables)}")
        
        if conflict_tables:
            logger.info(f"  Conflict tables: {', '.join(conflict_tables)}")
            logger.info("")
            logger.info("[RECOMMENDATIONS FOR CONFLICTS]")
            for table in conflict_tables:
                logger.info(f"  {table}:")
                logger.info(f"    1. Check existing table structure")
                logger.info(f"    2. Consider using different name (e.g., ai_{table})")
                logger.info(f"    3. Drop existing table if safe to do so")
        
        # Next steps
        logger.info("")
        logger.info("[NEXT STEPS]")
        if not conflict_tables:
            logger.info("  [SUCCESS] No conflicts found - proceed with table creation!")
            logger.info("  1. Run watchlist_manager.py to create watchlist_management table")
            logger.info("  2. Start AI testing with critical_priority_test.py")
        else:
            logger.info("  [ACTION NEEDED] Resolve conflicts before proceeding")
            logger.info("  1. Examine existing conflicting tables")
            logger.info("  2. Decide whether to rename or drop existing tables")
        
        logger.info("")
        logger.info("=" * 70)
    
    async def run_analysis(self):
        """Run the complete conflict analysis"""
        
        logger.info("Starting Railway PostgreSQL table conflict check...")
        
        try:
            # Connect to database
            conn = await self.connect_to_database()
            
            # Get existing tables
            existing_tables = await self.get_all_tables(conn)
            
            # Check for conflicts
            conflicts = await self.check_table_conflicts(existing_tables)
            
            # Show details for conflicting tables
            conflict_details = {}
            for table_name, has_conflict in conflicts.items():
                if has_conflict:
                    logger.info(f"")
                    logger.info(f"[EXISTING TABLE DETAILS] {table_name}:")
                    details = await self.get_table_info(conn, table_name)
                    logger.info(f"  Columns ({details['column_count']}): {', '.join(details['columns'][:3])}...")
                    logger.info(f"  Indexes ({details['index_count']}): {', '.join(details['indexes'])}")
                    conflict_details[table_name] = details
            
            # Display main results
            self.display_results(existing_tables, conflicts)
            
            await conn.close()
            
            return {
                'existing_tables': existing_tables,
                'conflicts': conflicts,
                'conflict_details': conflict_details,
                'safe_to_proceed': not any(conflicts.values())
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Analysis failed: {e}")
            raise

async def main():
    """Run the conflict check"""
    
    checker = SimpleConflictChecker()
    result = await checker.run_analysis()
    
    # Return exit code based on conflicts
    if result['safe_to_proceed']:
        logger.info("[FINAL RESULT] Safe to proceed with table creation")
        return 0
    else:
        logger.info("[FINAL RESULT] Conflicts found - manual intervention needed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
