"""
Examine Existing historical_market_data Table Structure

This script will:
1. Connect to Railway PostgreSQL
2. Examine the existing historical_market_data table structure
3. Compare it with our AI module requirements
4. Recommend whether to use existing table or create new one
"""

import asyncio
import asyncpg
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class TableStructureAnalyzer:
    """Analyze existing table structure for AI compatibility"""
    
    def __init__(self):
        self.database_url = "postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway"
        
        # Required columns for our AI modules
        self.ai_requirements = {
            'symbol': {'type': 'text', 'nullable': False, 'purpose': 'Stock symbol (AAPL, NVDA, etc.)'},
            'timeframe': {'type': 'text', 'nullable': False, 'purpose': 'Timeframe (1min, 5min, 1hour, etc.)'},
            'timestamp': {'type': 'timestamp', 'nullable': False, 'purpose': 'OHLC bar timestamp'},
            'open': {'type': 'numeric/decimal', 'nullable': False, 'purpose': 'Opening price'},
            'high': {'type': 'numeric/decimal', 'nullable': False, 'purpose': 'High price'},
            'low': {'type': 'numeric/decimal', 'nullable': False, 'purpose': 'Low price'},
            'close': {'type': 'numeric/decimal', 'nullable': False, 'purpose': 'Closing price'},
            'volume': {'type': 'bigint/numeric', 'nullable': False, 'purpose': 'Trading volume'},
            'created_at': {'type': 'timestamp', 'nullable': True, 'purpose': 'Record creation time'},
            'source': {'type': 'text', 'nullable': True, 'purpose': 'Data source (IBKR, etc.)'}
        }
    
    async def connect_to_database(self) -> asyncpg.Connection:
        """Connect to Railway PostgreSQL database"""
        try:
            conn = await asyncpg.connect(self.database_url)
            logger.info("[SUCCESS] Connected to Railway PostgreSQL database")
            return conn
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect: {e}")
            raise
    
    async def get_table_structure(self, conn: asyncpg.Connection, table_name: str) -> Dict[str, Any]:
        """Get detailed table structure"""
        
        # Get column information
        columns_query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale
        FROM information_schema.columns 
        WHERE table_name = $1 AND table_schema = 'public'
        ORDER BY ordinal_position;
        """
        
        # Get constraints
        constraints_query = """
        SELECT 
            tc.constraint_name,
            tc.constraint_type,
            kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = $1 AND tc.table_schema = 'public';
        """
        
        # Get indexes
        indexes_query = """
        SELECT 
            indexname,
            indexdef
        FROM pg_indexes 
        WHERE tablename = $1 AND schemaname = 'public';
        """
        
        # Get sample data
        sample_query = f"""
        SELECT * FROM {table_name} 
        ORDER BY timestamp DESC 
        LIMIT 5;
        """
        
        try:
            columns = await conn.fetch(columns_query, table_name)
            constraints = await conn.fetch(constraints_query, table_name)
            indexes = await conn.fetch(indexes_query, table_name)
            
            # Try to get sample data
            try:
                sample_data = await conn.fetch(sample_query)
            except Exception:
                sample_data = []
            
            return {
                'columns': [dict(row) for row in columns],
                'constraints': [dict(row) for row in constraints],
                'indexes': [dict(row) for row in indexes],
                'sample_data': [dict(row) for row in sample_data],
                'column_count': len(columns),
                'has_data': len(sample_data) > 0
            }
        except Exception as e:
            logger.error(f"[ERROR] Failed to analyze table {table_name}: {e}")
            return {
                'columns': [], 'constraints': [], 'indexes': [], 
                'sample_data': [], 'column_count': 0, 'has_data': False
            }
    
    def analyze_compatibility(self, table_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if existing table is compatible with AI requirements"""
        
        existing_columns = {col['column_name']: col for col in table_structure['columns']}
        
        compatibility = {
            'compatible': True,
            'missing_columns': [],
            'incompatible_columns': [],
            'compatible_columns': [],
            'recommendations': []
        }
        
        # Check each required column
        for req_col, req_specs in self.ai_requirements.items():
            if req_col not in existing_columns:
                if req_specs['nullable'] == False:
                    compatibility['missing_columns'].append({
                        'column': req_col,
                        'required_type': req_specs['type'],
                        'purpose': req_specs['purpose'],
                        'severity': 'high' if not req_specs['nullable'] else 'medium'
                    })
                    compatibility['compatible'] = False
            else:
                existing_col = existing_columns[req_col]
                existing_type = existing_col['data_type'].lower()
                required_type = req_specs['type'].lower()
                
                # Check type compatibility
                type_compatible = self._check_type_compatibility(existing_type, required_type)
                
                if type_compatible:
                    compatibility['compatible_columns'].append({
                        'column': req_col,
                        'existing_type': existing_type,
                        'status': 'compatible'
                    })
                else:
                    compatibility['incompatible_columns'].append({
                        'column': req_col,
                        'existing_type': existing_type,
                        'required_type': required_type,
                        'severity': 'high'
                    })
                    compatibility['compatible'] = False
        
        # Generate recommendations
        if compatibility['compatible']:
            compatibility['recommendations'].append("Existing table is compatible - can be used for AI modules")
        else:
            if compatibility['missing_columns']:
                compatibility['recommendations'].append("Add missing columns to existing table")
            if compatibility['incompatible_columns']:
                compatibility['recommendations'].append("Create separate AI-specific table (ai_historical_market_data)")
        
        return compatibility
    
    def _check_type_compatibility(self, existing_type: str, required_type: str) -> bool:
        """Check if data types are compatible"""
        
        type_mappings = {
            'text': ['text', 'varchar', 'character varying'],
            'numeric': ['numeric', 'decimal', 'double precision', 'real'],
            'bigint': ['bigint', 'integer', 'numeric'],
            'timestamp': ['timestamp', 'timestamp with time zone', 'timestamp without time zone']
        }
        
        for req_type, compatible_types in type_mappings.items():
            if req_type in required_type:
                return existing_type in compatible_types
        
        return existing_type == required_type
    
    def display_analysis(self, table_structure: Dict[str, Any], compatibility: Dict[str, Any]):
        """Display comprehensive analysis results"""
        
        logger.info("=" * 80)
        logger.info("    EXISTING TABLE STRUCTURE ANALYSIS")
        logger.info("=" * 80)
        
        # Table overview
        logger.info("")
        logger.info("[TABLE OVERVIEW] historical_market_data")
        logger.info(f"  Columns: {table_structure['column_count']}")
        logger.info(f"  Has Data: {table_structure['has_data']}")
        logger.info(f"  Constraints: {len(table_structure['constraints'])}")
        logger.info(f"  Indexes: {len(table_structure['indexes'])}")
        
        # Column details
        logger.info("")
        logger.info("[EXISTING COLUMNS]")
        for col in table_structure['columns']:
            nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
            logger.info(f"  {col['column_name']}: {col['data_type']} {nullable}")
        
        # Sample data
        if table_structure['sample_data']:
            logger.info("")
            logger.info("[SAMPLE DATA] (Latest 2 records)")
            for i, row in enumerate(table_structure['sample_data'][:2]):
                logger.info(f"  Record {i+1}:")
                for key, value in row.items():
                    logger.info(f"    {key}: {value}")
        
        # Compatibility analysis
        logger.info("")
        logger.info("[AI COMPATIBILITY ANALYSIS]")
        logger.info(f"  Overall Compatible: {compatibility['compatible']}")
        
        if compatibility['compatible_columns']:
            logger.info("")
            logger.info("  [COMPATIBLE COLUMNS]")
            for col in compatibility['compatible_columns']:
                logger.info(f"    ✓ {col['column']}: {col['existing_type']}")
        
        if compatibility['missing_columns']:
            logger.info("")
            logger.info("  [MISSING COLUMNS]")
            for col in compatibility['missing_columns']:
                logger.info(f"    ✗ {col['column']}: {col['required_type']} - {col['purpose']}")
        
        if compatibility['incompatible_columns']:
            logger.info("")
            logger.info("  [INCOMPATIBLE COLUMNS]")
            for col in compatibility['incompatible_columns']:
                logger.info(f"    ⚠ {col['column']}: {col['existing_type']} -> needs {col['required_type']}")
        
        # Recommendations
        logger.info("")
        logger.info("[RECOMMENDATIONS]")
        for i, rec in enumerate(compatibility['recommendations'], 1):
            logger.info(f"  {i}. {rec}")
        
        # Decision
        logger.info("")
        logger.info("[DECISION]")
        if compatibility['compatible']:
            logger.info("  [SUCCESS] Use existing historical_market_data table")
            logger.info("  Action: Update multi_timeframe_data_manager.py to use existing table")
        else:
            logger.info("  [CREATE NEW] Create ai_historical_market_data table")
            logger.info("  Action: Create separate table optimized for AI modules")
        
        logger.info("")
        logger.info("=" * 80)
    
    async def run_analysis(self):
        """Run complete table structure analysis"""
        
        logger.info("Analyzing existing historical_market_data table for AI compatibility...")
        
        try:
            conn = await self.connect_to_database()
            
            # Get table structure
            table_structure = await self.get_table_structure(conn, 'historical_market_data')
            
            if table_structure['column_count'] == 0:
                logger.error("[ERROR] Table historical_market_data not found or has no columns")
                return False
            
            # Analyze compatibility
            compatibility = self.analyze_compatibility(table_structure)
            
            # Display results
            self.display_analysis(table_structure, compatibility)
            
            await conn.close()
            
            return compatibility['compatible']
            
        except Exception as e:
            logger.error(f"[ERROR] Analysis failed: {e}")
            return False

async def main():
    """Run the table analysis"""
    
    analyzer = TableStructureAnalyzer()
    is_compatible = await analyzer.run_analysis()
    
    if is_compatible:
        logger.info("[FINAL RESULT] Existing table is suitable for AI modules")
        return 0
    else:
        logger.info("[FINAL RESULT] Need to create separate AI table")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
