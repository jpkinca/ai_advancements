"""
Simple Database Manager for AI Trading System
Production-ready database connectivity with PostgreSQL and SQLite fallback
"""

import os
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class SimpleDBManager:
    """Simplified database manager with SQLite fallback"""
    
    def __init__(self, use_sqlite_fallback=True):
        self.use_sqlite_fallback = use_sqlite_fallback
        
        if use_sqlite_fallback:
            self.db_path = Path(__file__).parent / "trading_data.db"
            self.connection_string = str(self.db_path)
            logger.info(f"Using SQLite fallback: {self.db_path}")
        else:
            # Railway PostgreSQL (requires environment variables)
            self.connection_string = self.get_postgresql_url()
    
    def get_postgresql_url(self):
        """Get PostgreSQL connection string from environment"""
        # Check for Railway environment variables
        railway_vars = {
            'PGHOST': os.getenv('PGHOST'),
            'PGPORT': os.getenv('PGPORT', '5432'),
            'PGDATABASE': os.getenv('PGDATABASE'),
            'PGUSER': os.getenv('PGUSER'),
            'PGPASSWORD': os.getenv('PGPASSWORD')
        }
        
        if all(railway_vars.values()):
            return f"postgresql://{railway_vars['PGUSER']}:{railway_vars['PGPASSWORD']}@{railway_vars['PGHOST']}:{railway_vars['PGPORT']}/{railway_vars['PGDATABASE']}"
        else:
            logger.warning("PostgreSQL environment variables not found, using SQLite fallback")
            self.use_sqlite_fallback = True
            self.db_path = Path(__file__).parent / "trading_data.db"
            return str(self.db_path)
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        if self.use_sqlite_fallback:
            conn = sqlite3.connect(self.connection_string)
            conn.row_factory = sqlite3.Row  # Enable column access by name
        else:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            conn = psycopg2.connect(self.connection_string, cursor_factory=RealDictCursor)
        
        try:
            yield conn
        finally:
            conn.close()
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test database connectivity"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if self.use_sqlite_fallback:
                    cursor.execute("SELECT sqlite_version()")
                    version = cursor.fetchone()[0]
                    return True, f"SQLite v{version}"
                else:
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()['version']
                    return True, f"PostgreSQL: {version[:50]}..."
        except Exception as e:
            return False, str(e)
    
    def initialize_tables(self):
        """Create basic tables for trading data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Basic market data table
                if self.use_sqlite_fallback:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS market_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT NOT NULL,
                            timestamp DATETIME NOT NULL,
                            price REAL NOT NULL,
                            volume INTEGER,
                            data_type TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Level II data table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS level_ii_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT NOT NULL,
                            timestamp DATETIME NOT NULL,
                            side TEXT NOT NULL,
                            price REAL NOT NULL,
                            size INTEGER NOT NULL,
                            exchange TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Trading signals table  
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS trading_signals (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT NOT NULL,
                            timestamp DATETIME NOT NULL,
                            signal_type TEXT NOT NULL,
                            confidence REAL,
                            price REAL,
                            strategy TEXT,
                            metadata TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                
                conn.commit()
                logger.info("Database tables initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Table initialization failed: {e}")
            return False


# Factory function for easy import
def get_db_manager(use_sqlite_fallback=True):
    """Factory function to get database manager instance"""
    return SimpleDBManager(use_sqlite_fallback=use_sqlite_fallback)


# Test connection on import
if __name__ == "__main__":
    manager = SimpleDBManager()
    success, message = manager.test_connection()
    print(f"Database connection test: {'Success' if success else 'Failed'} {message}")
    
    if success:
        table_init = manager.initialize_tables()
        print(f"Table initialization: {'Success' if table_init else 'Failed'}")
