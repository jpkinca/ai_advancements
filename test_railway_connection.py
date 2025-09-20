#!/usr/bin/env python3
"""
Railway PostgreSQL Connection Test

Simple test to verify Railway PostgreSQL connection works correctly
before running the full critical priority test.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add TradeAppComponents_fresh to path for Railway database manager
sys.path.append(r'c:\Users\nzcon\VSPython\TradeAppComponents_fresh')
from modules.database.railway_db_manager import RailwayPostgreSQLManager

# Configure logging for ASCII-only output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

async def test_railway_connection():
    """Test Railway PostgreSQL connection and basic operations"""
    
    print("=== Railway PostgreSQL Connection Test ===")
    print(f"[STARTING] Test at {datetime.now()}")
    
    try:
        # Step 1: Initialize Railway Manager
        print("\n[STEP 1] Initializing Railway PostgreSQL Manager...")
        railway_manager = RailwayPostgreSQLManager()
        
        # Step 2: Get connection info
        print("\n[STEP 2] Getting connection information...")
        conn_info = railway_manager.get_connection_info()
        print(f"[SUCCESS] Connected to: {conn_info['host']}:{conn_info['port']}")
        print(f"[SUCCESS] Database: {conn_info['database']}")
        print(f"[SUCCESS] Is Railway: {conn_info['is_railway']}")
        print(f"[SUCCESS] SSL Required: {conn_info['ssl_required']}")
        
        # Step 3: Health check
        print("\n[STEP 3] Running health check...")
        health_status = await railway_manager.health_check()
        print(f"[SUCCESS] Health Status: {health_status['status']}")
        print(f"[SUCCESS] Connection Time: {health_status['tests']['connection']['response_time_seconds']:.3f}s")
        
        # Step 4: Test basic SQL operations
        print("\n[STEP 4] Testing basic SQL operations...")
        session = railway_manager.get_session()
        
        try:
            # Test a simple query
            result = session.execute("SELECT current_database(), current_user, version()")
            row = result.fetchone()
            print(f"[SUCCESS] Database: {row[0]}")
            print(f"[SUCCESS] User: {row[1]}")
            print(f"[SUCCESS] PostgreSQL Version: {row[2][:50]}...")
            
            # Test table creation
            session.execute("""
                CREATE TABLE IF NOT EXISTS test_connection (
                    id SERIAL PRIMARY KEY,
                    test_time TIMESTAMPTZ DEFAULT NOW(),
                    message TEXT
                )
            """)
            session.commit()
            print("[SUCCESS] Test table created successfully")
            
            # Test insert
            session.execute("""
                INSERT INTO test_connection (message) 
                VALUES ('Railway connection test successful')
            """)
            session.commit()
            print("[SUCCESS] Test data inserted successfully")
            
            # Test select
            result = session.execute("SELECT COUNT(*) FROM test_connection")
            count = result.fetchone()[0]
            print(f"[SUCCESS] Test table has {count} records")
            
            # Cleanup
            session.execute("DROP TABLE IF EXISTS test_connection")
            session.commit()
            print("[SUCCESS] Test table cleaned up")
            
        finally:
            session.close()
        
        print("\n=== Test Summary ===")
        print("[SUCCESS] All Railway PostgreSQL tests passed!")
        print("[SUCCESS] Connection is working correctly")
        print("[SUCCESS] Ready for AI trading system integration")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Railway PostgreSQL test failed: {e}")
        print("[ERROR] Check Railway connection settings")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_railway_connection())
    exit(0 if success else 1)
