"""
PostgreSQL Connection Test

This script tests different PostgreSQL connection configurations to help
identify the correct connection parameters.
"""

import asyncio
import asyncpg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_connection_variants():
    """Test different PostgreSQL connection configurations"""
    
    logger.info("=" * 60)
    logger.info("    PostgreSQL Connection Test")
    logger.info("=" * 60)
    
    # Different connection options to try
    connection_variants = [
        {
            'name': 'Local with provided password',
            'url': 'postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@localhost:5432/ai_trading'
        },
        {
            'name': 'Local with provided password (postgres db)',
            'url': 'postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@localhost:5432/postgres'
        },
        {
            'name': 'Local default',
            'url': 'postgresql://postgres@localhost:5432/ai_trading'
        },
        {
            'name': 'Railway (if remote)',
            'url': 'postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@roundhouse.proxy.rlwy.net:5432/railway'
        }
    ]
    
    successful_connections = []
    
    for variant in connection_variants:
        logger.info(f"\n[TESTING] {variant['name']}")
        logger.info(f"  URL: {variant['url'][:50]}...")
        
        try:
            conn = await asyncpg.connect(variant['url'])
            
            # Test basic query
            result = await conn.fetchval("SELECT version()")
            logger.info(f"  [SUCCESS] Connected!")
            logger.info(f"  PostgreSQL Version: {result[:50]}...")
            
            # Test database listing
            databases = await conn.fetch("SELECT datname FROM pg_database WHERE datistemplate = false")
            db_names = [db['datname'] for db in databases]
            logger.info(f"  Available databases: {', '.join(db_names)}")
            
            await conn.close()
            successful_connections.append(variant)
            
        except Exception as e:
            logger.error(f"  [FAILED] {type(e).__name__}: {e}")
    
    logger.info(f"\n" + "=" * 60)
    logger.info(f"[SUMMARY] {len(successful_connections)} out of {len(connection_variants)} connections succeeded")
    
    if successful_connections:
        logger.info("[SUCCESSFUL CONNECTIONS]")
        for conn in successful_connections:
            logger.info(f"  ✓ {conn['name']}")
            logger.info(f"    URL: {conn['url']}")
        
        # Return the first successful connection for use
        return successful_connections[0]['url']
    else:
        logger.error("[ERROR] No successful connections found")
        logger.info("[TROUBLESHOOTING SUGGESTIONS]")
        logger.info("  1. Check if PostgreSQL is running: 'pg_ctl status'")
        logger.info("  2. Verify connection parameters (host, port, username, password)")
        logger.info("  3. Check if firewall is blocking connection")
        logger.info("  4. Verify database exists: 'psql -l'")
        logger.info("  5. Try connecting with psql command line first")
        return None

async def test_database_operations(connection_url: str):
    """Test basic database operations with the working connection"""
    
    logger.info(f"\n[TESTING DATABASE OPERATIONS]")
    
    try:
        conn = await asyncpg.connect(connection_url)
        
        # Test schema creation
        await conn.execute("CREATE SCHEMA IF NOT EXISTS test_schema")
        logger.info("  ✓ Schema creation works")
        
        # Test table creation
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_schema.test_table (
                id SERIAL PRIMARY KEY,
                name VARCHAR(50),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        logger.info("  ✓ Table creation works")
        
        # Test data insertion
        await conn.execute(
            "INSERT INTO test_schema.test_table (name) VALUES ($1)",
            "test_entry"
        )
        logger.info("  ✓ Data insertion works")
        
        # Test data retrieval
        result = await conn.fetchrow("SELECT * FROM test_schema.test_table LIMIT 1")
        logger.info(f"  ✓ Data retrieval works: {dict(result)}")
        
        # Clean up
        await conn.execute("DROP TABLE test_schema.test_table")
        await conn.execute("DROP SCHEMA test_schema")
        logger.info("  ✓ Cleanup completed")
        
        await conn.close()
        logger.info("[SUCCESS] All database operations working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Database operations failed: {e}")
        return False

async def main():
    """Run PostgreSQL connection tests"""
    
    # Test connection variants
    working_url = await test_connection_variants()
    
    if working_url:
        # Test database operations with working connection
        await test_database_operations(working_url)
        
        logger.info(f"\n[RECOMMENDED CONNECTION STRING]")
        logger.info(f"{working_url}")
        logger.info(f"\nYou can now use this connection string in your applications.")
    else:
        logger.error("\n[ERROR] Unable to establish PostgreSQL connection")
        logger.info("Please check your PostgreSQL installation and configuration.")

if __name__ == "__main__":
    asyncio.run(main())
