#!/usr/bin/env python3
"""
PostgreSQL Connection Test for Railway Database - Trading System Compatible
"""

import os
import sys
import psycopg2

def test_postgresql_connection():
    """Test connection to Railway PostgreSQL database using psycopg2 (same as trading system)"""
    try:
        # Get Railway PostgreSQL connection string
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("❌ DATABASE_URL environment variable not found")
            print("Please set DATABASE_URL to your Railway PostgreSQL connection string")
            print("Example: postgresql://username:password@host:port/database")
            print()
            print("For Railway, you can find this in your Railway project dashboard:")
            print("1. Go to your Railway project")
            print("2. Click on your PostgreSQL database")
            print("3. Go to the 'Connect' tab")
            print("4. Copy the 'PostgreSQL Connection URL'")
            return False

        print("🔗 Connecting to Railway PostgreSQL...")
        print(f"Database URL: {database_url[:50]}...")

        # Connect to PostgreSQL (same as trading system)
        conn = psycopg2.connect(database_url)
        conn.autocommit = True

        # Test connection with a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()
        print("✅ Connected successfully!")
        print(f"PostgreSQL version: {version[0][:50]}...")

        # Test table creation (same as trading system) - drop first to ensure clean schema
        cursor.execute("DROP TABLE IF EXISTS live_trades")
        cursor.execute("DROP TABLE IF EXISTS market_signals")

        cursor.execute("""
            CREATE TABLE live_trades (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                action VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                strategy VARCHAR(50),
                pnl TEXT DEFAULT '',
                status VARCHAR(20) DEFAULT 'open'
            )
        """)

        cursor.execute("""
            CREATE TABLE market_signals (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                signal_type VARCHAR(10) NOT NULL,
                confidence DECIMAL(3,2),
                price DECIMAL(10,2),
                volume INTEGER,
                source VARCHAR(50)
            )
        """)

        print("✅ Trading system tables created successfully!")

        # Test data insertion (same as trading system)
        cursor.execute("""
            INSERT INTO market_signals
            (symbol, signal_type, confidence, price, volume, source)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, ('TEST', 'BUY', 0.85, 150.00, 1000000, 'connection_test'))

        cursor.execute("""
            INSERT INTO live_trades
            (symbol, action, quantity, price, strategy, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, ('TEST', 'BUY', 100, 150.00, 'test_trade', 'pending'))

        trade_id = cursor.fetchone()[0]

        cursor.execute("""
            UPDATE live_trades
            SET pnl = %s, status = %s
            WHERE id = %s
        """, ('stop:147.00,target:153.00', 'active', trade_id))

        print("✅ Test data insertion successful!")

        # Query test data
        cursor.execute("SELECT COUNT(*) FROM market_signals WHERE source = 'connection_test'")
        signal_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM live_trades WHERE strategy = 'test_trade'")
        trade_count = cursor.fetchone()[0]

        print(f"✅ Database contains {signal_count} test signals and {trade_count} test trades")

        # Clean up test data
        cursor.execute("DELETE FROM market_signals WHERE source = 'connection_test'")
        cursor.execute("DELETE FROM live_trades WHERE strategy = 'test_trade'")
        print("✅ Test data cleanup completed")

        conn.close()
        print("✅ Connection closed successfully")
        return True

    except psycopg2.Error as e:
        print(f"❌ PostgreSQL error: {e}")
        print()
        print("Troubleshooting:")
        print("- Check your DATABASE_URL format")
        print("- Verify Railway database is running")
        print("- Check network connectivity to Railway")
        return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    print("PostgreSQL Railway Connection Test - Trading System Compatible")
    print("=" * 60)

    success = test_postgresql_connection()

    if success:
        print("\n🎉 PostgreSQL connection test PASSED!")
        print("Your Railway database is ready for the trading system.")
        print("\nNext steps:")
        print("1. Set DATABASE_URL environment variable in Railway")
        print("2. Deploy the trading system to Railway")
        print("3. Run during market hours to generate live signals")
        sys.exit(0)
    else:
        print("\n❌ PostgreSQL connection test FAILED!")
        print("Please check your DATABASE_URL and Railway configuration.")
        sys.exit(1)