"""
IBKR Gateway Connection Tester

Test IBKR Gateway connectivity and basic functionality before running
the full Chain-of-Alpha production pipeline.

Validates:
- IBKR Gateway connection
- Market data retrieval
- Basic TA-LIB indicators
- PostgreSQL connectivity
"""

import sys
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.production_config import get_config, validate_production_compliance
from src.ibkr_data_acquisition import IBKRDataAcquisition
# Note: Using direct database connection for testing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IBKRConnectionTester:
    """Test IBKR Gateway connection and basic functionality."""
    
    def __init__(self):
        self.config = get_config()
        self.data_acquisition = None
        self.database = None
    
    async def run_all_tests(self) -> bool:
        """
        Run comprehensive connection and functionality tests.
        
        Returns:
            True if all tests pass
        """
        logger.info("="*60)
        logger.info("IBKR GATEWAY CONNECTION TESTS")
        logger.info("="*60)
        
        tests = [
            ("Configuration Validation", self.test_configuration),
            ("IBKR Gateway Connection", self.test_ibkr_connection),
            ("Market Data Retrieval", self.test_market_data),
            ("TA-LIB Indicators", self.test_talib_indicators),
            ("PostgreSQL Database", self.test_database_connection),
            ("Data Persistence", self.test_data_persistence)
        ]
        
        results = {}
        all_passed = True
        
        for test_name, test_func in tests:
            logger.info(f"\n[TEST] {test_name}...")
            try:
                result = await test_func()
                results[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"[RESULT] {test_name}: {status}")
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"[ERROR] {test_name}: {e}")
                results[test_name] = False
                all_passed = False
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        
        if all_passed:
            logger.info("\n🎉 ALL TESTS PASSED - Ready for production!")
        else:
            logger.info("\n⚠️  Some tests failed - Check configuration and setup")
        
        return all_passed
    
    async def test_configuration(self) -> bool:
        """Test production configuration compliance."""
        try:
            # Validate compliance with CO-PILOT INSTRUCTIONS
            validate_production_compliance()
            
            # Check IBKR settings
            ibkr_config = self.config.get_ibkr_config()
            required_keys = ['host', 'port', 'client_id']
            for key in required_keys:
                if key not in ibkr_config:
                    logger.error(f"Missing IBKR config: {key}")
                    return False
            
            # Check database settings
            db_config = self.config.get_database_config()
            if not db_config.get('url'):
                logger.error("Missing database URL")
                return False
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    async def test_ibkr_connection(self) -> bool:
        """Test IBKR Gateway connection."""
        try:
            # Create data acquisition instance
            config_dict = {
                'ibkr_host': self.config.get_ibkr_config()['host'],
                'ibkr_port': self.config.get_ibkr_config()['port'],
                'ibkr_client_id': self.config.get_ibkr_config()['client_id'],
                'database_url': self.config.get_database_config()['url']
            }
            
            self.data_acquisition = IBKRDataAcquisition(config_dict)
            
            # Test connection
            connected = self.data_acquisition.connect_to_gateway()
            
            if connected:
                logger.info("✅ IBKR Gateway connection successful")
                
                # Test server info
                if hasattr(self.data_acquisition.ib, 'client'):
                    version = self.data_acquisition.ib.client.serverVersion()
                    logger.info(f"   Server version: {version}")
                
                # Disconnect
                self.data_acquisition.disconnect_from_gateway()
                return True
            else:
                logger.error("❌ IBKR Gateway connection failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ IBKR connection test failed: {e}")
            return False
    
    async def test_market_data(self) -> bool:
        """Test basic market data retrieval."""
        try:
            if not self.data_acquisition:
                return False
            
            # Test with a single ticker for speed
            test_config = {
                'tickers': ['AAPL'],
                'start_date': '2024-01-01',
                'end_date': datetime.now().strftime('%Y-%m-%d'),
                'ibkr_host': self.config.get_ibkr_config()['host'],
                'ibkr_port': self.config.get_ibkr_config()['port'],
                'ibkr_client_id': self.config.get_ibkr_config()['client_id'] + 1,  # Different client ID
                'database_url': self.config.get_database_config()['url']
            }
            
            test_acquisition = IBKRDataAcquisition(test_config)
            
            # Connect and fetch test data
            test_acquisition.connect_to_gateway()
            
            # Fetch single ticker data
            ticker_data = test_acquisition._fetch_single_ticker('AAPL')
            
            test_acquisition.disconnect_from_gateway()
            
            if ticker_data is not None and not ticker_data.empty:
                logger.info(f"✅ Market data retrieved: {len(ticker_data)} records")
                logger.info(f"   Columns: {list(ticker_data.columns)}")
                return True
            else:
                logger.error("❌ No market data retrieved")
                return False
                
        except Exception as e:
            logger.error(f"❌ Market data test failed: {e}")
            return False
    
    async def test_talib_indicators(self) -> bool:
        """Test TA-LIB indicator calculations."""
        try:
            import talib
            import numpy as np
            
            # Test with sample data
            test_data = np.random.rand(100) * 100 + 50  # Random price data
            
            # Test basic indicators
            sma = talib.SMA(test_data, timeperiod=20)
            rsi = talib.RSI(test_data, timeperiod=14)
            
            if not np.isnan(sma[-1]) and not np.isnan(rsi[-1]):
                logger.info("✅ TA-LIB indicators working")
                logger.info(f"   SMA: {sma[-1]:.2f}, RSI: {rsi[-1]:.2f}")
                return True
            else:
                logger.error("❌ TA-LIB indicators returned NaN")
                return False
                
        except ImportError:
            logger.error("❌ TA-LIB not installed")
            return False
        except Exception as e:
            logger.error(f"❌ TA-LIB test failed: {e}")
            return False
    
    async def test_database_connection(self) -> bool:
        """Test PostgreSQL database connection."""
        try:
            import asyncpg
            
            db_url = self.config.get_database_config()['url']
            
            # Test connection
            conn = await asyncpg.connect(db_url)
            
            # Test query
            result = await conn.fetchval('SELECT version()')
            
            await conn.close()
            
            logger.info("✅ PostgreSQL connection successful")
            logger.info(f"   Version: {result.split(',')[0]}")
            return True
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            return False
    
    async def test_data_persistence(self) -> bool:
        """Test data persistence to PostgreSQL."""
        try:
            import asyncpg
            import pandas as pd
            from datetime import datetime
            
            db_url = self.config.get_database_config()['url']
            conn = await asyncpg.connect(db_url)
            
            # Create test table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ibkr_test_data (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10),
                    test_date TIMESTAMPTZ,
                    test_value DECIMAL(10,2),
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert test data
            test_ticker = 'TEST'
            test_date = datetime.now()
            test_value = 123.45
            
            await conn.execute("""
                INSERT INTO ibkr_test_data (ticker, test_date, test_value)
                VALUES ($1, $2, $3)
            """, test_ticker, test_date, test_value)
            
            # Query test data
            result = await conn.fetchrow("""
                SELECT * FROM ibkr_test_data 
                WHERE ticker = $1 
                ORDER BY created_at DESC 
                LIMIT 1
            """, test_ticker)
            
            # Cleanup
            await conn.execute("DELETE FROM ibkr_test_data WHERE ticker = $1", test_ticker)
            await conn.close()
            
            if result and result['test_value'] == test_value:
                logger.info("✅ Data persistence test successful")
                return True
            else:
                logger.error("❌ Data persistence test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Data persistence test failed: {e}")
            return False
    
    async def test_ai_database_integration(self) -> bool:
        """Test database integration (simplified for testing)."""
        try:
            import asyncpg
            
            db_url = self.config.get_database_config()['url']
            conn = await asyncpg.connect(db_url)
            
            # Test creating a signals table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_ai_signals (
                    id SERIAL PRIMARY KEY,
                    signal_type VARCHAR(50),
                    confidence DECIMAL(5,4),
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Test inserting test signal
            await conn.execute("""
                INSERT INTO test_ai_signals (signal_type, confidence)
                VALUES ($1, $2)
            """, 'test_signal', 0.75)
            
            # Cleanup
            await conn.execute("DROP TABLE IF EXISTS test_ai_signals")
            await conn.close()
            
            logger.info("✅ Database integration test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database integration test failed: {e}")
            return False


async def main():
    """Main test execution."""
    try:
        # Check if IBKR Gateway is likely running
        logger.info("Starting IBKR Gateway connection tests...")
        logger.info("IMPORTANT: Ensure IBKR Gateway/TWS is running and API is enabled!")
        
        # Run tests
        tester = IBKRConnectionTester()
        success = await tester.run_all_tests()
        
        if success:
            logger.info("\n🚀 Ready to run production Chain-of-Alpha pipeline!")
        else:
            logger.info("\n🔧 Please fix issues before running production pipeline")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(main())