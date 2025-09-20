"""
Simple IBKR Gateway Connection Test

A simplified test that bypasses the complex configuration system
and directly tests IBKR connectivity for Chain-of-Alpha.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleIBKRTest:
    """Simple IBKR Gateway connectivity test."""
    
    def __init__(self):
        self.ib = None
    
    async def run_tests(self) -> bool:
        """Run basic connectivity tests."""
        logger.info("="*60)
        logger.info("SIMPLE IBKR GATEWAY CONNECTION TEST")
        logger.info("="*60)
        
        tests = [
            ("Import Dependencies", self.test_imports),
            ("IBKR Connection", self.test_ibkr_basic),
            ("TA-LIB Functionality", self.test_talib),
            ("PostgreSQL Connection", self.test_postgresql),
        ]
        
        results = {}
        all_passed = True
        
        for test_name, test_func in tests:
            logger.info(f"\n[TEST] {test_name}...")
            try:
                result = await test_func()
                results[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"[RESULT] {status}")
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"[ERROR] {e}")
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
            logger.info("\n🎉 ALL TESTS PASSED - IBKR Gateway is ready!")
        else:
            logger.info("\n⚠️  Some tests failed - Check setup")
        
        return all_passed
    
    async def test_imports(self) -> bool:
        """Test all required imports."""
        try:
            # Test core dependencies
            import pandas
            import numpy
            import torch
            import transformers
            logger.info("   Core libraries: ✅")
            
            # Test IBKR
            from ib_insync import IB, Stock, util
            logger.info("   ib_insync: ✅")
            
            # Test TA-LIB
            import talib
            logger.info("   TA-LIB: ✅")
            
            # Test PostgreSQL
            import asyncpg
            logger.info("   asyncpg: ✅")
            
            return True
            
        except ImportError as e:
            logger.error(f"   Missing dependency: {e}")
            return False
    
    async def test_ibkr_basic(self) -> bool:
        """Test basic IBKR Gateway connection."""
        try:
            from ib_insync import IB, Stock
            
            self.ib = IB()
            
            # Try to connect to Gateway
            logger.info("   Attempting connection to IBKR Gateway...")
            logger.info("   (Make sure IBKR Gateway/TWS is running with API enabled)")
            
            # Try standard Gateway port first
            try:
                self.ib.connect('127.0.0.1', 4002, clientId=300, timeout=10)
                logger.info("   Connected to Gateway on port 4002")
            except:
                # Try TWS port
                try:
                    self.ib.connect('127.0.0.1', 7497, clientId=300, timeout=10)
                    logger.info("   Connected to TWS on port 7497")
                except:
                    logger.error("   Could not connect to IBKR Gateway or TWS")
                    return False
            
            if not self.ib.isConnected():
                return False
            
            # Test basic functionality
            logger.info("   Testing contract creation...")
            contract = Stock('AAPL', 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            logger.info("   Contract qualified successfully")
            
            # Disconnect
            self.ib.disconnect()
            logger.info("   Disconnected successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"   IBKR test failed: {e}")
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
            return False
    
    async def test_talib(self) -> bool:
        """Test TA-LIB indicators."""
        try:
            import talib
            
            # Create test data
            test_data = np.random.rand(100) * 100 + 50
            
            # Test various indicators
            sma = talib.SMA(test_data, timeperiod=20)
            rsi = talib.RSI(test_data, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(test_data)
            
            # Check if indicators work
            if np.isnan(sma[-1]) or np.isnan(rsi[-1]):
                return False
            
            logger.info(f"   SMA: {sma[-1]:.2f}")
            logger.info(f"   RSI: {rsi[-1]:.2f}")
            logger.info(f"   MACD: {macd[-1]:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"   TA-LIB test failed: {e}")
            return False
    
    async def test_postgresql(self) -> bool:
        """Test PostgreSQL connection."""
        try:
            import asyncpg
            
            # Use the production database URL
            db_url = 'postgresql://postgres:TAqEkujnMknVURCcrYTIDOzQXbgBNtSX@turntable.proxy.rlwy.net:10410/railway'
            
            logger.info("   Connecting to PostgreSQL...")
            conn = await asyncpg.connect(db_url)
            
            # Test basic query
            version = await conn.fetchval('SELECT version()')
            logger.info(f"   PostgreSQL: {version.split(',')[0]}")
            
            # Test table creation
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ibkr_test (
                    id SERIAL PRIMARY KEY,
                    test_value VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Test insert
            await conn.execute("INSERT INTO ibkr_test (test_value) VALUES ($1)", "test_connection")
            
            # Test select
            result = await conn.fetchval("SELECT test_value FROM ibkr_test ORDER BY created_at DESC LIMIT 1")
            
            if result != "test_connection":
                return False
            
            # Cleanup
            await conn.execute("DELETE FROM ibkr_test WHERE test_value = $1", "test_connection")
            await conn.close()
            
            logger.info("   Database operations successful")
            return True
            
        except Exception as e:
            logger.error(f"   PostgreSQL test failed: {e}")
            return False


async def main():
    """Main test execution."""
    try:
        tester = SimpleIBKRTest()
        success = await tester.run_tests()
        
        if success:
            logger.info("\n🚀 READY FOR PRODUCTION CHAIN-OF-ALPHA!")
            logger.info("\nNext steps:")
            logger.info("1. Ensure IBKR Gateway is running")
            logger.info("2. Run: python chain_of_alpha_production.py")
        else:
            logger.info("\n🔧 Please fix issues before proceeding")
            logger.info("\nTroubleshooting:")
            logger.info("- Install missing dependencies: pip install -r requirements_production.txt")
            logger.info("- Start IBKR Gateway/TWS with API enabled")
            logger.info("- Check database connectivity")
        
        return success
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(main())