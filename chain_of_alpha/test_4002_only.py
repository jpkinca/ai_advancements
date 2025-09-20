"""
IBKR Gateway Port 4002 (Paper Trading) Connection Test

Tests ONLY port 4002 for paper trading connection.
"""

import asyncio
import logging
import socket
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_port_4002_only():
    """Test connection to port 4002 (Paper Trading) ONLY."""
    
    logger.info("="*60)
    logger.info("IBKR GATEWAY PORT 4002 (PAPER TRADING) TEST")
    logger.info("="*60)
    
    # 1. Test raw socket connection to port 4002
    logger.info("\n[TEST] Raw socket connection to 127.0.0.1:4002")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', 4002))
        sock.close()
        
        if result == 0:
            logger.info("✅ Port 4002 is OPEN - Paper Trading Gateway is running")
        else:
            logger.info("❌ Port 4002 is CLOSED - Paper Trading Gateway is NOT running")
            return False
    except Exception as e:
        logger.error(f"❌ Socket test failed: {e}")
        return False
    
    # 2. Test ib_insync connection to port 4002
    logger.info("\n[TEST] IB-insync connection to port 4002")
    try:
        from ib_insync import IB, Stock
        
        ib = IB()
        logger.info("   Connecting to Paper Trading Gateway on port 4002...")
        
        # Connect to paper trading port
        ib.connect('127.0.0.1', 4002, clientId=400, timeout=15)
        
        if ib.isConnected():
            logger.info("✅ Successfully connected to Paper Trading Gateway!")
            logger.info(f"   Server version: {ib.client.serverVersion()}")
            
            # Get account info
            accounts = ib.managedAccounts()
            logger.info(f"   Paper trading accounts: {accounts}")
            
            # Test contract creation
            logger.info("   Testing contract qualification...")
            contract = Stock('AAPL', 'SMART', 'USD')
            qualified = ib.qualifyContracts(contract)
            
            if qualified:
                logger.info("   ✅ Contract qualification successful")
                logger.info(f"   Contract details: {qualified[0]}")
            
            # Test market data request
            logger.info("   Testing market data request...")
            ticker = ib.reqMktData(contract)
            await asyncio.sleep(3)  # Wait for data
            
            if hasattr(ticker, 'last') and ticker.last > 0:
                logger.info(f"   ✅ Market data received: AAPL = ${ticker.last}")
            elif hasattr(ticker, 'close') and ticker.close > 0:
                logger.info(f"   ✅ Market data received: AAPL close = ${ticker.close}")
            else:
                logger.info("   ⚠️ No real-time market data (normal for paper trading)")
            
            # Cancel market data and disconnect
            ib.cancelMktData(contract)
            ib.disconnect()
            logger.info("   ✅ Clean disconnection from Paper Trading Gateway")
            
            return True
            
        else:
            logger.error("   ❌ Connection failed - not connected")
            return False
            
    except Exception as e:
        logger.error(f"   ❌ Connection failed: {e}")
        return False


async def test_historical_data():
    """Test historical data retrieval from port 4002."""
    
    logger.info("\n[TEST] Historical data retrieval from port 4002")
    try:
        from ib_insync import IB, Stock
        from datetime import datetime, timedelta
        
        ib = IB()
        ib.connect('127.0.0.1', 4002, clientId=401, timeout=15)
        
        if ib.isConnected():
            contract = Stock('AAPL', 'SMART', 'USD')
            ib.qualifyContracts(contract)
            
            # Request historical data
            logger.info("   Requesting historical data for AAPL...")
            end_date = datetime.now()
            
            bars = ib.reqHistoricalData(
                contract=contract,
                endDateTime=end_date,
                durationStr='30 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if bars:
                logger.info(f"   ✅ Retrieved {len(bars)} historical bars")
                logger.info(f"   Latest bar: {bars[-1]}")
                return True
            else:
                logger.info("   ❌ No historical data retrieved")
                return False
                
        else:
            logger.error("   ❌ Connection failed for historical data test")
            return False
            
    except Exception as e:
        logger.error(f"   ❌ Historical data test failed: {e}")
        return False
    finally:
        if 'ib' in locals() and ib.isConnected():
            ib.disconnect()


async def main():
    """Main execution - test port 4002 only."""
    logger.info(f"Testing IBKR Paper Trading Gateway (port 4002) at {datetime.now()}")
    
    # Test basic connection
    connection_success = await test_port_4002_only()
    
    if connection_success:
        logger.info("\n✅ Port 4002 connection successful!")
        
        # Test historical data
        data_success = await test_historical_data()
        
        if data_success:
            logger.info("\n🎉 PORT 4002 (PAPER TRADING) IS FULLY OPERATIONAL!")
            logger.info("🚀 Ready to run Chain-of-Alpha production pipeline with paper trading data")
        else:
            logger.info("\n⚠️ Connection works but historical data may be limited")
            logger.info("🚀 Still ready for Chain-of-Alpha pipeline")
            
    else:
        logger.info("\n❌ Port 4002 connection failed")
        logger.info("📋 Please ensure:")
        logger.info("   1. IBKR Gateway is running")
        logger.info("   2. Paper Trading mode is selected")
        logger.info("   3. API is enabled on port 4002")
        logger.info("   4. Client ID 400+ is allowed")
    
    return connection_success


if __name__ == "__main__":
    asyncio.run(main())