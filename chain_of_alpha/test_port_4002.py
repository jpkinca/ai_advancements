"""
Direct IBKR Gateway Connection Test - Port 4002

Attempts to connect directly to IBKR Gateway on port 4002
and provides detailed connection diagnostics.
"""

import asyncio
import logging
import socket
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_port_4002():
    """Test direct connection to port 4002."""
    
    logger.info("="*60)
    logger.info("IBKR GATEWAY PORT 4002 CONNECTION TEST")
    logger.info("="*60)
    
    # 1. Test raw socket connection
    logger.info("\n[TEST 1] Raw socket connection to 127.0.0.1:4002")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', 4002))
        sock.close()
        
        if result == 0:
            logger.info("✅ Port 4002 is OPEN and accepting connections")
        else:
            logger.info("❌ Port 4002 is CLOSED or not accessible")
            logger.info("   Connection refused - IBKR Gateway is likely not running")
    except Exception as e:
        logger.error(f"❌ Socket test failed: {e}")
    
    # 2. Test alternative ports
    logger.info("\n[TEST 2] Checking alternative IBKR ports")
    ports_to_test = [4001, 4002, 7496, 7497, 7498]
    
    for port in ports_to_test:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            status = "✅ OPEN" if result == 0 else "❌ CLOSED"
            port_type = {
                4001: "(Live Trading Gateway)",
                4002: "(Paper Trading Gateway)", 
                7496: "(Paper Trading TWS)",
                7497: "(Live Trading TWS)",
                7498: "(Gateway Alternative)"
            }.get(port, "")
            
            logger.info(f"   Port {port}: {status} {port_type}")
            
        except Exception as e:
            logger.info(f"   Port {port}: ❌ ERROR - {e}")
    
    # 3. Test ib_insync connection
    logger.info("\n[TEST 3] IB-insync connection test")
    try:
        from ib_insync import IB
        
        ib = IB()
        logger.info("   Attempting connection to 127.0.0.1:4002...")
        
        try:
            ib.connect('127.0.0.1', 4002, clientId=301, timeout=10)
            
            if ib.isConnected():
                logger.info("✅ Successfully connected to IBKR Gateway!")
                logger.info(f"   Server version: {ib.client.serverVersion()}")
                logger.info(f"   Connection time: {ib.client.connectTime()}")
                
                # Test basic functionality
                logger.info("   Testing contract creation...")
                from ib_insync import Stock
                contract = Stock('AAPL', 'SMART', 'USD')
                ib.qualifyContracts(contract)
                logger.info("   ✅ Contract qualification successful")
                
                # Disconnect
                ib.disconnect()
                logger.info("   ✅ Clean disconnection")
                
            else:
                logger.error("   ❌ Connection failed - not connected")
                
        except ConnectionRefusedError:
            logger.error("   ❌ Connection refused - IBKR Gateway not running on port 4002")
        except Exception as e:
            logger.error(f"   ❌ Connection failed: {e}")
            
    except ImportError:
        logger.error("   ❌ ib_insync not available")
    
    # 4. Provide instructions
    logger.info("\n" + "="*60)
    logger.info("IBKR GATEWAY SETUP INSTRUCTIONS")
    logger.info("="*60)
    logger.info("\n1. Download IBKR Gateway:")
    logger.info("   https://www.interactivebrokers.com/en/trading/ib-gateway.php")
    
    logger.info("\n2. Start IBKR Gateway:")
    logger.info("   - Launch IB Gateway application")
    logger.info("   - Login with your IBKR credentials")
    logger.info("   - Choose 'Live Trading' or 'Paper Trading'")
    
    logger.info("\n3. Enable API Access:")
    logger.info("   - Go to Configure → Settings → API → Settings")
    logger.info("   - Check 'Enable ActiveX and Socket Clients'")
    logger.info("   - Set Socket port to 4001 (Live) or 4002 (Paper)")
    logger.info("   - Add client ID 300-400 to trusted clients")
    logger.info("   - Check 'Read-Only API'")
    
    logger.info("\n4. Firewall/Network:")
    logger.info("   - Ensure Windows Firewall allows IB Gateway")
    logger.info("   - Check antivirus software permissions")
    
    logger.info("\n5. Test Connection:")
    logger.info("   - Run: python test_port_4002.py")
    logger.info("   - Should see 'Port 4002 is OPEN'")
    
    logger.info("\nCurrent Status: IBKR Gateway appears to be NOT RUNNING")
    logger.info("Please start IBKR Gateway and enable API access.")


async def attempt_live_connection():
    """Attempt a live connection and provide real-time feedback."""
    
    logger.info("\n" + "="*60)
    logger.info("LIVE CONNECTION ATTEMPT")
    logger.info("="*60)
    
    try:
        from ib_insync import IB, Stock
        
        ib = IB()
        
        # Enable detailed logging
        import ib_insync
        ib_insync.util.startLoop()
        
        logger.info("Attempting connection...")
        logger.info("If this hangs, IBKR Gateway is not running or not accepting connections")
        
        # Try connection with extended timeout
        ib.connect('127.0.0.1', 4002, clientId=302, timeout=15)
        
        if ib.isConnected():
            logger.info("🎉 CONNECTION SUCCESSFUL!")
            
            # Get account info
            accounts = ib.managedAccounts()
            logger.info(f"   Available accounts: {accounts}")
            
            # Test market data request
            logger.info("   Testing market data request...")
            contract = Stock('AAPL', 'SMART', 'USD')
            ib.qualifyContracts(contract)
            
            # Request market data
            ticker = ib.reqMktData(contract)
            await asyncio.sleep(2)  # Wait for data
            
            if ticker.last > 0:
                logger.info(f"   ✅ Market data received: AAPL = ${ticker.last}")
            else:
                logger.info("   ⚠️ No market data received (may need market data subscription)")
            
            ib.cancelMktData(contract)
            ib.disconnect()
            
            logger.info("🚀 READY FOR PRODUCTION CHAIN-OF-ALPHA!")
            return True
            
        else:
            logger.error("❌ Connection failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Live connection failed: {e}")
        return False


async def main():
    """Main execution."""
    logger.info(f"Testing IBKR Gateway connection at {datetime.now()}")
    
    # Basic connection tests
    await test_port_4002()
    
    # Attempt live connection
    success = await attempt_live_connection()
    
    if success:
        logger.info("\n🎯 IBKR Gateway is working! Ready to run Chain-of-Alpha pipeline.")
    else:
        logger.info("\n🔧 Please start IBKR Gateway and try again.")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())