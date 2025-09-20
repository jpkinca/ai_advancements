#!/usr/bin/env python3
"""
Quick IBKR Gateway Connection Test

Simple test to verify IB Gateway is running and accessible on port 4002.
Use this before running the full FAISS pattern integration.

Author: GitHub Copilot
Date: 2025-09-03
"""

import socket
import logging
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_port_connection(host='127.0.0.1', port=4002):
    """Test if IB Gateway is listening on the specified port"""
    try:
        logger.info(f"[TESTING] Checking connection to {host}:{port}...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout
        
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            logger.info(f"[SUCCESS] IB Gateway is running on {host}:{port}")
            return True
        else:
            logger.error(f"[ERROR] Cannot connect to {host}:{port}")
            logger.error(f"[ERROR] Make sure IB Gateway is running on port {port}")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] Connection test failed: {e}")
        return False

def test_ib_connection():
    """Test actual IB connection using ib_insync"""
    try:
        logger.info("[TESTING] Testing ib_insync connection...")
        
        from ib_insync import IB
        import asyncio
        
        ib = IB()
        
        # Try connection with timeout handling
        try:
            ib.connect('127.0.0.1', 4002, clientId=999, timeout=10)
            
            if ib.isConnected():
                logger.info("[SUCCESS] ib_insync connection successful")
                
                # Get account info as additional test
                try:
                    accounts = ib.managedAccounts()
                    account = accounts[0] if accounts else "Paper Trading Account"
                    logger.info(f"[SUCCESS] Connected to account: {account}")
                except Exception as e:
                    logger.info(f"[INFO] Account info not available: {e}")
                
                ib.disconnect()
                return True
            else:
                logger.error("[ERROR] ib_insync connection failed")
                return False
                
        except Exception as conn_e:
            logger.error(f"[ERROR] Connection attempt failed: {conn_e}")
            if ib.isConnected():
                ib.disconnect()
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] ib_insync test failed: {e}")
        return False

def test_connect_me_function():
    """Test your custom connect_me function"""
    try:
        logger.info("[TESTING] Testing connect_me function...")
        
        from ibkr_api.connect_me import get_managed_ibkr_connection
        
        # Use a registered component name (from client ID registry)
        ib = get_managed_ibkr_connection("pattern_detector")
        
        if ib and ib.isConnected():
            logger.info("[SUCCESS] connect_me function works perfectly")
            
            # Test basic functionality
            try:
                accounts = ib.managedAccounts()
                logger.info(f"[SUCCESS] Retrieved {len(accounts)} managed accounts")
            except Exception as e:
                logger.info(f"[INFO] Account query test: {e}")
            
            return True
        else:
            logger.error("[ERROR] connect_me function failed - no connection")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] connect_me test failed: {e}")
        return False

def main():
    """Run all connection tests"""
    
    logger.info("=== IBKR GATEWAY CONNECTION TEST ===")
    logger.info("")
    
    # Test 1: Port connectivity
    logger.info("[TEST 1] Port Connectivity")
    port_ok = test_port_connection()
    
    if not port_ok:
        logger.error("")
        logger.error("=== CONNECTION FAILED ===")
        logger.error("IB Gateway is not running or not configured properly")
        logger.error("")
        logger.error("To fix:")
        logger.error("1. Start IB Gateway application")
        logger.error("2. Login with paper trading account")
        logger.error("3. Enable API on port 4002")
        logger.error("4. Allow localhost connections")
        return False
    
    # Test 2: ib_insync connection
    logger.info("")
    logger.info("[TEST 2] ib_insync Connection")
    ib_ok = test_ib_connection()
    
    # Test 3: connect_me function
    logger.info("")
    logger.info("[TEST 3] connect_me Function")
    connect_me_ok = test_connect_me_function()
    
    # Final result
    logger.info("")
    logger.info("=== TEST RESULTS ===")
    logger.info(f"Port Connectivity: {'PASS' if port_ok else 'FAIL'}")
    logger.info(f"ib_insync Connection: {'PASS' if ib_ok else 'FAIL'}")
    logger.info(f"connect_me Function: {'PASS' if connect_me_ok else 'FAIL'}")
    
    if port_ok and ib_ok and connect_me_ok:
        logger.info("")
        logger.info("[SUCCESS] All tests passed! IB Gateway is ready.")
        logger.info("You can now run: python ai_advancements/test_integration.py")
        return True
    else:
        logger.info("")
        logger.info("[WARNING] Some tests failed. Check IB Gateway configuration.")
        return False

if __name__ == "__main__":
    main()
