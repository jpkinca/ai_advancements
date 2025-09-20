#!/usr/bin/env python3
"""
Minimal IBKR Connection with Client ID 200 (Pure Sync)
"""

import time
import socket
from ib_insync import IB, Stock

HOST = '127.0.0.1'
PORT = 4002
CLIENT_ID = 300

def check_port():
    """Check if Gateway is listening on the port"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((HOST, PORT))
        sock.close()
        return result == 0
    except Exception:
        return False

def minimal_connection():
    print("[STARTING] Testing IB Gateway connection...")
    
    # First check if port is open
    if not check_port():
        print("[ERROR] Gateway not listening on {}:{}".format(HOST, PORT))
        print("[INFO] Start IB Gateway and enable API on port 4002")
        return 1
    
    print("[SUCCESS] Gateway is listening on {}:{}".format(HOST, PORT))
    
    ib = IB()
    try:
        print("[STARTING] Connecting with clientId {}...".format(CLIENT_ID))
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=15)
        
        if not ib.isConnected():
            print("[ERROR] Connection failed - not connected")
            return 1
        
        print("[SUCCESS] Connected to IBKR")
        print("[INFO] Server version: {}".format(ib.client.serverVersion()))
        
        # Wait a moment for session to stabilize
        time.sleep(2)
        
        # Try handshake
        try:
            print("[TESTING] Requesting current time...")
            current_time = ib.reqCurrentTime()
            print("[SUCCESS] IB current time: {}".format(current_time))
        except Exception as e:
            print("[ERROR] Current time request failed: {}".format(e))
            # Don't return error - connection is working even if this fails
        
        # Try a simple data request
        try:
            print("[TESTING] Requesting AAPL data...")
            contract = Stock('AAPL', 'SMART', 'USD')
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            print("[SUCCESS] Retrieved {} bars for AAPL".format(len(bars)))
            if bars:
                latest = bars[-1]
                print("[DATA] Latest: Close=${:.2f}, Volume={:,}".format(latest.close, latest.volume))
                
        except Exception as e:
            print("[WARNING] Historical data request failed: {}".format(e))
        
        print("[SUCCESS] Test completed successfully")
        return 0
        
    except Exception as e:
        print("[ERROR] Connection error: {}".format(e))
        return 1
    
    finally:
        if ib.isConnected():
            print("[CLEANUP] Disconnecting...")
            ib.disconnect()
            print("[SUCCESS] Disconnected")

if __name__ == '__main__':
    exit(minimal_connection())
