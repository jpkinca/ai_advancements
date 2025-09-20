#!/usr/bin/env python3
"""
IB Gateway Connection Diagnostic
- Tests raw socket connection first
- Shows detailed connection steps
- Checks for common Gateway configuration issues
"""

import socket
import time
import threading
from ib_insync import IB

HOST = '127.0.0.1'
PORT = 4002
CLIENT_ID = 200

def test_raw_socket():
    """Test raw socket connection to see if Gateway accepts connections"""
    print("[DIAGNOSTIC] Testing raw socket connection...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((HOST, PORT))
        
        # Send a simple byte to see if Gateway responds
        sock.send(b'TEST\r\n')
        time.sleep(1)
        
        # Try to receive something
        sock.settimeout(2)
        try:
            response = sock.recv(1024)
            print(f"[SUCCESS] Gateway responded: {response[:50]}...")
            sock.close()
            return True
        except socket.timeout:
            print("[WARNING] Gateway accepted connection but no response")
            sock.close()
            return True
        
    except socket.timeout:
        print("[ERROR] Connection timeout - Gateway may not be responding")
        return False
    except ConnectionRefusedError:
        print("[ERROR] Connection refused - Gateway not running or API disabled")
        return False
    except Exception as e:
        print(f"[ERROR] Socket test failed: {e}")
        return False

def connection_timeout_handler():
    """Print a message if connection takes too long"""
    time.sleep(10)
    print("[WARNING] Connection taking longer than 10 seconds...")
    print("[INFO] This may indicate:")
    print("  - IB Gateway login required")
    print("  - API permissions not enabled")
    print("  - Wrong port (try 7497 for TWS)")
    print("  - Firewall blocking connection")

def main():
    print("=== IB Gateway Connection Diagnostic ===")
    
    # Test 1: Raw socket
    if not test_raw_socket():
        print("\n[FAILED] Cannot establish basic connection to Gateway")
        print("Check that IB Gateway is running and API is enabled")
        return 1
    
    print("\n[SUCCESS] Basic socket connection works")
    print("[INFO] Testing IB API connection...")
    
    # Start timeout warning thread
    timeout_thread = threading.Thread(target=connection_timeout_handler, daemon=True)
    timeout_thread.start()
    
    ib = IB()
    try:
        print(f"[TESTING] Connecting to {HOST}:{PORT} with clientId {CLIENT_ID}...")
        start_time = time.time()
        
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=20)
        
        elapsed = time.time() - start_time
        print(f"[SUCCESS] Connected in {elapsed:.1f} seconds")
        
        if not ib.isConnected():
            print("[ERROR] Connect returned but not connected")
            return 1
        
        print(f"[SUCCESS] API connection established")
        print(f"[INFO] Server version: {ib.client.serverVersion()}")
        
        # Test API call
        try:
            current_time = ib.reqCurrentTime()
            print(f"[SUCCESS] Current time: {current_time}")
        except Exception as e:
            print(f"[WARNING] API call failed: {e}")
        
        return 0
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] Connection failed after {elapsed:.1f} seconds: {e}")
        
        # Common troubleshooting
        print("\n[TROUBLESHOOTING]")
        print("1. Is IB Gateway fully logged in?")
        print("2. Is API enabled in Gateway settings?")
        print("3. Is 'Enable ActiveX and Socket Clients' checked?")
        print("4. Is port 4002 configured for API?")
        print("5. Is 'Allow connections from localhost only' enabled?")
        print("6. Try different clientId (current: {})".format(CLIENT_ID))
        
        return 1
    
    finally:
        if ib.isConnected():
            print("[CLEANUP] Disconnecting...")
            ib.disconnect()

if __name__ == '__main__':
    exit(main())
