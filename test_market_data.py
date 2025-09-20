#!/usr/bin/env python3
"""
Quick IBKR Data Test

Simple test to verify we can get live market data from IBKR for pattern generation.
"""

import os
import sys
import logging
from datetime import datetime
import asyncio

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_market_data():
    """Test getting market data from IBKR"""
    try:
        logger.info("[TESTING] Getting live market data from IBKR...")

        from ib_insync import IB, Stock

        # Connect to IBKR directly on the current event loop (avoids cross-loop issues)
        ib = IB()
        try:
            await ib.connectAsync('127.0.0.1', 4002, clientId=14, timeout=15)
        except Exception as e:
            logger.error(f"[ERROR] Async connect failed: {e}")
            return False

        if not ib.isConnected():
            logger.error("[ERROR] Failed to connect to IBKR")
            return False

        logger.info("[SUCCESS] Connected to IBKR (direct async)")

        # Perform immediate handshake to verify API traffic
        try:
            current_time = await ib.reqCurrentTimeAsync()
            logger.info(f"[SUCCESS] Handshake OK. IB current time: {current_time}")
        except Exception as e:
            logger.error(f"[ERROR] Handshake failed: {e}")
            # Ensure we cleanly disconnect before returning
            ib.disconnect()
            return False

        # Test getting historical data for AAPL
        contract = Stock('AAPL', 'SMART', 'USD')

        # Get last 5 days of daily data (async)
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=datetime.now(),
                durationStr='5 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
        except Exception as e:
            logger.error(f"[ERROR] Daily data request failed: {e}")
            return False

        if bars:
            logger.info(f"[SUCCESS] Retrieved {len(bars)} daily bars for AAPL")

            # Show sample data
            latest = bars[-1]
            logger.info(f"[DATA] Latest bar: Date={latest.date}, Close=${latest.close:.2f}, Volume={latest.volume:,}")

            # Test minute data (async)
            try:
                minute_bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=datetime.now(),
                    durationStr='1 D',
                    barSizeSetting='1 min',
                    whatToShow='TRADES',
                    useRTH=False,
                    formatDate=1
                )
            except Exception as e:
                logger.error(f"[ERROR] Minute data request failed: {e}")
                return False

            if minute_bars:
                logger.info(f"[SUCCESS] Retrieved {len(minute_bars)} minute bars for AAPL")
                logger.info("[SUCCESS] Market data retrieval is working perfectly!")
                return True
            else:
                logger.warning("[WARNING] No minute data retrieved")
                return False
        else:
            logger.error("[ERROR] No daily data retrieved")
            ib.disconnect()
            return False

    except Exception as e:
        logger.error(f"[ERROR] Market data test failed: {e}")
        return False
    finally:
        try:
            # Best-effort disconnect if still connected
            if 'ib' in locals() and getattr(ib, 'isConnected', lambda: False)():
                ib.disconnect()
                logger.info("[SUCCESS] Disconnected from IBKR")
        except Exception:
            pass

async def main():
    """Run market data test"""
    logger.info("=== IBKR MARKET DATA TEST ===")

    # Ensure proper event loop policy on Windows
    if sys.platform.startswith('win'):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            logger.info("[INFO] Windows event loop policy set to WindowsSelectorEventLoopPolicy")
        except Exception as _:
            pass
    
    if await test_market_data():
        logger.info("")
        logger.info("[SUCCESS] Market data test passed!")
        logger.info("Ready for FAISS pattern generation with live data.")
    else:
        logger.info("")
        logger.info("[FAILED] Market data test failed.")
        logger.info("Check IB Gateway connection and permissions.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
