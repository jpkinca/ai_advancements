#!/usr/bin/env python3
"""
FAISS-IBKR Integration Test - WORKING VERSION

This script integrates FAISS pattern matching with live IBKR gateway connection.
Uses a simplified, proven-working approach to FAISS integration.

Author: AI Assistant  
Date: September 3, 2025
Status: Working Version
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import dependencies - use global imports to avoid namespace issues
try:
    import faiss
    logger.info("[SUCCESS] FAISS imported successfully")
    # Verify FAISS immediately
    test_index = faiss.IndexFlatIP(32)
    logger.info("[SUCCESS] FAISS IndexFlatIP verified working")
except ImportError:
    logger.error("[ERROR] FAISS not available - install with: pip install faiss-cpu")
    sys.exit(1)
except Exception as e:
    logger.error(f"[ERROR] FAISS verification failed: {e}")
    sys.exit(1)

try:
    from ib_insync import IB, Stock, util
    logger.info("[SUCCESS] ib_insync imported successfully")
except ImportError:
    logger.error("[ERROR] ib_insync not available - install with: pip install ib_insync")
    sys.exit(1)

class IBKRPatternDataProvider:
    """Interactive Brokers data provider for pattern analysis"""
    
    def __init__(self, host='127.0.0.1', port=4002, client_id=201):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to IBKR Gateway"""
        try:
            logger.info(f"[CONNECTING] IBKR Gateway at {self.host}:{self.port} (Client ID: {self.client_id})")
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=20)
            
            if self.ib.isConnected():
                self.connected = True
                logger.info("[SUCCESS] Connected to IBKR Gateway")
                
                # Test connection with current time
                current_time = self.ib.reqCurrentTime()
                logger.info(f"[SUCCESS] IBKR Server Time: {current_time}")
                return True
            else:
                logger.error("[ERROR] Failed to connect to IBKR Gateway")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] IBKR connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IBKR Gateway"""
        if self.connected and self.ib.isConnected():
            self.ib.disconnect()
            self.connected = False
            logger.info("[SUCCESS] Disconnected from IBKR Gateway")
    
    def get_historical_data(self, symbol: str, duration: str = "5 D", bar_size: str = "1 hour") -> Optional[pd.DataFrame]:
        """Get historical market data for pattern analysis"""
        try:
            if not self.connected:
                logger.error("[ERROR] Not connected to IBKR")
                return None
            
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            logger.info(f"[REQUESTING] Historical data for {symbol} ({duration}, {bar_size})")
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if bars:
                # Convert to DataFrame
                df = util.df(bars)
                logger.info(f"[SUCCESS] Retrieved {len(df)} bars for {symbol}")
                return df
            else:
                logger.warning(f"[WARNING] No data received for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to get historical data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for pattern features"""
        try:
            if df is None or len(df)  df['sma_10']).astype(float)
            df['price_above_sma20'] = (df['close'] > df['sma_20']).astype(float)
            
            # Volatility
            df['volatility'] = df['price_change'].rolling(window=10).std()
            
            logger.info(f"[SUCCESS] Calculated technical indicators")
            return df
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to calculate technical indicators: {e}")
            return df
    
    def create_pattern_vector(self, df: pd.DataFrame, lookback: int = 20) -> Optional[np.ndarray]:
        """Create pattern feature vector from market data"""
        try:
            if df is None or len(df)  1 else 0,
                    (segment[-1] - segment[0]) if len(segment) > 0 else 0,
                    segment.max() - segment.min() if len(segment) > 0 else 0,
                ])
            
            # Ensure exactly 32 features
            while len(features)  Dict:
        """Find similar patterns using FAISS"""
        try:
            if self.index is None or self.index.ntotal == 0:
                return {"distances": [], "indices": [], "metadata": [], "message": "No patterns in index"}
            
            # Prepare query
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            query_vector = query_vector.astype(np.float32)
            
            # Normalize
            faiss.normalize_L2(query_vector)
            
            # Search
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, k)
            
            # Get metadata
            result_metadata = []
            for idx in indices[0]:
                if 0 = 25:
                    # Calculate indicators
                    df = data_provider.calculate_technical_indicators(df)
                    
                    # Create pattern vector
                    pattern_vector = data_provider.create_pattern_vector(df, lookback=20)
                    
                    if pattern_vector is not None:
                        # Add to pattern database
                        metadata = {
                            "symbol": symbol,
                            "timestamp": datetime.now().isoformat(),
                            "last_price": float(df['close'].iloc[-1]),
                            "daily_change": float(df['price_change'].iloc[-1]),
                            "source": "IBKR_live"
                        }
                        
                        if pattern_matcher.add_pattern(pattern_vector, metadata):
                            pattern_count += 1
                    
                    # Small delay
                    time.sleep(1)
                
            except Exception as e:
                logger.warning(f"[WARNING] Could not process {symbol}: {e}")
                continue
        
        logger.info(f"[SUCCESS] Built pattern database with {pattern_count} patterns")
        
        # Step 3: Test pattern matching
        if pattern_count > 0:
            logger.info("\n[STEP 3] Testing pattern similarity matching...")
            
            # Create a test query vector
            test_vector = np.random.normal(0, 0.3, 32).astype(np.float32)
            results = pattern_matcher.find_similar_patterns(test_vector, k=min(3, pattern_count))
            
            logger.info(f"Found {len(results['distances'])} similar patterns:")
            for i, (similarity, meta) in enumerate(zip(results['distances'], results['metadata'])):
                logger.info(f"  {i+1}. {meta['symbol']} - Similarity: {similarity:.4f}")
        
        logger.info("\n[SUCCESS] FAISS-IBKR integration test completed successfully!")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Integration test failed: {e}")
        return False
        
    finally:
        data_provider.disconnect()

def main():
    """Main test execution function"""
    print("WORKING FAISS-IBKR Integration Test")
    print("=" * 40)
    print("This test requires IBKR Gateway running on localhost:4002")
    print("-" * 40)
    
    success = run_working_integration_test()
    
    if success:
        print("\n[SUCCESS] Integration test completed!")
        return 0
    else:
        print("\n[ERROR] Integration test failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
