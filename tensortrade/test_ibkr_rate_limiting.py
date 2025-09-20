#!/usr/bin/env python3
"""
test_ibkr_rate_limiting.py

Test script to verify IBKR rate limiting and connection management improvements.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from watchlist_loader import fetch_price_history_ibkr, load_watchlist
    print("✅ Successfully imported enhanced watchlist_loader")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_rate_limiting():
    """Test the rate limiting and connection management features."""
    print("🧪 Testing IBKR Rate Limiting and Connection Management")
    print("=" * 60)
    
    # Load a small subset for testing
    try:
        symbols = load_watchlist()[:5]  # Just first 5 symbols
        print(f"📋 Testing with symbols: {symbols}")
        
        # Test the enhanced fetch function
        print(f"\n📡 Fetching 30 days of data with rate limiting...")
        data = fetch_price_history_ibkr(
            symbols=symbols,
            start="2024-07-01",
            end="2024-07-31", 
            interval="1d",
            rate_limit_delay=0.2,  # 200ms delay for testing
            batch_size=3,          # Small batch for testing
            retry_attempts=2       # Fewer retries for testing
        )
        
        print(f"✅ Successfully fetched {len(data)} bars")
        print(f"📊 Data shape: {data.shape}")
        print(f"🏷️ Symbols in result: {sorted(data['symbol'].unique())}")
        
        # Show sample data
        print(f"\n📈 Sample data:")
        print(data.head(10).to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rate_limiting()
    if success:
        print(f"\n🎉 Rate limiting test completed successfully!")
    else:
        print(f"\n💥 Rate limiting test failed!")
        sys.exit(1)
