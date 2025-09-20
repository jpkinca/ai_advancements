"""
test_pnl_tracker.py
==================
Test script for the new P&L tracking functionality.

This script demonstrates the trade-level P&L tracking capabilities
by simulating some basic trading actions.
"""

from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from db_utils import get_engine
from trade_pnl_tracker import TradePnLTracker
import argparse


def test_pnl_tracking():
    """Test the P&L tracking with simulated trades"""
    
    # Get database connection
    db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/tensortrade')
    engine = get_engine(db_url)
    
    # Create test episode
    episode_id = 999999  # Test episode ID
    
    # Initialize P&L tracker
    tracker = TradePnLTracker(engine, episode_id)
    
    print("🧪 Testing TensorTrade P&L Tracking System")
    print("=" * 60)
    
    # Test 1: Simple long position
    print("\n📈 Test 1: Opening long position in AAPL")
    tracker.process_action(
        symbol="AAPL",
        new_weight=0.5,  # 50% of portfolio
        current_price=150.0,
        timestamp=datetime.now(),
        portfolio_value=100000.0
    )
    
    # Test 2: Increase position
    print("📈 Test 2: Increasing AAPL position")
    tracker.process_action(
        symbol="AAPL",
        new_weight=0.8,  # Increase to 80%
        current_price=152.0,
        timestamp=datetime.now() + timedelta(minutes=30),
        portfolio_value=100000.0
    )
    
    # Test 3: Add different stock
    print("📈 Test 3: Opening position in MSFT")
    tracker.process_action(
        symbol="MSFT",
        new_weight=0.3,  # 30% of portfolio
        current_price=300.0,
        timestamp=datetime.now() + timedelta(hours=1),
        portfolio_value=100000.0
    )
    
    # Test 4: Partial close AAPL
    print("📉 Test 4: Partially closing AAPL position")
    tracker.process_action(
        symbol="AAPL",
        new_weight=0.3,  # Reduce to 30%
        current_price=155.0,  # Profitable exit
        timestamp=datetime.now() + timedelta(hours=2),
        portfolio_value=100000.0
    )
    
    # Test 5: Close MSFT at loss
    print("📉 Test 5: Closing MSFT position at loss")
    tracker.process_action(
        symbol="MSFT",
        new_weight=0.0,  # Full exit
        current_price=295.0,  # Losing trade
        timestamp=datetime.now() + timedelta(hours=3),
        portfolio_value=100000.0
    )
    
    # Test 6: Get current P&L
    print("\n💰 Current P&L Status:")
    current_prices = {"AAPL": 158.0, "MSFT": 295.0}
    current_pnl = tracker.get_current_pnl(current_prices)
    
    for symbol, pnl in current_pnl.items():
        print(f"   {symbol}: ${pnl:,.2f}")
    
    # Test 7: Generate full P&L report
    print("\n📊 Generating Full P&L Report:")
    tracker.print_pnl_report()
    
    # Test 8: Close remaining position
    print("\n📉 Test 8: Closing remaining AAPL position")
    tracker.process_action(
        symbol="AAPL",
        new_weight=0.0,  # Full exit
        current_price=158.0,  # Final profitable exit
        timestamp=datetime.now() + timedelta(hours=4),
        portfolio_value=100000.0
    )
    
    # Final report
    print("\n🏁 Final P&L Report after all trades closed:")
    tracker.print_pnl_report()
    
    print("\n✅ P&L Tracking Test Complete!")
    print("   Check the database tables 'tt_trades' and 'tt_portfolio_snapshots'")
    print("   for persistent trade records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test P&L tracking system")
    parser.add_argument("--db-url", help="Database URL", 
                       default=os.getenv('DATABASE_URL'))
    args = parser.parse_args()
    
    if args.db_url:
        os.environ['DATABASE_URL'] = args.db_url
    
    test_pnl_tracking()
