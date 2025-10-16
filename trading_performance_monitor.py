#!/usr/bin/env python3
"""
Trading Performance Monitor
Analyze minimal viable trading system results
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingPerformanceMonitor:
    """Monitor and analyze trading system performance"""

    def __init__(self, db_path='minimal_trading_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def get_trading_summary(self, days_back=7):
        """Get comprehensive trading performance summary"""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        # Get trades
        trades_query = f"""
        SELECT * FROM live_trades
        WHERE timestamp >= '{cutoff_date}'
        ORDER BY timestamp DESC
        """
        trades_df = pd.read_sql_query(trades_query, self.conn)

        # Get signals
        signals_query = f"""
        SELECT * FROM market_signals
        WHERE timestamp >= '{cutoff_date}'
        ORDER BY timestamp DESC
        """
        signals_df = pd.read_sql_query(signals_query, self.conn)

        return trades_df, signals_df

    def calculate_performance_metrics(self, trades_df, signals_df):
        """Calculate key performance metrics"""
        metrics = {}

        if len(trades_df) == 0:
            return {"error": "No trades found"}

        # Basic counts
        metrics['total_trades'] = len(trades_df)
        metrics['total_signals'] = len(signals_df)
        metrics['signal_to_trade_ratio'] = len(trades_df) / len(signals_df) if len(signals_df) > 0 else 0

        # Win/Loss analysis (simplified - would need actual P&L data)
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']

        metrics['buy_trades'] = len(buy_trades)
        metrics['sell_trades'] = len(sell_trades)

        # Signal quality
        if len(signals_df) > 0:
            avg_confidence = signals_df['confidence'].mean()
            high_conf_signals = len(signals_df[signals_df['confidence'] > 0.7])
            metrics['avg_signal_confidence'] = avg_confidence
            metrics['high_conf_signals'] = high_conf_signals
            metrics['high_conf_ratio'] = high_conf_signals / len(signals_df)

        # Volume analysis
        if 'volume' in signals_df.columns:
            avg_volume = signals_df['volume'].mean()
            metrics['avg_signal_volume'] = avg_volume

        return metrics

    def generate_report(self, days_back=7):
        """Generate comprehensive performance report"""
        logger.info(f"📊 GENERATING PERFORMANCE REPORT (Last {days_back} days)")
        logger.info("="*60)

        try:
            trades_df, signals_df = self.get_trading_summary(days_back)
            metrics = self.calculate_performance_metrics(trades_df, signals_df)

            if "error" in metrics:
                logger.warning("No trading data available")
                return

            print(f"📈 TRADING PERFORMANCE SUMMARY")
            print(f"Period: Last {days_back} days")
            print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 40)

            print(f"📊 Volume Metrics:")
            print(f"  Signals Generated: {metrics['total_signals']}")
            print(f"  Trades Executed: {metrics['total_trades']}")
            print(f"  Signal-to-Trade Ratio: {metrics['signal_to_trade_ratio']:.1%}")

            print(f"\n🎯 Signal Quality:")
            if 'avg_signal_confidence' in metrics:
                print(f"  Average Confidence: {metrics['avg_signal_confidence']:.2f}")
                print(f"  High Confidence Signals (>70%): {metrics['high_conf_signals']}")
                print(f"  High Confidence Ratio: {metrics['high_conf_ratio']:.1%}")

            print(f"\n📈 Trade Direction:")
            print(f"  Buy Trades: {metrics['buy_trades']}")
            print(f"  Sell Trades: {metrics['sell_trades']}")

            if len(trades_df) > 0:
                print(f"\n📋 Recent Trades:")
                recent_trades = trades_df.head(5)
                for _, trade in recent_trades.iterrows():
                    print(f"  {trade['timestamp'][:19]} | {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")

            # Performance assessment
            print(f"\n🎖️ PERFORMANCE ASSESSMENT:")
            signal_ratio = metrics['signal_to_trade_ratio']

            if signal_ratio > 0.6:
                print("  ✅ Signal Quality: EXCELLENT (>60% conversion)")
            elif signal_ratio > 0.4:
                print("  ⚠️ Signal Quality: GOOD (40-60% conversion)")
            elif signal_ratio > 0.2:
                print("  📊 Signal Quality: MODERATE (20-40% conversion)")
            else:
                print("  ❌ Signal Quality: NEEDS IMPROVEMENT (<20% conversion)")

            if 'high_conf_ratio' in metrics and metrics['high_conf_ratio'] > 0.5:
                print("  ✅ Confidence Levels: STRONG (>50% high confidence)")
            elif 'high_conf_ratio' in metrics:
                print("  ⚠️ Confidence Levels: MODERATE (needs threshold tuning)")

            print(f"\n💡 RECOMMENDATIONS:")
            if signal_ratio < 0.3:
                print("  • Increase signal confidence threshold from 40% to 50%")
                print("  • Review momentum calculation parameters")
            if metrics['total_signals'] < 10:
                print("  • Run longer trading sessions for better statistics")
                print("  • Consider adding more symbols or reducing filters")

            print(f"\n🎯 NEXT STEPS:")
            print("  1. Monitor win rate and Sharpe ratio (need P&L data)")
            print("  2. Adjust signal parameters based on performance")
            print("  3. Gradually increase position sizes")
            print("  4. Integrate Level II data for better signals")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")

    def export_data(self, filename=None):
        """Export trading data for external analysis"""
        if not filename:
            filename = f"trading_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        try:
            trades_df, signals_df = self.get_trading_summary(days_back=30)

            # Combine data
            combined_data = {
                'trades': trades_df.to_dict('records'),
                'signals': signals_df.to_dict('records')
            }

            # Save as JSON for easy analysis
            import json
            json_filename = filename.replace('.csv', '.json')
            with open(json_filename, 'w') as f:
                json.dump(combined_data, f, indent=2, default=str)

            logger.info(f"Data exported to: {json_filename}")
            return json_filename

        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return None


def main():
    """Main performance monitoring function"""
    print("📊 TRADING PERFORMANCE MONITOR")
    print("="*40)

    monitor = TradingPerformanceMonitor()

    # Generate performance report
    monitor.generate_report(days_back=7)

    # Export data for analysis
    print(f"\n💾 Exporting data for analysis...")
    export_file = monitor.export_data()
    if export_file:
        print(f"✅ Data exported to: {export_file}")

    monitor.conn.close()


if __name__ == "__main__":
    main()