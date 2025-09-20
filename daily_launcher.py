#!/usr/bin/env python3
"""
Daily AI Trading System Launcher

This script provides automated execution of daily trading system operations
with minimal user intervention. Designed for consistent daily workflow execution.

Author: AI Trading System Team
Created: September 20, 2025
"""

import sys
import os
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import time

# Configure logging with Windows-compatible encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'daily_operations_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class DailyOperationsLauncher:
    """Main launcher for daily AI trading operations"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.ai_engines_path = self.base_path.parent / "ai_engines"
        self.tensortrade_path = self.base_path / "tensortrade"
        
        # Operational status tracking
        self.operations_status = {
            'database_health': False,
            'ibkr_connection': False,
            'data_collection': False,
            'signal_generation': False,
            'performance_report': False
        }
        
        logger.info(f"[INIT] Daily Operations Launcher initialized")
        logger.info(f"   Base path: {self.base_path}")
        logger.info(f"   AI Engines: {self.ai_engines_path}")
        logger.info(f"   TensorTrade: {self.tensortrade_path}")
    
    def check_database_health(self) -> bool:
        """Check database connection health"""
        logger.info("[HEALTH] Checking database connection...")
        
        try:
            # Import and test database connection using correct path
            sys.path.append(str(self.base_path))
            
            # Try multiple possible import paths
            try:
                from modules.database.railway_db_manager import RailwayPostgreSQLManager
            except ImportError:
                try:
                    sys.path.append(str(self.base_path.parent / "TradeAppComponents_fresh"))
                    from modules.database.railway_db_manager import RailwayPostgreSQLManager
                except ImportError:
                    # Fallback to direct database connection
                    import psycopg2
                    DATABASE_URL = "postgresql://postgres:TAqEkujnMknVURCcrYTIDOzQXbgBNtSX@turntable.proxy.rlwy.net:10410/railway"
                    conn = psycopg2.connect(DATABASE_URL)
                    conn.close()
                    logger.info("[HEALTH] Database connection: SUCCESS (direct connection)")
                    self.operations_status['database_health'] = True
                    return True
            
            db = RailwayPostgreSQLManager()
            session = db.get_session()
            session.close()
            
            logger.info("[HEALTH] Database connection: SUCCESS")
            self.operations_status['database_health'] = True
            return True
            
        except Exception as e:
            logger.error(f"[HEALTH] Database connection: FAILED - {e}")
            return False
    
    def check_ibkr_connection(self) -> bool:
        """Check IBKR Gateway connection"""
        logger.info("[HEALTH] Checking IBKR Gateway connection...")
        
        try:
            from ib_insync import IB
            
            ib = IB()
            ib.connect('127.0.0.1', 4002, clientId=999, timeout=10)
            
            if ib.isConnected():
                server_version = ib.client.serverVersion()
                logger.info(f"[HEALTH] IBKR Gateway: CONNECTED (version {server_version})")
                ib.disconnect()
                self.operations_status['ibkr_connection'] = True
                return True
            else:
                logger.error("[HEALTH] IBKR Gateway: CONNECTION FAILED")
                return False
                
        except Exception as e:
            logger.error(f"[HEALTH] IBKR Gateway: ERROR - {e}")
            return False
    
    def load_ibd50_stocks(self) -> Tuple[bool, List[str]]:
        """Load IBD 50 stock universe"""
        logger.info("[STOCKS] Loading IBD 50 stock universe...")
        
        try:
            sys.path.append(str(self.base_path))
            from ibd50_database_manager import IBD50DatabaseManager
            
            manager = IBD50DatabaseManager()
            stocks_result = manager.get_ibd50_stocks(as_dataframe=False)  # Ensure we get a list
            
            # Ensure stocks is a list of strings
            if isinstance(stocks_result, list):
                stocks = stocks_result
            else:
                stocks = stocks_result.tolist() if hasattr(stocks_result, 'tolist') else list(stocks_result)
            
            # Get sector breakdown using stocks by sector method
            sector_breakdown = {}
            try:
                stocks_df = manager.get_stocks_by_sector()
                if not stocks_df.empty and 'sector' in stocks_df.columns and 'stock_count' in stocks_df.columns:
                    sector_breakdown = dict(zip(stocks_df['sector'], stocks_df['stock_count']))
                else:
                    # Fallback: count sectors from the main stocks DataFrame
                    stocks_df = manager.get_ibd50_stocks(as_dataframe=True)
                    if not stocks_df.empty and 'sector' in stocks_df.columns:
                        sector_breakdown = stocks_df['sector'].value_counts().to_dict()
            except Exception as e:
                logger.warning(f"[STOCKS] Could not get sector breakdown: {e}")
                sector_breakdown = {"Unknown": len(stocks)}
            
            logger.info(f"[STOCKS] Loaded {len(stocks)} IBD 50 stocks")
            for sector, count in sector_breakdown.items():
                logger.info(f"   {sector}: {count} stocks")
            
            return True, stocks
            
        except Exception as e:
            logger.error(f"[STOCKS] Failed to load IBD 50 stocks: {e}")
            return False, []
    
    def run_data_collection(self) -> bool:
        """Execute data collection process"""
        logger.info("[DATA] Starting historical data collection...")
        
        try:
            # Change to data collector directory
            collector_path = self.ai_engines_path / "ai_market_data_collector"
            collector_script = collector_path / "ai_market_data_collector.py"
            
            if not collector_script.exists():
                logger.error(f"[DATA] Data collector not found: {collector_script}")
                return False
            
            # Run data collector
            os.chdir(collector_path)
            result = subprocess.run([
                sys.executable, "ai_market_data_collector.py", "--once"
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                logger.info("[DATA] Data collection: SUCCESS")
                logger.info(f"   Output: {result.stdout[-200:]}")  # Last 200 chars
                self.operations_status['data_collection'] = True
                return True
            else:
                logger.error(f"[DATA] Data collection: FAILED - {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("[DATA] Data collection: TIMEOUT (10 minutes)")
            return False
        except Exception as e:
            logger.error(f"[DATA] Data collection: ERROR - {e}")
            return False
        finally:
            # Return to base directory
            os.chdir(self.base_path)
    
    def generate_trading_signals(self, stocks: List[str]) -> bool:
        """Generate trading signals for stock universe"""
        logger.info(f"[SIGNALS] Generating trading signals for {len(stocks)} stocks...")
        
        try:
            # Import signal generation components
            sys.path.append(str(self.tensortrade_path))
            from src.signal_bridge import TensorTradeSignalBridge
            from modules.database.railway_db_manager import RailwayPostgreSQLManager
            import numpy as np
            
            # Initialize signal bridge
            db_manager = RailwayPostgreSQLManager()
            bridge = TensorTradeSignalBridge(db_manager)
            
            # Generate signals for first 10 stocks (demo)
            signals_generated = []
            for symbol in stocks[:10]:
                try:
                    # Simulate RL model prediction (replace with actual model)
                    action_probs = np.random.dirichlet([1, 2, 1])  # [Hold, Buy, Sell]
                    
                    rl_signals = bridge.export_rl_signals(
                        symbols=[symbol],
                        actions=[action_probs],
                        current_prices={symbol: 150.0}  # Would get from real-time data
                    )
                    
                    if rl_signals:
                        signal = rl_signals[0]
                        signals_generated.append(signal)
                        
                        action = signal.signal_type.value.replace('_', ' ').title()
                        logger.info(f"   {symbol}: {action} (conf: {signal.confidence:.2f}, str: {signal.strength:.2f})")
                
                except Exception as e:
                    logger.warning(f"   {symbol}: Signal generation failed - {str(e)[:50]}")
            
            # Summary
            signal_summary = {}
            for signal in signals_generated:
                signal_type = signal.signal_type.value
                signal_summary[signal_type] = signal_summary.get(signal_type, 0) + 1
            
            logger.info(f"[SIGNALS] Generated {len(signals_generated)} signals:")
            for signal_type, count in signal_summary.items():
                action = signal_type.replace('_', ' ').title()
                logger.info(f"   {action}: {count} signals")
            
            self.operations_status['signal_generation'] = True
            return True
            
        except Exception as e:
            logger.error(f"[SIGNALS] Signal generation failed: {e}")
            return False
    
    def generate_performance_report(self) -> bool:
        """Generate daily performance report"""
        logger.info("[REPORT] Generating daily performance report...")
        
        try:
            from modules.database.railway_db_manager import RailwayPostgreSQLManager
            from sqlalchemy import text
            
            db_manager = RailwayPostgreSQLManager()
            session = db_manager.get_session()
            
            # Get data quality metrics
            today = datetime.now().date()
            result = session.execute(text('''
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT symbol) as symbols_with_data
                FROM ai_historical_market_data 
                WHERE DATE(timestamp) = :today AND timeframe = '1min'
            '''), {'today': today})
            
            data_metrics = result.fetchone()
            total_records = data_metrics[0] if data_metrics else 0
            symbols_count = data_metrics[1] if data_metrics else 0
            
            # Create report
            report = {
                'date': today.isoformat(),
                'timestamp': datetime.now().isoformat(),
                'operations_status': self.operations_status,
                'data_metrics': {
                    'total_records': total_records,
                    'symbols_with_data': symbols_count,
                    'data_quality': 'Good' if total_records > 10000 else 'Low'
                },
                'system_health': all(self.operations_status.values())
            }
            
            # Save report
            reports_dir = self.base_path / "daily_reports"
            reports_dir.mkdir(exist_ok=True)
            
            report_file = reports_dir / f"daily_report_{today.strftime('%Y%m%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"[REPORT] Performance report: SAVED to {report_file}")
            logger.info(f"   Total records: {total_records:,}")
            logger.info(f"   Symbols: {symbols_count}")
            logger.info(f"   System health: {'HEALTHY' if report['system_health'] else 'ISSUES'}")
            
            session.close()
            self.operations_status['performance_report'] = True
            return True
            
        except Exception as e:
            logger.error(f"[REPORT] Performance report failed: {e}")
            return False
    
    def run_pre_market_routine(self) -> bool:
        """Execute complete pre-market routine"""
        logger.info("STARTING PRE-MARKET ROUTINE")
        logger.info("=" * 50)
        
        success = True
        
        # Step 1: Health checks
        if not self.check_database_health():
            success = False
        
        if not self.check_ibkr_connection():
            success = False
        
        # Step 2: Load stock universe
        stocks_loaded, stocks = self.load_ibd50_stocks()
        if not stocks_loaded:
            success = False
        
        # Step 3: Data collection
        if not self.run_data_collection():
            success = False
        
        if success:
            logger.info("PRE-MARKET ROUTINE: COMPLETED SUCCESSFULLY")
        else:
            logger.error("PRE-MARKET ROUTINE: COMPLETED WITH ERRORS")
        
        return success
    
    def run_market_hours_routine(self) -> bool:
        """Execute market hours signal generation"""
        logger.info("STARTING MARKET HOURS ROUTINE")
        logger.info("=" * 50)
        
        # Load stocks
        stocks_loaded, stocks = self.load_ibd50_stocks()
        if not stocks_loaded:
            logger.error("MARKET HOURS: Cannot load stock universe")
            return False
        
        # Generate signals
        if self.generate_trading_signals(stocks):
            logger.info("MARKET HOURS ROUTINE: COMPLETED SUCCESSFULLY")
            return True
        else:
            logger.error("MARKET HOURS ROUTINE: SIGNAL GENERATION FAILED")
            return False
    
    def run_end_of_day_routine(self) -> bool:
        """Execute end-of-day analysis"""
        logger.info("STARTING END-OF-DAY ROUTINE")
        logger.info("=" * 50)
        
        success = True
        
        # Final data collection
        if not self.run_data_collection():
            success = False
        
        # Generate performance report
        if not self.generate_performance_report():
            success = False
        
        if success:
            logger.info("END-OF-DAY ROUTINE: COMPLETED SUCCESSFULLY")
        else:
            logger.error("END-OF-DAY ROUTINE: COMPLETED WITH ERRORS")
        
        return success
    
    def run_weekend_analysis(self) -> bool:
        """Execute weekend comprehensive analysis"""
        logger.info("STARTING WEEKEND ANALYSIS")
        logger.info("=" * 50)
        
        try:
            # Run weekend AI tester
            weekend_script = self.base_path / "weekend_ai_tester.py"
            
            if not weekend_script.exists():
                logger.error(f"[WEEKEND] Weekend AI tester not found: {weekend_script}")
                return False
            
            result = subprocess.run([
                sys.executable, str(weekend_script)
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                logger.info("WEEKEND ANALYSIS: COMPLETED SUCCESSFULLY")
                logger.info(f"   Output summary: {result.stdout[-300:]}")  # Last 300 chars
                return True
            else:
                logger.error(f"WEEKEND ANALYSIS: FAILED - {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("WEEKEND ANALYSIS: TIMEOUT (1 hour)")
            return False
        except Exception as e:
            logger.error(f"WEEKEND ANALYSIS: ERROR - {e}")
            return False

def main():
    """Main entry point with command-line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Daily AI Trading System Operations Launcher')
    parser.add_argument('--pre-market', action='store_true', help='Run pre-market routine')
    parser.add_argument('--market-hours', action='store_true', help='Run market hours signal generation')
    parser.add_argument('--end-of-day', action='store_true', help='Run end-of-day analysis')
    parser.add_argument('--weekend', action='store_true', help='Run weekend comprehensive analysis')
    parser.add_argument('--all', action='store_true', help='Run complete daily sequence')
    parser.add_argument('--health-check', action='store_true', help='Run system health checks only')
    
    args = parser.parse_args()
    
    launcher = DailyOperationsLauncher()
    
    print("AI Trading System - Daily Operations Launcher")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = True
    
    if args.health_check:
        print("Running health checks...")
        launcher.check_database_health()
        launcher.check_ibkr_connection()
        launcher.load_ibd50_stocks()
        
    elif args.pre_market:
        success = launcher.run_pre_market_routine()
        
    elif args.market_hours:
        success = launcher.run_market_hours_routine()
        
    elif args.end_of_day:
        success = launcher.run_end_of_day_routine()
        
    elif args.weekend:
        success = launcher.run_weekend_analysis()
        
    elif args.all:
        # Run complete daily sequence
        print("Running complete daily sequence...")
        
        # Pre-market
        success &= launcher.run_pre_market_routine()
        print()
        
        # Market hours (simulate multiple times)
        times = ["10:00", "12:00", "14:00"]
        for time_str in times:
            print(f"Market hours signal generation ({time_str})...")
            success &= launcher.run_market_hours_routine()
            print()
        
        # End of day
        success &= launcher.run_end_of_day_routine()
        
    else:
        # Default: show usage
        parser.print_help()
        return 1
    
    print()
    print("=" * 60)
    if success:
        print("ALL OPERATIONS COMPLETED SUCCESSFULLY")
        return 0
    else:
        print("SOME OPERATIONS FAILED - CHECK LOGS")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)