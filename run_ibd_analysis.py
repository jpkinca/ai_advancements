"""
Run Sweet Spot & Danger Zone System on IBD Stocks

Processes all IBD stocks from ai_stock_universe, runs the AI trading system,
and generates performance reports.
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
from pathlib import Path
import json
from datetime import datetime

# Import our AI trading system
from sweet_spot_danger_zone_system import create_dual_signal_system

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class IBDStocksProcessor:
    """Process IBD stocks with Sweet Spot & Danger Zone system"""

    def __init__(self):
        self.database_url = "postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway"
        self.results_dir = Path("ibd_analysis_results")
        self.results_dir.mkdir(exist_ok=True)

    async def connect_to_database(self) -> asyncpg.Connection:
        """Connect to Railway PostgreSQL database"""
        try:
            conn = await asyncpg.connect(self.database_url)
            logger.info("[SUCCESS] Connected to Railway PostgreSQL database")
            return conn
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect: {e}")
            raise

    async def get_ibd_stocks(self, conn: asyncpg.Connection) -> List[str]:
        """Get list of active stocks from ai_historical_market_data"""
        try:
            stocks = await conn.fetch("""
                SELECT DISTINCT symbol FROM ai_historical_market_data
                ORDER BY symbol
            """)

            stock_list = [row['symbol'] for row in stocks]
            logger.info(f"[INFO] Found {len(stock_list)} stocks with historical data")
            return stock_list

        except Exception as e:
            logger.error(f"[ERROR] Failed to get stocks: {e}")
            return []

    async def load_stock_data(self, conn: asyncpg.Connection, symbol: str, limit: int = 500) -> pd.DataFrame:
        """Load historical data for a stock"""
        try:
            # Try different timeframes in order of preference
            timeframes = ['1week', '1month', '1min', '5min']

            for timeframe in timeframes:
                query = """
                    SELECT
                        timestamp,
                        open_price as open,
                        high_price as high,
                        low_price as low,
                        close_price as close,
                        volume
                    FROM ai_historical_market_data
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                """

                rows = await conn.fetch(query, symbol, timeframe, limit)

                if rows:
                    # Convert to DataFrame
                    df = pd.DataFrame([dict(row) for row in rows])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp').sort_index()

                    # Convert price columns to float (from Decimal)
                    price_columns = ['open', 'high', 'low', 'close']
                    for col in price_columns:
                        if col in df.columns:
                            df[col] = df[col].astype(float)

                    # Convert volume to float if needed
                    if 'volume' in df.columns:
                        df['volume'] = df['volume'].astype(float)

                    logger.info(f"[INFO] Loaded {len(df)} bars for {symbol} ({timeframe})")
                    return df

            # If no data found in any timeframe
            logger.warning(f"[WARNING] No data found for {symbol} in any timeframe")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"[ERROR] Failed to load data for {symbol}: {e}")
            return pd.DataFrame()

    async def process_single_stock(self, symbol: str, conn: asyncpg.Connection) -> Dict[str, Any]:
        """Process a single stock with the AI system"""
        logger.info(f"[PROCESSING] {symbol}")

        try:
            # Load data (method now tries multiple timeframes automatically)
            df = await self.load_stock_data(conn, symbol, 500)

            if df.empty:
                return {
                    'symbol': symbol,
                    'status': 'no_data',
                    'error': 'No historical data available'
                }

            # Ensure we have enough data
            if len(df) < 100:
                return {
                    'symbol': symbol,
                    'status': 'insufficient_data',
                    'bars': len(df),
                    'error': f'Only {len(df)} bars available, need minimum 100'
                }

            # Create and train the system
            system = create_dual_signal_system()

            # Train (with error handling)
            try:
                metrics = system.train(df)
            except Exception as e:
                return {
                    'symbol': symbol,
                    'status': 'training_failed',
                    'bars': len(df),
                    'error': f'Training failed: {str(e)}'
                }

            # Generate signals
            try:
                signals = system.generate_signals(df)
            except Exception as e:
                return {
                    'symbol': symbol,
                    'status': 'signals_failed',
                    'bars': len(df),
                    'error': f'Signal generation failed: {str(e)}'
                }

            # Run backtest
            try:
                backtest_results = system.backtest(df, initial_capital=10000)
            except Exception as e:
                return {
                    'symbol': symbol,
                    'status': 'backtest_failed',
                    'bars': len(df),
                    'error': f'Backtest failed: {str(e)}'
                }

            # Compile results
            result = {
                'symbol': symbol,
                'status': 'success',
                'bars': len(df),
                'date_range': {
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat()
                },
                'training_metrics': metrics,
                'backtest_results': {
                    'total_return': backtest_results['total_return'],
                    'annualized_return': backtest_results['annualized_return'],
                    'sharpe_ratio': backtest_results['sharpe_ratio'],
                    'max_drawdown': backtest_results['max_drawdown'],
                    'win_rate': backtest_results['win_rate'],
                    'total_trades': backtest_results['total_trades']
                },
                'recent_signals': signals.tail(5)[['combined_signal', 'position_size', 'confidence_score']].to_dict('records'),
                'feature_importance': system.get_feature_importance()
            }

            logger.info(f"[SUCCESS] {symbol}: Return {backtest_results['total_return']:.1%}, Sharpe {backtest_results['sharpe_ratio']:.2f}")
            return result

        except Exception as e:
            logger.error(f"[ERROR] Failed to process {symbol}: {e}")
            return {
                'symbol': symbol,
                'status': 'error',
                'error': str(e)
            }

    async def process_all_stocks(self, max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Process all IBD stocks"""
        logger.info("Starting IBD stocks analysis...")

        conn = await self.connect_to_database()
        ibd_stocks = await self.get_ibd_stocks(conn)

        if not ibd_stocks:
            logger.error("[ERROR] No IBD stocks found")
            return []

        logger.info(f"[INFO] Processing {len(ibd_stocks)} IBD stocks (max {max_concurrent} concurrent)")

        # Process stocks (sequentially for now to avoid memory issues)
        results = []
        successful = 0
        failed = 0

        for i, symbol in enumerate(ibd_stocks, 1):
            logger.info(f"[PROGRESS] Processing {i}/{len(ibd_stocks)}: {symbol}")

            result = await self.process_single_stock(symbol, conn)
            results.append(result)

            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1

        await conn.close()

        logger.info(f"[SUMMARY] Processed {len(results)} stocks: {successful} successful, {failed} failed")
        return results

    def save_results(self, results: List[Dict[str, Any]]):
        """Save analysis results to files"""

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"ibd_analysis_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Create summary report
        successful_results = [r for r in results if r['status'] == 'success']

        if successful_results:
            summary = {
                'analysis_timestamp': timestamp,
                'total_stocks': len(results),
                'successful_stocks': len(successful_results),
                'failed_stocks': len(results) - len(successful_results),
                'performance_summary': {
                    'avg_total_return': np.mean([r['backtest_results']['total_return'] for r in successful_results]),
                    'avg_sharpe_ratio': np.mean([r['backtest_results']['sharpe_ratio'] for r in successful_results]),
                    'avg_win_rate': np.mean([r['backtest_results']['win_rate'] for r in successful_results]),
                    'best_performer': max(successful_results, key=lambda x: x['backtest_results']['total_return'])['symbol'],
                    'worst_performer': min(successful_results, key=lambda x: x['backtest_results']['total_return'])['symbol']
                },
                'top_performers': sorted(
                    [(r['symbol'], r['backtest_results']['total_return']) for r in successful_results],
                    key=lambda x: x[1], reverse=True
                )[:10]
            }

            summary_file = self.results_dir / f"ibd_analysis_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            # Print summary
            print("\n" + "="*80)
            print("IBD STOCKS ANALYSIS SUMMARY")
            print("="*80)
            print(f"Total Stocks: {summary['total_stocks']}")
            print(f"Successful: {summary['successful_stocks']}")
            print(f"Failed: {summary['failed_stocks']}")
            print(".1%")
            print(".2f")
            print(".1%")
            print(f"Best Performer: {summary['performance_summary']['best_performer']}")
            print(f"Worst Performer: {summary['performance_summary']['worst_performer']}")
            print("\nTop 10 Performers:")
            for i, (symbol, ret) in enumerate(summary['top_performers'][:10], 1):
                print(".1%")

        logger.info(f"[SAVED] Results saved to {self.results_dir}")

    async def run_analysis(self):
        """Run the complete IBD stocks analysis"""

        logger.info("Starting Sweet Spot & Danger Zone analysis on IBD stocks...")

        try:
            results = await self.process_all_stocks()

            if results:
                self.save_results(results)
                logger.info("[FINAL RESULT] IBD analysis completed successfully")
                return True
            else:
                logger.error("[FINAL RESULT] No results generated")
                return False

        except Exception as e:
            logger.error(f"[ERROR] Analysis failed: {e}")
            return False

async def main():
    """Run IBD stocks analysis"""

    processor = IBDStocksProcessor()
    success = await processor.run_analysis()

    if success:
        logger.info("[COMPLETE] IBD stocks analysis finished. Check results in 'ibd_analysis_results/'")
        return 0
    else:
        logger.info("[FAILED] Analysis failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())