#!/usr/bin/env python3
"""
XGBoost Backtesting Script

This script performs comprehensive backtesting of XGBoost trading models
using historical market data and evaluates strategy performance.

Features:
- Multiple performance metrics (Sharpe ratio, max drawdown, win rate)
- Walk-forward analysis for realistic evaluation
- Risk-adjusted returns analysis
- Comparative analysis against buy-and-hold
- Detailed trade logging and visualization
"""

import asyncio
import argparse
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

from xgboost_trading_model import XGBoostTradingModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """Advanced backtesting engine for XGBoost trading strategies"""

    def __init__(self, model: XGBoostTradingModel):
        self.model = model
        self.results = {}

    def calculate_performance_metrics(self, trades: List[Dict[str, Any]],
                                    initial_capital: float = 10000.0) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""

        if not trades:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}

        # Extract trade data
        trade_df = pd.DataFrame(trades)
        trade_df['date'] = pd.to_datetime(trade_df['date'])

        # Calculate portfolio value over time
        portfolio_values = [initial_capital]
        current_capital = initial_capital

        for _, trade in trade_df.iterrows():
            if trade['type'] == 'SELL' and 'pnl' in trade:
                current_capital += trade['pnl']
            portfolio_values.append(current_capital)

        # Calculate returns
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()

        # Performance metrics
        total_return = (current_capital - initial_capital) / initial_capital * 100

        # Sharpe ratio (assuming 252 trading days per year)
        if len(portfolio_returns) > 1:
            sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        cumulative = pd.Series(portfolio_values).cummax()
        drawdown = (pd.Series(portfolio_values) - cumulative) / cumulative
        max_drawdown = drawdown.min() * 100

        # Win rate
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        win_rate = winning_trades / len(trades) * 100 if trades else 0

        # Profit factor
        gross_profit = sum([t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0])
        gross_loss = abs(sum([t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'final_capital': current_capital
        }

    def walk_forward_analysis(self, window_size: int = 252,
                            step_size: int = 21) -> Dict[str, Any]:
        """Perform walk-forward analysis for robust evaluation"""

        if self.model.feature_data is None:
            return {'error': 'No feature data available'}

        df = self.model.feature_data.copy()
        total_periods = len(df)

        results = []
        start_idx = 0

        while start_idx + window_size < total_periods:
            end_idx = start_idx + window_size

            # Training window
            train_data = df.iloc[start_idx:end_idx]

            # Test window
            test_start = end_idx
            test_end = min(end_idx + step_size, total_periods)
            test_data = df.iloc[test_start:test_end]

            if len(test_data) < 10:  # Minimum test size
                break

            # Train models on training window
            self.model.feature_data = train_data
            X, _, y_cls = self.model.prepare_targets()

            if X.empty or y_cls.empty:
                start_idx += step_size
                continue

            # Quick training without hyperparameter tuning for speed
            self.model.train_classification_model(hyperparameter_tune=False)

            # Test on out-of-sample data
            self.model.feature_data = test_data
            X_test, _, _ = self.model.prepare_targets()

            if not X_test.empty:
                signals = self.model.predict_signals(X_test)

                # Simulate trades
                test_trades = []
                position = 0

                for i, (idx, row) in enumerate(test_data.iterrows()):
                    if i >= len(signals):
                        break

                    signal = signals[i]
                    current_price = row['close']

                    if signal == 1 and position == 0:  # Buy
                        position = 1
                        entry_price = current_price
                        test_trades.append({
                            'type': 'BUY',
                            'price': current_price,
                            'date': idx
                        })

                    elif signal == -1 and position == 1:  # Sell
                        position = 0
                        exit_price = current_price
                        pnl = (exit_price - entry_price) / entry_price * 10000  # Assume $10k capital
                        test_trades.append({
                            'type': 'SELL',
                            'price': current_price,
                            'date': idx,
                            'pnl': pnl
                        })

                # Calculate metrics for this window
                metrics = self.calculate_performance_metrics(test_trades)
                metrics['window_start'] = df.index[start_idx].strftime('%Y-%m-%d')
                metrics['window_end'] = df.index[end_idx-1].strftime('%Y-%m-%d')
                metrics['test_start'] = df.index[test_start].strftime('%Y-%m-%d')
                metrics['test_end'] = df.index[test_end-1].strftime('%Y-%m-%d')

                results.append(metrics)

            start_idx += step_size

        # Aggregate results
        if results:
            avg_return = np.mean([r['total_return'] for r in results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
            avg_max_dd = np.mean([r['max_drawdown'] for r in results])

            return {
                'walk_forward_results': results,
                'average_return': avg_return,
                'average_sharpe': avg_sharpe,
                'average_max_drawdown': avg_max_dd,
                'num_windows': len(results)
            }
        else:
            return {'error': 'No valid walk-forward windows'}

    def compare_with_buy_and_hold(self, trades: List[Dict[str, Any]],
                                 initial_capital: float = 10000.0) -> Dict[str, float]:
        """Compare strategy performance with buy-and-hold"""

        if not self.model.feature_data:
            return {}

        df = self.model.feature_data.copy()

        # Buy and hold return
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        buy_hold_return = (end_price - start_price) / start_price * 100

        # Strategy return
        if trades:
            final_capital = trades[-1].get('capital', initial_capital)
            strategy_return = (final_capital - initial_capital) / initial_capital * 100
        else:
            strategy_return = 0.0

        # Calculate alpha (excess return)
        alpha = strategy_return - buy_hold_return

        return {
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'alpha': alpha,
            'outperformance': 'Yes' if alpha > 0 else 'No'
        }

async def run_backtest(symbol: str, start_date: str, end_date: Optional[str] = None,
                      initial_capital: float = 10000.0, walk_forward: bool = False) -> Dict[str, Any]:
    """Run comprehensive backtest for a symbol"""

    logger.info(f"Starting backtest for {symbol}")

    # Initialize model
    model = XGBoostTradingModel(symbol=symbol, start_date=start_date, end_date=end_date)

    # Load and prepare data
    await model.load_data()
    if model.raw_data is None or model.raw_data.empty:
        return {'error': f'No data available for {symbol}'}

    model.add_technical_indicators()
    model.prepare_targets()

    # Train models
    logger.info("Training models...")
    model.train_regression_model(hyperparameter_tune=False)
    model.train_classification_model(hyperparameter_tune=False)

    # Initialize backtest engine
    backtest_engine = BacktestEngine(model)

    # Run standard backtest
    logger.info("Running standard backtest...")
    total_return, trades = model.backtest_strategy(initial_capital)

    # Calculate detailed metrics
    metrics = backtest_engine.calculate_performance_metrics(trades, initial_capital)

    # Compare with buy-and-hold
    comparison = backtest_engine.compare_with_buy_and_hold(trades, initial_capital)

    # Walk-forward analysis (optional, as it's time-consuming)
    walk_forward_results = {}
    if walk_forward:
        logger.info("Running walk-forward analysis...")
        walk_forward_results = backtest_engine.walk_forward_analysis()

    # Compile results
    results = {
        'symbol': symbol,
        'backtest_period': {
            'start': start_date,
            'end': end_date or datetime.now().strftime('%Y-%m-%d')
        },
        'data_points': len(model.feature_data) if model.feature_data is not None else 0,
        'performance_metrics': metrics,
        'buy_hold_comparison': comparison,
        'total_trades': len(trades),
        'trade_log': trades[:20],  # First 20 trades for brevity
        'walk_forward_analysis': walk_forward_results
    }

    logger.info(f"Backtest completed for {symbol}")
    logger.info(f"Total Return: {metrics['total_return']:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")

    return results

def save_backtest_results(results: Dict[str, Any], output_file: Optional[str] = None):
    """Save backtest results to JSON file"""
    if not output_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'xgboost_backtest_results_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Backtest results saved to {output_file}")

def print_backtest_summary(results: Dict[str, Any]):
    """Print formatted backtest summary"""
    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    print("\n" + "="*80)
    print("XGBoost Backtest Summary")
    print("="*80)
    print(f"Symbol: {results['symbol']}")
    print(f"Period: {results['backtest_period']['start']} to {results['backtest_period']['end']}")
    print(f"Data Points: {results['data_points']}")
    print()

    metrics = results['performance_metrics']
    print("Performance Metrics:")
    print(f"  Total Return: {metrics['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Total Trades: {metrics['total_trades']}")
    print()

    comparison = results['buy_hold_comparison']
    print("Buy & Hold Comparison:")
    print(f"  Strategy Return: {comparison['strategy_return']:.2f}%")
    print(f"  Buy & Hold Return: {comparison['buy_hold_return']:.2f}%")
    print(f"  Alpha: {comparison['alpha']:.2f}%")
    print(f"  Outperformance: {comparison['outperformance']}")
    print()

    if 'walk_forward_analysis' in results and 'average_return' in results['walk_forward_analysis']:
        wf = results['walk_forward_analysis']
        print("Walk-Forward Analysis:")
        print(f"  Average Return: {wf['average_return']:.2f}%")
        print(f"  Average Sharpe: {wf['average_sharpe']:.2f}")
        print(f"  Average Max DD: {wf['average_max_drawdown']:.2f}%")
        print(f"  Number of Windows: {wf['num_windows']}")

    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Backtest XGBoost trading models')
    parser.add_argument('--symbol', '-s', required=True, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--start-date', '-d', default='2020-01-01', help='Start date for backtest')
    parser.add_argument('--end-date', '-e', help='End date for backtest')
    parser.add_argument('--capital', '-c', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--walk-forward', '-w', action='store_true', help='Run walk-forward analysis')
    parser.add_argument('--output', '-o', help='Output file for results')

    args = parser.parse_args()

    try:
        results = asyncio.run(run_backtest(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital,
            walk_forward=args.walk_forward
        ))

        save_backtest_results(results, args.output)
        print_backtest_summary(results)

    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\nzcon\VSPython\ai_advancements\backtest_xgboost.py