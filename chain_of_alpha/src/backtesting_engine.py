"""
Backtesting Engine Module for Chain-of-Alpha MVP

Implements quantitative backtesting of alpha factors using vectorbt
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BacktestingEngine:
    """
    Engine for backtesting alpha factors using vectorbt
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vbt = None
        self._initialize_vectorbt()

    def _initialize_vectorbt(self):
        """Initialize vectorbt library"""
        try:
            import vectorbt as vbt
            self.vbt = vbt
            logger.info("Vectorbt initialized successfully")
        except ImportError:
            logger.error("Vectorbt not available. Install with: pip install vectorbt")
            raise

    def backtest_factors(self, factors: List[Dict[str, Any]], market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Backtest multiple factors

        Args:
            factors: List of factor dictionaries with computed values
            market_data: Market data for backtesting

        Returns:
            Factors with backtest results
        """
        logger.info(f"Backtesting {len(factors)} factors")

        backtested_factors = []

        for factor in factors:
            try:
                result = self._backtest_single_factor(factor, market_data)
                if result:
                    backtested_factors.append(result)

            except Exception as e:
                logger.error(f"Backtest failed for factor {factor['id']}: {e}")
                continue

        logger.info(f"Successfully backtested {len(backtested_factors)} factors")
        return backtested_factors

    def _backtest_single_factor(self, factor: Dict[str, Any], market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Backtest a single factor"""

        try:
            # Extract factor values
            factor_values = factor.get('values')
            if factor_values is None:
                logger.warning(f"No values found for factor {factor['id']}")
                return None

            # Prepare data for backtesting
            backtest_data = self._prepare_backtest_data(factor_values, market_data)

            if backtest_data is None:
                return None

            # Run the backtest
            portfolio = self._run_factor_backtest(backtest_data)

            if portfolio is None:
                return None

            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(portfolio)

            # Create result
            result = factor.copy()
            result.update({
                'backtest_results': {
                    'portfolio': portfolio,
                    'metrics': metrics,
                    'backtest_time': datetime.now().isoformat()
                }
            })

            return result

        except Exception as e:
            logger.error(f"Error backtesting factor {factor['id']}: {e}")
            return None

    def _prepare_backtest_data(self, factor_values: pd.Series, market_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare data for backtesting"""

        try:
            # Ensure we have the necessary price data
            if 'close' not in market_data.columns:
                logger.error("Market data missing 'close' prices")
                return None

            # Align factor values with market data
            if isinstance(market_data.index, pd.MultiIndex):
                # Handle multi-index (ticker, date)
                close_prices = market_data['close'].unstack(level=0) if 'ticker' in market_data.index.names else market_data['close']
            else:
                close_prices = market_data['close']

            # Align factor with close prices
            if isinstance(factor_values.index, pd.MultiIndex):
                factor_aligned = factor_values.unstack(level=0) if len(factor_values.index.levels) > 1 else factor_values
            else:
                factor_aligned = factor_values

            # Ensure datetime index
            if not isinstance(close_prices.index, pd.DatetimeIndex):
                close_prices.index = pd.to_datetime(close_prices.index)
            if not isinstance(factor_aligned.index, pd.DatetimeIndex):
                factor_aligned.index = pd.to_datetime(factor_aligned.index)

            # Align the indices
            common_index = close_prices.index.intersection(factor_aligned.index)
            if len(common_index) == 0:
                logger.warning("No overlapping dates between factor and price data")
                return None

            close_prices = close_prices.loc[common_index]
            factor_aligned = factor_aligned.loc[common_index]

            # Create backtest DataFrame
            backtest_df = pd.DataFrame({
                'close': close_prices,
                'factor': factor_aligned
            }).dropna()

            if len(backtest_df) < 100:  # Minimum data requirement
                logger.warning(f"Insufficient data for backtesting: {len(backtest_df)} observations")
                return None

            return backtest_df

        except Exception as e:
            logger.error(f"Error preparing backtest data: {e}")
            return None

    def _run_factor_backtest(self, data: pd.DataFrame) -> Optional[Any]:
        """Run the actual backtest using vectorbt"""

        try:
            # Create price data
            price = data['close']
            factor = data['factor']

            # Create signals based on factor
            # Long when factor > threshold, short when factor < -threshold
            threshold = factor.std() * 0.5  # Adaptive threshold based on factor volatility

            long_signal = (factor > threshold).astype(int)
            short_signal = (factor < -threshold).astype(int)

            # Create entries and exits
            entries = long_signal
            exits = short_signal

            # Run backtest
            portfolio = self.vbt.Portfolio.from_signals(
                close=price,
                entries=entries,
                exits=exits,
                freq='D',  # Daily frequency
                init_cash=10000,  # Starting capital
                fees=0.001,  # 0.1% trading fees
                slippage=0.001  # 0.1% slippage
            )

            return portfolio

        except Exception as e:
            logger.error(f"Error running vectorbt backtest: {e}")
            return None

    def _calculate_performance_metrics(self, portfolio: Any) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""

        try:
            metrics = {}

            # Basic returns
            total_return = portfolio.total_return()
            annual_return = portfolio.annualized_return()
            volatility = portfolio.annualized_volatility()

            metrics.update({
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'annual_volatility': float(volatility),
                'sharpe_ratio': float(annual_return / volatility) if volatility > 0 else 0,
            })

            # Risk metrics
            max_drawdown = portfolio.max_drawdown()
            var_95 = portfolio.value_at_risk(0.05)
            cvar_95 = portfolio.conditional_value_at_risk(0.05)

            metrics.update({
                'max_drawdown': float(max_drawdown),
                'value_at_risk_95': float(var_95),
                'conditional_var_95': float(cvar_95),
            })

            # Trading metrics
            total_trades = portfolio.trades().count()
            win_rate = portfolio.trades().winning_rate()
            profit_factor = portfolio.trades().profit_factor()
            avg_trade_return = portfolio.trades().returns().mean()

            metrics.update({
                'total_trades': int(total_trades),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor) if not np.isnan(profit_factor) else 0,
                'avg_trade_return': float(avg_trade_return) if not np.isnan(avg_trade_return) else 0,
            })

            # Benchmark comparison (buy and hold)
            try:
                benchmark = self._calculate_benchmark_return(portfolio)
                metrics['benchmark_return'] = float(benchmark)
                metrics['alpha'] = float(annual_return - benchmark)
            except:
                metrics['benchmark_return'] = 0
                metrics['alpha'] = 0

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def _calculate_benchmark_return(self, portfolio: Any) -> float:
        """Calculate buy-and-hold benchmark return"""

        try:
            # Get the price series used in backtest
            price_data = portfolio.close

            # Calculate buy-and-hold return
            start_price = price_data.iloc[0]
            end_price = price_data.iloc[-1]
            benchmark_return = (end_price / start_price) - 1

            # Annualize
            days = len(price_data)
            annual_benchmark = (1 + benchmark_return) ** (252 / days) - 1

            return annual_benchmark

        except Exception as e:
            logger.error(f"Error calculating benchmark return: {e}")
            return 0

    def compare_factors(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare performance across multiple factors

        Args:
            factors: List of factors with backtest results

        Returns:
            Comparison statistics
        """
        try:
            comparison = {
                'factor_count': len(factors),
                'best_performer': None,
                'worst_performer': None,
                'summary_stats': {}
            }

            if not factors:
                return comparison

            # Extract metrics
            performances = []
            for factor in factors:
                metrics = factor.get('backtest_results', {}).get('metrics', {})
                if metrics:
                    performances.append({
                        'id': factor['id'],
                        'sharpe': metrics.get('sharpe_ratio', 0),
                        'return': metrics.get('annual_return', 0),
                        'max_dd': metrics.get('max_drawdown', 0),
                        'win_rate': metrics.get('win_rate', 0)
                    })

            if not performances:
                return comparison

            # Find best and worst
            best_sharpe = max(performances, key=lambda x: x['sharpe'])
            worst_sharpe = min(performances, key=lambda x: x['sharpe'])

            comparison['best_performer'] = best_sharpe
            comparison['worst_performer'] = worst_sharpe

            # Summary statistics
            sharpe_ratios = [p['sharpe'] for p in performances]
            returns = [p['return'] for p in performances]

            comparison['summary_stats'] = {
                'avg_sharpe': float(np.mean(sharpe_ratios)),
                'std_sharpe': float(np.std(sharpe_ratios)),
                'avg_return': float(np.mean(returns)),
                'std_return': float(np.std(returns)),
                'sharpe_range': float(max(sharpe_ratios) - min(sharpe_ratios))
            }

            return comparison

        except Exception as e:
            logger.error(f"Error comparing factors: {e}")
            return {}

    def save_backtest_results(self, factors: List[Dict[str, Any]], filepath: str):
        """Save backtest results to file"""
        try:
            # Extract key results for serialization
            serializable_results = []
            for factor in factors:
                result = {
                    'id': factor['id'],
                    'expression': factor['expression'],
                    'explanation': factor.get('explanation', ''),
                    'metrics': factor.get('backtest_results', {}).get('metrics', {}),
                    'backtest_time': factor.get('backtest_results', {}).get('backtest_time')
                }
                serializable_results.append(result)

            import json
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)

            logger.info(f"Saved backtest results for {len(factors)} factors to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")