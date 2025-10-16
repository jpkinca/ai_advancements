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
        """Prepare data for backtesting with proper alignment and validation"""

        try:
            # Ensure we have the necessary price data
            if 'close' not in market_data.columns:
                logger.error("Market data missing 'close' prices")
                return None

            # Handle multi-index data properly
            if isinstance(market_data.index, pd.MultiIndex):
                logger.warning("Multi-index detected; using first ticker only for MVP")
                first_ticker = market_data.index.get_level_values('ticker').unique()[0] if 'ticker' in market_data.index.names else market_data.index.get_level_values(0)[0]
                
                # Extract data for first ticker
                try:
                    ticker_data = market_data.xs(first_ticker, level='ticker' if 'ticker' in market_data.index.names else 0)
                    close_prices = ticker_data['close']
                except KeyError:
                    logger.error(f"Could not extract data for ticker {first_ticker}")
                    return None
                
                # Align factor values properly - factor_values should have MultiIndex after dropna
                if isinstance(factor_values.index, pd.MultiIndex):
                    try:
                        # Extract factor values for this ticker
                        factor_for_ticker = factor_values.xs(first_ticker, level='ticker' if 'ticker' in factor_values.index.names else 1)
                        # Ensure date alignment
                        factor_aligned = factor_for_ticker.reindex(close_prices.index).ffill().fillna(0)
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Could not extract factor for ticker {first_ticker}: {e}")
                        # Try reindexing directly if factor doesn't have ticker level
                        try:
                            factor_aligned = factor_values.reindex(close_prices.index, method='ffill').fillna(0)
                        except Exception:
                            logger.warning("Reindexing failed, using zeros")
                            factor_aligned = pd.Series(0, index=close_prices.index)
                else:
                    # Factor is not multi-index (happens after dropna), try to reconstruct alignment
                    logger.warning("Factor values lost MultiIndex structure, attempting reconstruction")
                    # Since we can't easily reconstruct, use zeros for now
                    factor_aligned = pd.Series(0, index=close_prices.index)
            else:
                close_prices = market_data['close']
                factor_aligned = factor_values.reindex(close_prices.index, method='ffill').fillna(0)

            # Create DataFrame with proper alignment
            backtest_df = pd.DataFrame({
                'close': close_prices,
                'factor': factor_aligned
            }).dropna()

            # Check minimum data requirement (configurable, default 252 trading days)
            min_days = self.config.get('min_backtest_days', 252)
            if len(backtest_df) < min_days:
                logger.warning(f"Insufficient data: {len(backtest_df)} < min {min_days} days required")
                # For MVP testing, allow shorter periods but warn
                if len(backtest_df) < 50:  # Absolute minimum
                    return None

            # Validate factor has variation
            if backtest_df['factor'].std() == 0:
                logger.warning("Factor has no variation (constant values)")
                return None

            logger.info(f"Prepared backtest data: {len(backtest_df)} observations, factor std: {backtest_df['factor'].std():.4f}")
            return backtest_df

        except Exception as e:
            logger.error(f"Error preparing backtest data: {e}")
            return None

    def _run_factor_backtest(self, data: pd.DataFrame) -> Optional[Any]:
        """Run the actual backtest using vectorbt with improved signal logic"""

        try:
            # Create price data
            price = data['close']
            factor = data['factor']

            # Normalize factor using z-score for better signal quality
            factor_normalized = (factor - factor.mean()) / factor.std()

            # Create signals based on normalized factor
            signal_threshold = self.config.get('signal_threshold_factor', 0.5)
            threshold = signal_threshold  # Now using normalized factor

            # Generate long/short signals with no overlap
            long_signal = (factor_normalized > threshold).astype(int)
            short_signal = (factor_normalized < -threshold).astype(int)
            
            # Ensure no overlapping positions
            entries = long_signal & ~short_signal  # Only long when not short
            exits = short_signal | (long_signal.shift(1) == 1) & (long_signal == 0)  # Exit on short signal or long signal ends

            # Log signal statistics
            long_count = entries.sum()
            exit_count = exits.sum()
            logger.info(f"Generated {long_count} long signals, {exit_count} exit signals")

            if long_count == 0:
                logger.warning("No trading signals generated - factor may be too weak")
                # Return minimal portfolio for metrics calculation
                entries = pd.Series([1] + [0] * (len(price) - 1), index=price.index)  # Single buy-and-hold trade
                exits = pd.Series([0] * (len(price) - 1) + [1], index=price.index)   # Exit at end

            # Run backtest
            init_cash = self.config.get('initial_capital', 10000)
            fees = self.config.get('commission', 0.001)
            
            portfolio = self.vbt.Portfolio.from_signals(
                close=price,
                entries=entries,
                exits=exits,
                freq='D',  # Daily frequency
                init_cash=init_cash,
                fees=fees,
                slippage=self.config.get('slippage', 0.001)
            )

            return portfolio

        except Exception as e:
            logger.error(f"Error running vectorbt backtest: {e}")
            return None

    def _calculate_performance_metrics(self, portfolio: Any) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics with proper risk-adjusted measures"""

        try:
            metrics = {}

            # Basic returns
            total_return = portfolio.total_return()
            annual_return = portfolio.annualized_return()
            volatility = portfolio.annualized_volatility()

            # Risk-free rate (configurable, default 2%)
            risk_free_rate = self.config.get('risk_free_rate', 0.02)

            metrics.update({
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'annual_volatility': float(volatility),
                'sharpe_ratio': float((annual_return - risk_free_rate) / volatility) if volatility > 0 else 0,
            })

            # Risk metrics (handle API changes in VectorBT)
            try:
                max_drawdown = portfolio.max_drawdown()
                metrics['max_drawdown'] = float(max_drawdown)
            except Exception as e:
                logger.warning(f"Could not calculate max_drawdown: {e}")
                metrics['max_drawdown'] = 0

            try:
                # Handle VectorBT API changes - try different methods
                if hasattr(portfolio, 'value_at_risk'):
                    var_95 = portfolio.value_at_risk()  # No parameter version
                else:
                    # Alternative calculation
                    returns = portfolio.returns()
                    var_95 = returns.quantile(0.05)
                metrics['value_at_risk_95'] = float(var_95)
            except Exception as e:
                logger.warning(f"Could not calculate VaR: {e}")
                metrics['value_at_risk_95'] = 0

            # Trading metrics
            try:
                trades = portfolio.trades()
                total_trades = trades.count()
                win_rate = trades.winning_rate() if hasattr(trades, 'winning_rate') else 0
                profit_factor = trades.profit_factor() if hasattr(trades, 'profit_factor') else 0
                avg_trade_return = trades.returns().mean() if hasattr(trades, 'returns') else 0

                metrics.update({
                    'total_trades': int(total_trades) if not np.isnan(total_trades) else 0,
                    'win_rate': float(win_rate) if not np.isnan(win_rate) else 0,
                    'profit_factor': float(profit_factor) if not np.isnan(profit_factor) else 0,
                    'avg_trade_return': float(avg_trade_return) if not np.isnan(avg_trade_return) else 0,
                })

                # Add warning for low trade count
                if metrics['total_trades'] < 10:
                    metrics['warning'] = 'Low trade count - results may not be statistically significant'
                    logger.warning(f"Low trade count: {metrics['total_trades']} trades")

            except Exception as e:
                logger.warning(f"Could not calculate trade metrics: {e}")
                metrics.update({
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'avg_trade_return': 0,
                    'warning': 'Failed to calculate trade metrics'
                })

            # Benchmark comparison (buy and hold)
            try:
                benchmark = self._calculate_benchmark_return(portfolio)
                metrics['benchmark_return'] = float(benchmark)
                metrics['alpha'] = float(annual_return - benchmark)
            except Exception as e:
                logger.warning(f"Could not calculate benchmark: {e}")
                metrics['benchmark_return'] = 0
                metrics['alpha'] = 0

            # Additional risk-adjusted metrics
            try:
                # Sortino ratio (downside deviation)
                returns = portfolio.returns()
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_deviation = negative_returns.std() * np.sqrt(252)  # Annualized
                    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
                else:
                    sortino_ratio = float('inf') if annual_return > risk_free_rate else 0
                
                metrics['sortino_ratio'] = float(sortino_ratio) if not np.isinf(sortino_ratio) else 999
                
                # Calmar ratio
                calmar_ratio = annual_return / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
                metrics['calmar_ratio'] = float(calmar_ratio)

            except Exception as e:
                logger.warning(f"Could not calculate advanced metrics: {e}")
                metrics['sortino_ratio'] = 0
                metrics['calmar_ratio'] = 0

            logger.info(f"Calculated metrics: Sharpe={metrics['sharpe_ratio']:.3f}, Return={metrics['annual_return']:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'error': str(e), 'sharpe_ratio': 0, 'annual_return': 0}

    def _calculate_benchmark_return(self, portfolio: Any) -> float:
        """Calculate buy-and-hold benchmark return with proper date handling"""

        try:
            # Get the price series used in backtest
            price_data = portfolio.close

            # Calculate buy-and-hold return
            start_price = price_data.iloc[0]
            end_price = price_data.iloc[-1]
            benchmark_return = (end_price / start_price) - 1

            # Calculate actual time period for proper annualization
            if hasattr(price_data.index, 'to_pydatetime'):
                start_date = pd.to_datetime(price_data.index[0])
                end_date = pd.to_datetime(price_data.index[-1])
                days_elapsed = (end_date - start_date).days
            else:
                # Fallback to counting observations
                days_elapsed = len(price_data)

            # Annualize properly
            if days_elapsed > 0:
                annual_benchmark = (1 + benchmark_return) ** (365.25 / days_elapsed) - 1
            else:
                annual_benchmark = 0

            logger.info(f"Benchmark: {benchmark_return:.3f} over {days_elapsed} days, annualized: {annual_benchmark:.3f}")
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