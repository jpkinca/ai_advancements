"""
Advanced Backtesting Framework for Sweet Spot & Danger Zone System
Comprehensive performance analysis and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

from sweet_spot_danger_zone_system import DualSignalTradingSystem

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    initial_capital: float = 100000
    commission_per_trade: float = 0.001  # 0.1% commission
    slippage: float = 0.0005  # 0.05% slippage
    max_position_size: float = 0.1  # Max 10% of capital per position
    rebalance_frequency: str = 'daily'  # 'daily', 'hourly', 'minute'
    risk_free_rate: float = 0.02  # 2% annual risk-free rate

    # Walk-forward validation
    walk_forward_window: int = 252  # 1 year training window
    walk_forward_step: int = 21  # 1 month step
    validation_window: int = 63  # 3 months validation

class PerformanceMetrics:
    """Comprehensive performance metrics calculation"""

    @staticmethod
    def calculate_returns(portfolio_values: pd.Series) -> pd.Series:
        """Calculate periodic returns"""
        return portfolio_values.pct_change().fillna(0)

    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns"""
        return (1 + returns).cumprod()

    @staticmethod
    def calculate_annualized_return(total_return: float, periods: int, freq: str = 'daily') -> float:
        """Calculate annualized return based on frequency"""
        if freq == 'daily':
            annual_periods = 252
        elif freq == 'hourly':
            annual_periods = 252 * 6.5
        elif freq == 'minute':
            annual_periods = 252 * 6.5 * 60
        else:
            annual_periods = 252

        return (1 + total_return) ** (annual_periods / periods) - 1

    @staticmethod
    def calculate_volatility(returns: pd.Series, freq: str = 'daily') -> float:
        """Calculate annualized volatility"""
        if freq == 'daily':
            annual_periods = 252
        elif freq == 'hourly':
            annual_periods = 252 * 6.5
        else:
            annual_periods = 252

        return returns.std() * np.sqrt(annual_periods)

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, freq: str = 'daily') -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        volatility = PerformanceMetrics.calculate_volatility(returns, freq)
        return excess_returns.mean() / volatility if volatility > 0 else 0

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, freq: str = 'daily') -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        return excess_returns.mean() / downside_volatility if downside_volatility > 0 else 0

    @staticmethod
    def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()

    @staticmethod
    def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        return annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    @staticmethod
    def calculate_win_rate(returns: pd.Series) -> float:
        """Calculate win rate"""
        return (returns > 0).mean()

    @staticmethod
    def calculate_profit_factor(returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    @staticmethod
    def calculate_alpha_beta(returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Calculate alpha and beta vs benchmark"""
        # Simple linear regression
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        alpha = returns.mean() - beta * benchmark_returns.mean()

        return alpha, beta

class WalkForwardValidator:
    """Walk-forward validation for robust model testing"""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def validate(self, system: DualSignalTradingSystem, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform walk-forward validation
        """
        logger.info("Performing walk-forward validation...")

        results = []
        n_periods = len(df)

        for i in range(self.config.walk_forward_window,
                      n_periods - self.config.validation_window,
                      self.config.walk_forward_step):

            # Define training and validation windows
            train_start = i - self.config.walk_forward_window
            train_end = i
            val_end = min(i + self.config.validation_window, n_periods)

            train_data = df.iloc[train_start:train_end]
            val_data = df.iloc[i:val_end]

            # Retrain model on training data
            system.train(train_data)

            # Test on validation data
            backtest_result = system.backtest(val_data, self.config.initial_capital)

            results.append({
                'train_end_date': df.index[train_end-1],
                'val_start_date': df.index[i],
                'val_end_date': df.index[val_end-1],
                'sharpe_ratio': backtest_result['sharpe_ratio'],
                'total_return': backtest_result['total_return'],
                'max_drawdown': backtest_result['max_drawdown'],
                'win_rate': backtest_result['win_rate']
            })

        # Aggregate results
        results_df = pd.DataFrame(results)

        summary = {
            'mean_sharpe': results_df['sharpe_ratio'].mean(),
            'std_sharpe': results_df['sharpe_ratio'].std(),
            'mean_return': results_df['total_return'].mean(),
            'mean_max_dd': results_df['max_drawdown'].mean(),
            'mean_win_rate': results_df['win_rate'].mean(),
            'sharpe_consistency': (results_df['sharpe_ratio'] > 0).mean(),
            'return_consistency': (results_df['total_return'] > 0).mean(),
            'detailed_results': results_df
        }

        return summary

class AdvancedBacktester:
    """Advanced backtesting with realistic trading conditions"""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run_backtest(self, system: DualSignalTradingSystem, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive backtest with realistic trading conditions
        """
        logger.info("Running advanced backtest...")

        signals = system.generate_signals(df)

        # Initialize portfolio
        capital = self.config.initial_capital
        position = 0  # Current position size
        portfolio_values = []
        trades = []

        # Simulate trading
        for i in range(len(signals)):
            current_signal = signals.iloc[i]
            current_price = df.iloc[i]['close']

            # Calculate target position
            target_position = current_signal['position_size'] * capital / current_price
            target_position = np.clip(target_position, -self.config.max_position_size * capital / current_price,
                                    self.config.max_position_size * capital / current_price)

            # Calculate trade size
            trade_size = target_position - position

            if abs(trade_size) > 0.01:  # Minimum trade threshold
                # Apply slippage and commission
                execution_price = current_price * (1 + np.sign(trade_size) * self.config.slippage)
                commission = abs(trade_size * execution_price * self.config.commission_per_trade)

                # Execute trade
                capital -= trade_size * execution_price + commission
                position = target_position

                trades.append({
                    'date': df.index[i],
                    'price': execution_price,
                    'size': trade_size,
                    'commission': commission,
                    'signal': current_signal['combined_signal'],
                    'capital': capital
                })

            # Calculate portfolio value (position + cash)
            portfolio_value = capital + position * current_price
            portfolio_values.append(portfolio_value)

        # Create results
        portfolio_series = pd.Series(portfolio_values, index=df.index)
        returns = PerformanceMetrics.calculate_returns(portfolio_series)

        # Calculate comprehensive metrics
        total_return = portfolio_series.iloc[-1] / self.config.initial_capital - 1
        annualized_return = PerformanceMetrics.calculate_annualized_return(
            total_return, len(df), self.config.rebalance_frequency)

        volatility = PerformanceMetrics.calculate_volatility(returns, self.config.rebalance_frequency)
        sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(
            returns, self.config.risk_free_rate, self.config.rebalance_frequency)
        sortino_ratio = PerformanceMetrics.calculate_sortino_ratio(
            returns, self.config.risk_free_rate, self.config.rebalance_frequency)

        max_drawdown = PerformanceMetrics.calculate_max_drawdown(portfolio_series)
        calmar_ratio = PerformanceMetrics.calculate_calmar_ratio(annualized_return, max_drawdown)

        win_rate = PerformanceMetrics.calculate_win_rate(returns)
        profit_factor = PerformanceMetrics.calculate_profit_factor(returns)

        # Monthly returns analysis
        monthly_returns = returns.groupby(pd.Grouper(freq='M')).apply(lambda x: (1 + x).prod() - 1)
        monthly_win_rate = (monthly_returns > 0).mean()

        return {
            'portfolio_values': portfolio_series,
            'returns': returns,
            'trades': pd.DataFrame(trades),
            'signals': signals,

            # Performance metrics
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,

            # Additional analysis
            'monthly_returns': monthly_returns,
            'monthly_win_rate': monthly_win_rate,
            'total_trades': len(trades),
            'avg_trade_size': np.mean([abs(t['size']) for t in trades]) if trades else 0,
            'avg_commission': np.mean([t['commission'] for t in trades]) if trades else 0,

            # Risk metrics
            'value_at_risk_95': np.percentile(returns, 5),
            'expected_shortfall_95': returns[returns <= np.percentile(returns, 5)].mean(),
            'largest_win': returns.max(),
            'largest_loss': returns.min()
        }

class BacktestVisualizer:
    """Visualization tools for backtest analysis"""

    @staticmethod
    def plot_portfolio_performance(results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot portfolio performance over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        portfolio_values = results['portfolio_values']

        # Portfolio value
        axes[0, 0].plot(portfolio_values.index, portfolio_values.values)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)

        # Drawdown
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak * 100
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        axes[0, 1].set_title('Portfolio Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)

        # Monthly returns
        monthly_returns = results['monthly_returns'] * 100
        axes[1, 0].bar(range(len(monthly_returns)), monthly_returns.values)
        axes[1, 0].set_title('Monthly Returns')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].axhline(y=0, color='black', linestyle='--')

        # Returns distribution
        returns = results['returns'] * 100
        axes[1, 1].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--')
        axes[1, 1].set_title('Returns Distribution')
        axes[1, 1].set_xlabel('Return (%)')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_signal_analysis(signals: pd.DataFrame, save_path: Optional[str] = None):
        """Plot signal analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Signal distribution
        signal_counts = signals['combined_signal'].value_counts().sort_index()
        axes[0, 0].bar(signal_counts.index, signal_counts.values)
        axes[0, 0].set_title('Signal Distribution')
        axes[0, 0].set_xlabel('Signal Type (0=Avoid, 1=Weak Buy, 2=Strong Buy)')
        axes[0, 0].set_ylabel('Count')

        # Probability distributions
        if 'sweet_probability' in signals.columns:
            axes[0, 1].hist(signals['sweet_probability'].dropna(), bins=30, alpha=0.7, label='Sweet Spot')
        if 'danger_probability' in signals.columns:
            axes[0, 1].hist(signals['danger_probability'].dropna(), bins=30, alpha=0.7, label='Danger Zone')
        axes[0, 1].set_title('Probability Distributions')
        axes[0, 1].set_xlabel('Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        # Position size distribution
        if 'position_size' in signals.columns:
            axes[1, 0].hist(signals['position_size'].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Position Size Distribution')
            axes[1, 0].set_xlabel('Position Size')
            axes[1, 0].set_ylabel('Frequency')

        # Confidence score over time
        if 'confidence_score' in signals.columns:
            axes[1, 1].plot(signals.index, signals['confidence_score'])
            axes[1, 1].set_title('Confidence Score Over Time')
            axes[1, 1].set_ylabel('Confidence Score')
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def generate_performance_report(results: Dict[str, Any]) -> str:
        """Generate comprehensive performance report"""
        report = f"""
SWEET SPOT & DANGER ZONE BACKTEST REPORT
{'='*50}

PERFORMANCE METRICS:
{'-'*20}
Total Return: {results['total_return']:.2%}
Annualized Return: {results['annualized_return']:.2%}
Volatility: {results['volatility']:.2%}
Sharpe Ratio: {results['sharpe_ratio']:.2f}
Sortino Ratio: {results['sortino_ratio']:.2f}
Maximum Drawdown: {results['max_drawdown']:.2%}
Calmar Ratio: {results['calmar_ratio']:.2f}
Win Rate: {results['win_rate']:.2%}
Profit Factor: {results['profit_factor']:.2f}

TRADING STATISTICS:
{'-'*20}
Total Trades: {results['total_trades']}
Average Trade Size: ${results['avg_trade_size']:.2f}
Average Commission: ${results['avg_commission']:.2f}
Monthly Win Rate: {results['monthly_win_rate']:.2%}

RISK METRICS:
{'-'*20}
VaR (95%): {results['value_at_risk_95']:.2%}
Expected Shortfall (95%): {results['expected_shortfall_95']:.2%}
Largest Win: {results['largest_win']:.2%}
Largest Loss: {results['largest_loss']:.2%}

INTERPRETATION:
{'-'*20}
Sharpe Ratio > 1.0: Good risk-adjusted returns
Maximum Drawdown < 20%: Reasonable risk control
Win Rate > 50%: Profitable strategy
Profit Factor > 1.5: Strong profit generation

CONCLUSION:
{'-'*20}
Strategy {'PERFORMS WELL' if results['sharpe_ratio'] > 1.0 and results['max_drawdown'] > -0.20 else 'NEEDS IMPROVEMENT'}
"""

        return report

class ComparativeAnalyzer:
    """Compare multiple backtest results"""

    @staticmethod
    def compare_strategies(results_list: List[Dict[str, Any]], strategy_names: List[str]) -> pd.DataFrame:
        """Compare multiple strategy results"""
        comparison_data = []

        for result, name in zip(results_list, strategy_names):
            comparison_data.append({
                'Strategy': name,
                'Total Return': result['total_return'],
                'Annualized Return': result['annualized_return'],
                'Sharpe Ratio': result['sharpe_ratio'],
                'Max Drawdown': result['max_drawdown'],
                'Win Rate': result['win_rate'],
                'Profit Factor': result['profit_factor'],
                'Total Trades': result['total_trades']
            })

        return pd.DataFrame(comparison_data)

    @staticmethod
    def plot_comparison(comparison_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot strategy comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        metrics = ['Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor', 'Total Trades']

        for i, metric in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            bars = ax.bar(range(len(comparison_df)), comparison_df[metric])
            ax.set_title(metric)
            ax.set_xticks(range(len(comparison_df)))
            ax.set_xticklabels(comparison_df['Strategy'], rotation=45)

            # Color bars based on performance
            for j, bar in enumerate(bars):
                if metric in ['Annualized Return', 'Sharpe Ratio', 'Win Rate', 'Profit Factor']:
                    color = 'green' if comparison_df[metric].iloc[j] > 0 else 'red'
                elif metric == 'Max Drawdown':
                    color = 'green' if comparison_df[metric].iloc[j] > -0.20 else 'red'
                else:
                    color = 'blue'
                bar.set_color(color)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Convenience functions
def run_complete_backtest(system: DualSignalTradingSystem,
                         df: pd.DataFrame,
                         config: BacktestConfig = None) -> Dict[str, Any]:
    """Run complete backtest with visualization and reporting"""
    config = config or BacktestConfig()

    # Run backtest
    backtester = AdvancedBacktester(config)
    results = backtester.run_backtest(system, df)

    # Generate visualizations
    BacktestVisualizer.plot_portfolio_performance(results)
    BacktestVisualizer.plot_signal_analysis(results['signals'])

    # Generate report
    report = BacktestVisualizer.generate_performance_report(results)
    print(report)

    return results

def run_walk_forward_validation(system: DualSignalTradingSystem,
                               df: pd.DataFrame,
                               config: BacktestConfig = None) -> Dict[str, Any]:
    """Run walk-forward validation"""
    config = config or BacktestConfig()

    validator = WalkForwardValidator(config)
    wf_results = validator.validate(system, df)

    print("WALK-FORWARD VALIDATION RESULTS:")
    print(f"Mean Sharpe Ratio: {wf_results['mean_sharpe']:.2f}")
    print(f"Sharpe Consistency: {wf_results['sharpe_consistency']:.2%}")
    print(f"Return Consistency: {wf_results['return_consistency']:.2%}")

    return wf_results

# Example usage
def example_backtest():
    """Example of complete backtesting workflow"""
    from sweet_spot_danger_zone_system import example_usage

    # Get trained system and data
    system, signals, basic_results = example_usage()

    # Run advanced backtest
    config = BacktestConfig(
        initial_capital=100000,
        commission_per_trade=0.001,
        slippage=0.0005
    )

    results = run_complete_backtest(system, signals, config)

    # Run walk-forward validation
    wf_results = run_walk_forward_validation(system, signals, config)

    return results, wf_results

if __name__ == "__main__":
    results, wf_results = example_backtest()