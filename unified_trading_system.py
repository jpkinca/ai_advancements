"""
Unified Trading System Integration
Integrates Sweet Spot & Danger Zone detection with TensorTrade architecture
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import warnings

from sweet_spot_danger_zone_system import DualSignalTradingSystem
from backtesting_framework import AdvancedBacktester
from real_time_signals import RealTimeSignalGenerator, StreamingConfig

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for unified trading system integration"""
    enable_real_time: bool = True
    enable_backtesting: bool = True
    enable_paper_trading: bool = False
    enable_live_trading: bool = False

    # Risk management
    max_position_size: float = 0.1  # Max position as % of portfolio
    max_drawdown_limit: float = 0.05  # Max drawdown before stopping
    daily_loss_limit: float = 0.02  # Daily loss limit

    # Signal thresholds
    min_confidence: float = 0.6  # Minimum confidence for trade execution
    strong_signal_threshold: float = 0.8  # Threshold for strong signals

    # Update frequencies
    signal_update_freq: float = 1.0  # Signal update frequency (seconds)
    portfolio_update_freq: float = 5.0  # Portfolio update frequency (seconds)

    # Data sources
    data_sources: List[str] = None  # List of data source names

    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = ['yahoo', 'alpha_vantage', 'ibkr']

class PortfolioManager:
    """Portfolio management for unified trading system"""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position data
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0

    def get_position_size(self, symbol: str, price: float, signal_strength: float) -> float:
        """Calculate position size based on signal strength and risk management"""
        available_capital = self.current_capital * 0.1  # Use 10% of capital per trade
        position_value = available_capital * signal_strength

        # Limit position size
        max_position = self.current_capital * 0.05  # Max 5% of capital per position
        position_value = min(position_value, max_position)

        # Calculate shares
        shares = position_value / price
        return shares

    def open_position(self, symbol: str, shares: float, price: float, signal_data: Dict[str, Any]):
        """Open a new position"""
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return

        position = {
            'symbol': symbol,
            'shares': shares,
            'entry_price': price,
            'entry_time': datetime.now(),
            'current_price': price,
            'signal_data': signal_data,
            'pnl': 0.0,
            'status': 'open'
        }

        self.positions[symbol] = position
        cost = shares * price
        self.current_capital -= cost

        logger.info(f"Opened position: {symbol} {shares:.2f} shares @ ${price:.2f}")

    def close_position(self, symbol: str, price: float, reason: str = "manual"):
        """Close an existing position"""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return

        position = self.positions[symbol]
        entry_value = position['shares'] * position['entry_price']
        exit_value = position['shares'] * price
        pnl = exit_value - entry_value

        # Update position
        position.update({
            'exit_price': price,
            'exit_time': datetime.now(),
            'pnl': pnl,
            'status': 'closed',
            'close_reason': reason
        })

        # Update capital
        self.current_capital += exit_value
        self.daily_pnl += pnl

        # Record trade
        trade_record = position.copy()
        self.trade_history.append(trade_record)

        # Remove from active positions
        del self.positions[symbol]

        logger.info(f"Closed position: {symbol} PnL: ${pnl:.2f} ({reason})")

    def update_positions(self, current_prices: Dict[str, float]):
        """Update position values with current prices"""
        total_pnl = 0.0

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                old_price = position['current_price']
                new_price = current_prices[symbol]
                position['current_price'] = new_price

                # Update unrealized PnL
                position['pnl'] = position['shares'] * (new_price - position['entry_price'])
                total_pnl += position['pnl']

        return total_pnl

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Get total portfolio value"""
        position_value = sum(
            pos['shares'] * current_prices.get(pos['symbol'], pos['current_price'])
            for pos in self.positions.values()
        )
        return self.current_capital + position_value

    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Get portfolio statistics"""
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
        losing_trades = len([t for t in self.trade_history if t['pnl'] < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] < 0]) if losing_trades > 0 else 0

        return {
            'current_capital': self.current_capital,
            'total_positions': len(self.positions),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown
        }

class DataFeedManager:
    """Manages multiple data feeds for the unified system"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.data_feeds: Dict[str, Any] = {}  # feed_name -> feed_instance
        self.active_feeds: List[str] = []

    def add_feed(self, name: str, feed_instance: Any):
        """Add a data feed"""
        self.data_feeds[name] = feed_instance
        logger.info(f"Added data feed: {name}")

    def start_feeds(self, symbols: List[str]):
        """Start data feeds for given symbols"""
        for feed_name in self.config.data_sources:
            if feed_name in self.data_feeds:
                try:
                    self.data_feeds[feed_name].start(symbols)
                    self.active_feeds.append(feed_name)
                    logger.info(f"Started data feed: {feed_name}")
                except Exception as e:
                    logger.error(f"Failed to start feed {feed_name}: {e}")

    def stop_feeds(self):
        """Stop all active data feeds"""
        for feed_name in self.active_feeds:
            try:
                self.data_feeds[feed_name].stop()
                logger.info(f"Stopped data feed: {feed_name}")
            except Exception as e:
                logger.error(f"Failed to stop feed {feed_name}: {e}")

        self.active_feeds.clear()

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices from active feeds"""
        prices = {}

        for symbol in symbols:
            for feed_name in self.active_feeds:
                try:
                    price = self.data_feeds[feed_name].get_price(symbol)
                    if price is not None:
                        prices[symbol] = price
                        break  # Use first available price
                except Exception as e:
                    logger.error(f"Error getting price for {symbol} from {feed_name}: {e}")

        return prices

class UnifiedTradingSystem:
    """
    Unified trading system integrating all components
    Combines signal generation, portfolio management, and execution
    """

    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()

        # Core components
        self.signal_system = DualSignalTradingSystem()
        self.portfolio = PortfolioManager()
        self.data_manager = DataFeedManager(self.config)

        # Real-time components
        self.real_time_generator = None
        if self.config.enable_real_time:
            streaming_config = StreamingConfig(
                update_interval=self.config.signal_update_freq,
                enable_alerts=True
            )
            self.real_time_generator = RealTimeSignalGenerator(
                self.signal_system, streaming_config
            )

        # Backtesting component
        self.backtester = None
        if self.config.enable_backtesting:
            self.backtester = AdvancedBacktester()

        # Control flags
        self.running = False
        self.paper_trading = self.config.enable_paper_trading
        self.live_trading = self.config.enable_live_trading

        # Monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.last_portfolio_update = datetime.now()

        logger.info("Unified trading system initialized")

    def add_data_feed(self, name: str, feed: Any):
        """Add a data feed to the system"""
        self.data_manager.add_feed(name, feed)

    def initialize_system(self, symbols: List[str], historical_data: Dict[str, pd.DataFrame]):
        """Initialize the system with symbols and historical data"""
        logger.info(f"Initializing system for {len(symbols)} symbols")

        # Store historical data for signal generation
        self._historical_data = historical_data.copy()

        # Add data to real-time generator if enabled
        if self.real_time_generator:
            for symbol, data in historical_data.items():
                self.real_time_generator.add_market_data(symbol, data)

        # Start data feeds
        self.data_manager.start_feeds(symbols)

        logger.info("System initialization complete")

    def start_trading(self):
        """Start the unified trading system"""
        if self.running:
            logger.warning("System already running")
            return

        self.running = True

        # Start real-time signal generation
        if self.real_time_generator:
            self.real_time_generator.start_streaming()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("Unified trading system started")

    def stop_trading(self):
        """Stop the unified trading system"""
        self.running = False

        # Stop real-time generation
        if self.real_time_generator:
            self.real_time_generator.stop_streaming()

        # Stop data feeds
        self.data_manager.stop_feeds()

        # Wait for monitoring thread
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        logger.info("Unified trading system stopped")

    def _monitoring_loop(self):
        """Main monitoring and trading loop"""
        while self.running:
            try:
                # Get current prices
                symbols = list(self.real_time_generator.market_data.keys()) if self.real_time_generator else []
                current_prices = self.data_manager.get_latest_prices(symbols)

                if current_prices:
                    # Update portfolio positions
                    self.portfolio.update_positions(current_prices)

                    # Check for trading opportunities
                    self._check_trading_signals(current_prices)

                    # Risk management checks
                    self._risk_management_checks()

                # Update portfolio stats periodically
                if (datetime.now() - self.last_portfolio_update).seconds >= self.config.portfolio_update_freq:
                    self._update_portfolio_stats()
                    self.last_portfolio_update = datetime.now()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(1.0)  # Check every second

    def _check_trading_signals(self, current_prices: Dict[str, float]):
        """Check for trading signals and execute trades"""
        for symbol, price in current_prices.items():
            # Get latest signal from real-time generator
            if self.real_time_generator:
                signals = self.real_time_generator.get_signals_for_symbol(symbol, 1)
                if not signals:
                    continue

                signal_data = signals[0]
            else:
                # Generate signal directly using historical data
                if not hasattr(self, '_historical_data') or symbol not in self._historical_data:
                    continue

                data = self._historical_data[symbol]
                if len(data) < 50:
                    continue

                signals = self.signal_system.generate_signals(data)
                if len(signals) == 0:
                    continue

                signal_data = signals.iloc[-1].to_dict()
                signal_data['symbol'] = symbol
                signal_data['price'] = price

            # Check signal confidence
            confidence = signal_data.get('confidence_score', 0)
            if confidence < self.config.min_confidence:
                continue

            combined_signal = signal_data.get('combined_signal', 0)
            signal_strength = signal_data.get('position_size', 0.5)

            # Execute trades based on signal
            self._execute_trade_decision(symbol, price, combined_signal, signal_strength, signal_data)

    def _execute_trade_decision(self, symbol: str, price: float, signal: int,
                               signal_strength: float, signal_data: Dict[str, Any]):
        """Execute trade based on signal decision"""
        has_position = symbol in self.portfolio.positions

        if signal == 2 and not has_position:  # Strong buy signal
            # Open new position
            shares = self.portfolio.get_position_size(symbol, price, signal_strength)
            if shares > 0:
                self.portfolio.open_position(symbol, shares, price, signal_data)

        elif signal == 0 and has_position:  # Avoid signal (close position)
            self.portfolio.close_position(symbol, price, "signal_exit")

        elif signal == 1 and has_position:  # Weak buy - consider reducing position
            # Could implement position sizing adjustment here
            pass

    def _risk_management_checks(self):
        """Perform risk management checks"""
        portfolio_value = self.portfolio.get_portfolio_value({})

        # Check drawdown limit
        drawdown = (self.portfolio.initial_capital - portfolio_value) / self.portfolio.initial_capital
        if drawdown > self.config.max_drawdown_limit:
            logger.warning(".2%")
            # Could implement emergency stop logic here

        # Check daily loss limit
        if self.portfolio.daily_pnl < -self.portfolio.initial_capital * self.config.daily_loss_limit:
            logger.warning(".2%")
            # Could implement daily stop logic here

    def _update_portfolio_stats(self):
        """Update and log portfolio statistics"""
        stats = self.portfolio.get_portfolio_stats()
        portfolio_value = self.portfolio.get_portfolio_value({})

        logger.info(f"Portfolio Value: ${portfolio_value:.2f}, "
                   f"Cash: ${stats['current_capital']:.2f}, "
                   f"Positions: {stats['total_positions']}, "
                   f"Daily PnL: ${stats['daily_pnl']:.2f}")

    def run_backtest(self, test_data: Dict[str, pd.DataFrame],
                    start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtest on historical data"""
        if not self.backtester:
            raise ValueError("Backtesting not enabled")

        logger.info("Running backtest...")

        # For now, run backtest on the first symbol's data
        # In a full implementation, this would handle multiple symbols
        first_symbol = list(test_data.keys())[0]
        data = test_data[first_symbol]

        # Filter date range
        mask = (data.index >= start_date) & (data.index <= end_date)
        filtered_data = data[mask]

        # Run backtest
        results = self.backtester.run_backtest(
            self.signal_system, filtered_data
        )

        logger.info("Backtest completed")
        return results

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'running': self.running,
            'paper_trading': self.paper_trading,
            'live_trading': self.live_trading,
            'portfolio': self.portfolio.get_portfolio_stats(),
            'active_positions': list(self.portfolio.positions.keys()),
            'data_feeds': self.data_manager.active_feeds
        }

        if self.real_time_generator:
            status['signal_stats'] = self.real_time_generator.get_performance_stats()

        return status

# Convenience functions for easy setup
def create_unified_system(initial_capital: float = 100000.0) -> UnifiedTradingSystem:
    """Create a unified trading system with default settings"""
    config = IntegrationConfig()
    system = UnifiedTradingSystem(config)

    # Initialize portfolio
    system.portfolio = PortfolioManager(initial_capital)

    return system

def setup_basic_data_feeds(system: UnifiedTradingSystem, symbols: List[str]):
    """Setup basic data feed structure (placeholder for actual feeds)"""
    # This would be replaced with actual data feed implementations
    # For now, just log the setup
    logger.info(f"Setting up data feeds for symbols: {symbols}")

    # Placeholder feed class
    class PlaceholderFeed:
        def __init__(self):
            self.prices = {}

        def start(self, symbols):
            for symbol in symbols:
                self.prices[symbol] = 100.0  # Placeholder price

        def stop(self):
            pass

        def get_price(self, symbol):
            return self.prices.get(symbol, 100.0)

    # Add placeholder feeds
    for feed_name in system.config.data_sources:
        feed = PlaceholderFeed()
        system.add_data_feed(feed_name, feed)

# Example usage and testing
def example_unified_system():
    """
    Example of setting up and running the unified trading system
    """
    from sweet_spot_danger_zone_system import example_usage

    # Get trained system
    trained_system, signals, results = example_usage()

    # Create unified system
    unified_system = create_unified_system(100000.0)

    # Replace signal system with trained one
    unified_system.signal_system = trained_system

    # Setup symbols and data
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    # Generate sample historical data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1H')

    historical_data = {}
    for symbol in symbols:
        # Generate synthetic price data
        price = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, len(dates))))
        volume = np.random.lognormal(15, 0.5, len(dates))

        data = pd.DataFrame({
            'open': price,
            'high': price * (1 + np.random.normal(0, 0.005, len(dates))),
            'low': price * (1 + np.random.normal(0, -0.005, len(dates))),
            'close': price,
            'volume': volume
        }, index=dates)

        historical_data[symbol] = data

    # Setup data feeds
    setup_basic_data_feeds(unified_system, symbols)

    # Initialize system
    unified_system.initialize_system(symbols, historical_data)

    # Start trading
    unified_system.start_trading()

    try:
        # Run for demonstration
        for i in range(30):  # 30 seconds
            time.sleep(1)

            # Print status every 10 seconds
            if i % 10 == 0:
                status = unified_system.get_system_status()
                print(f"\nSystem Status at {i}s:")
                print(f"Running: {status['running']}")
                print(f"Portfolio Value: ${status['portfolio']['current_capital']:.2f}")
                print(f"Active Positions: {status['active_positions']}")
                if 'signal_stats' in status:
                    print(f"Signals Generated: {status['signal_stats']['update_count']}")

    finally:
        # Stop trading
        unified_system.stop_trading()

    # Run backtest
    print("\nRunning backtest...")
    backtest_results = unified_system.run_backtest(
        historical_data, '2024-01-15', '2024-01-30'
    )

    print("Backtest Results:")
    print(f"Total Return: {backtest_results.get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {backtest_results.get('max_drawdown', 0):.2%}")

    return unified_system

if __name__ == "__main__":
    system = example_unified_system()