"""
Real-Time Signal Generation for Sweet Spot & Danger Zone System
Production-ready streaming signal generation with low latency
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import queue
import warnings

from sweet_spot_danger_zone_system import DualSignalTradingSystem, SweetSpotDetector, DangerZoneDetector

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class StreamingConfig:
    """Configuration for real-time signal generation"""
    update_interval: float = 1.0  # Seconds between signal updates
    lookback_window: int = 100  # Historical periods for feature calculation
    signal_buffer_size: int = 1000  # Maximum signals to keep in memory
    enable_alerts: bool = True  # Enable signal alerts
    alert_thresholds: Dict[str, float] = None  # Custom alert thresholds

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'strong_buy': 0.8,  # Sweet prob > 0.8 and danger prob < 0.2
                'weak_buy': 0.6,
                'danger': 0.7,      # Danger prob > 0.7
                'confidence': 0.75  # Overall confidence threshold
            }

class SignalBuffer:
    """Thread-safe buffer for storing recent signals"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()

    def add_signal(self, signal: Dict[str, Any]):
        """Add new signal to buffer"""
        try:
            self.buffer.put(signal, block=False)
        except queue.Full:
            # Remove oldest signal
            try:
                self.buffer.get(block=False)
                self.buffer.put(signal, block=False)
            except queue.Empty:
                pass

    def get_recent_signals(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent signals"""
        signals = []
        with self.lock:
            # Convert queue to list temporarily
            temp_signals = []
            while not self.buffer.empty():
                temp_signals.append(self.buffer.get())

            # Get most recent n signals
            signals = temp_signals[-n:] if len(temp_signals) >= n else temp_signals

            # Put signals back in queue
            for signal in temp_signals:
                try:
                    self.buffer.put(signal, block=False)
                except queue.Full:
                    break

        return signals

    def get_all_signals(self) -> List[Dict[str, Any]]:
        """Get all signals in buffer"""
        signals = []
        with self.lock:
            temp_signals = []
            while not self.buffer.empty():
                temp_signals.append(self.buffer.get())

            signals = temp_signals

            # Put signals back
            for signal in temp_signals:
                try:
                    self.buffer.put(signal, block=False)
                except queue.Full:
                    break

        return signals

class AlertSystem:
    """Real-time alert system for signal notifications"""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.alert_handlers: List[Callable] = []

    def add_alert_handler(self, handler: Callable):
        """Add alert handler function"""
        self.alert_handlers.append(handler)

    def trigger_alert(self, alert_type: str, signal_data: Dict[str, Any]):
        """Trigger alerts for all handlers"""
        if not self.config.enable_alerts:
            return

        alert_message = self._format_alert(alert_type, signal_data)

        for handler in self.alert_handlers:
            try:
                handler(alert_type, alert_message, signal_data)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

    def _format_alert(self, alert_type: str, signal_data: Dict[str, Any]) -> str:
        """Format alert message"""
        symbol = signal_data.get('symbol', 'UNKNOWN')
        timestamp = signal_data.get('timestamp', datetime.now())

        if alert_type == 'strong_buy':
            sweet_prob = signal_data.get('sweet_probability', 0)
            danger_prob = signal_data.get('danger_probability', 0)
            return f"🚀 STRONG BUY ALERT: {symbol} at {timestamp} (Sweet: {sweet_prob:.2f}, Danger: {danger_prob:.2f})"

        elif alert_type == 'weak_buy':
            sweet_prob = signal_data.get('sweet_probability', 0)
            return f"📈 WEAK BUY SIGNAL: {symbol} at {timestamp} (Sweet: {sweet_prob:.2f})"

        elif alert_type == 'danger':
            danger_prob = signal_data.get('danger_probability', 0)
            return f"⚠️ DANGER ALERT: {symbol} at {timestamp} (Danger: {danger_prob:.2f})"

        elif alert_type == 'high_confidence':
            confidence = signal_data.get('confidence_score', 0)
            return f"🎯 HIGH CONFIDENCE: {symbol} at {timestamp} (Confidence: {confidence:.2f})"

        return f"📊 SIGNAL: {alert_type.upper()} for {symbol}"

class RealTimeSignalGenerator:
    """
    Real-time signal generation engine
    Processes streaming market data and generates live signals
    """

    def __init__(self, system: DualSignalTradingSystem, config: StreamingConfig = None):
        self.system = system
        self.config = config or StreamingConfig()

        # Data management
        self.market_data: Dict[str, pd.DataFrame] = {}  # symbol -> historical data
        self.current_prices: Dict[str, float] = {}  # symbol -> latest price

        # Signal management
        self.signal_buffer = SignalBuffer(self.config.signal_buffer_size)
        self.alert_system = AlertSystem(self.config)

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.update_thread: Optional[threading.Thread] = None

        # Performance tracking
        self.last_update_time = None
        self.update_count = 0
        self.avg_update_time = 0.0

        logger.info("Real-time signal generator initialized")

    def add_market_data(self, symbol: str, data: pd.DataFrame):
        """Add historical market data for a symbol"""
        self.market_data[symbol] = data.copy()
        logger.info(f"Added market data for {symbol}: {len(data)} periods")

    def update_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None):
        """Update latest price for a symbol"""
        if timestamp is None:
            timestamp = datetime.now()

        self.current_prices[symbol] = price

        # Add to market data if we have the symbol
        if symbol in self.market_data:
            new_row = pd.DataFrame({
                'open': [price], 'high': [price], 'low': [price], 'close': [price],
                'volume': [0], 'timestamp': [timestamp]
            })
            new_row.set_index('timestamp', inplace=True)

            # Append to existing data
            self.market_data[symbol] = pd.concat([self.market_data[symbol], new_row])

            # Keep only recent data
            if len(self.market_data[symbol]) > self.config.lookback_window * 2:
                self.market_data[symbol] = self.market_data[symbol].tail(self.config.lookback_window * 2)

    def generate_signals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate signals for a specific symbol"""
        if symbol not in self.market_data:
            return None

        data = self.market_data[symbol]
        if len(data) < 50:  # Minimum data requirement
            return None

        try:
            # Generate signals using the dual system
            signals = self.system.generate_signals(data)

            if len(signals) == 0:
                return None

            # Get latest signal
            latest_signal = signals.iloc[-1]

            # Create signal data structure
            signal_data = {
                'symbol': symbol,
                'timestamp': data.index[-1],
                'price': data.iloc[-1]['close'],
                'combined_signal': latest_signal.get('combined_signal', 0),
                'position_size': latest_signal.get('position_size', 0),
                'confidence_score': latest_signal.get('confidence_score', 0),
                'sweet_probability': latest_signal.get('sweet_probability', 0.5),
                'danger_probability': latest_signal.get('danger_probability', 0.5),
                'sweet_prediction': latest_signal.get('sweet_prediction', 0),
                'danger_prediction': latest_signal.get('danger_prediction', 0)
            }

            return signal_data

        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return None

    def _update_signals(self):
        """Background thread function for continuous signal updates"""
        while self.running:
            start_time = time.time()

            try:
                # Generate signals for all symbols
                futures = []
                for symbol in self.market_data.keys():
                    future = self.executor.submit(self.generate_signals, symbol)
                    futures.append((symbol, future))

                # Collect results
                for symbol, future in futures:
                    try:
                        signal_data = future.result(timeout=5.0)
                        if signal_data:
                            # Add to buffer
                            self.signal_buffer.add_signal(signal_data)

                            # Check for alerts
                            self._check_alerts(signal_data)

                    except Exception as e:
                        logger.error(f"Error processing signals for {symbol}: {e}")

                # Update performance metrics
                update_time = time.time() - start_time
                self.update_count += 1
                self.avg_update_time = (self.avg_update_time * (self.update_count - 1) + update_time) / self.update_count
                self.last_update_time = datetime.now()

            except Exception as e:
                logger.error(f"Error in signal update loop: {e}")

            # Wait for next update
            time.sleep(self.config.update_interval)

    def _check_alerts(self, signal_data: Dict[str, Any]):
        """Check if signal triggers any alerts"""
        sweet_prob = signal_data.get('sweet_probability', 0)
        danger_prob = signal_data.get('danger_probability', 0)
        confidence = signal_data.get('confidence_score', 0)

        # Strong buy alert
        if (sweet_prob > self.config.alert_thresholds['strong_buy'] and
            danger_prob < 0.2):
            self.alert_system.trigger_alert('strong_buy', signal_data)

        # Weak buy alert
        elif sweet_prob > self.config.alert_thresholds['weak_buy']:
            self.alert_system.trigger_alert('weak_buy', signal_data)

        # Danger alert
        if danger_prob > self.config.alert_thresholds['danger']:
            self.alert_system.trigger_alert('danger', signal_data)

        # High confidence alert
        if confidence > self.config.alert_thresholds['confidence']:
            self.alert_system.trigger_alert('high_confidence', signal_data)

    def start_streaming(self):
        """Start real-time signal generation"""
        if self.running:
            logger.warning("Streaming already running")
            return

        self.running = True
        self.update_thread = threading.Thread(target=self._update_signals, daemon=True)
        self.update_thread.start()

        logger.info("Real-time signal generation started")

    def stop_streaming(self):
        """Stop real-time signal generation"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5.0)

        self.executor.shutdown(wait=True)
        logger.info("Real-time signal generation stopped")

    def get_latest_signals(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent signals"""
        return self.signal_buffer.get_recent_signals(n)

    def get_signals_for_symbol(self, symbol: str, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent signals for a specific symbol"""
        all_signals = self.signal_buffer.get_all_signals()
        symbol_signals = [s for s in all_signals if s.get('symbol') == symbol]
        return symbol_signals[-n:] if len(symbol_signals) >= n else symbol_signals

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'running': self.running,
            'last_update': self.last_update_time,
            'update_count': self.update_count,
            'avg_update_time': self.avg_update_time,
            'symbols_tracked': len(self.market_data),
            'buffer_size': len(self.signal_buffer.get_all_signals())
        }

class SignalDashboard:
    """Real-time signal monitoring dashboard"""

    def __init__(self, signal_generator: RealTimeSignalGenerator):
        self.signal_generator = signal_generator
        self.stats_history = []

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        perf_stats = self.signal_generator.get_performance_stats()
        latest_signals = self.signal_generator.get_latest_signals(20)

        # Calculate signal statistics
        if latest_signals:
            signal_df = pd.DataFrame(latest_signals)

            signal_stats = {
                'total_signals': len(latest_signals),
                'avg_confidence': signal_df['confidence_score'].mean(),
                'avg_sweet_prob': signal_df['sweet_probability'].mean(),
                'avg_danger_prob': signal_df['danger_probability'].mean(),
                'signal_distribution': signal_df['combined_signal'].value_counts().to_dict(),
                'high_confidence_signals': len(signal_df[signal_df['confidence_score'] > 0.8])
            }
        else:
            signal_stats = {
                'total_signals': 0,
                'avg_confidence': 0,
                'avg_sweet_prob': 0,
                'avg_danger_prob': 0,
                'signal_distribution': {},
                'high_confidence_signals': 0
            }

        # Symbol-specific stats
        symbol_stats = {}
        for symbol in self.signal_generator.market_data.keys():
            symbol_signals = self.signal_generator.get_signals_for_symbol(symbol, 10)
            if symbol_signals:
                symbol_df = pd.DataFrame(symbol_signals)
                symbol_stats[symbol] = {
                    'latest_signal': symbol_signals[-1]['combined_signal'],
                    'avg_confidence': symbol_df['confidence_score'].mean(),
                    'signal_count': len(symbol_signals)
                }

        return {
            'performance': perf_stats,
            'signal_stats': signal_stats,
            'symbol_stats': symbol_stats,
            'latest_signals': latest_signals[-5:]  # Last 5 signals
        }

    def print_dashboard(self):
        """Print dashboard to console"""
        data = self.get_dashboard_data()

        print("\n" + "="*60)
        print("SWEET SPOT & DANGER ZONE SIGNAL DASHBOARD")
        print("="*60)

        # Performance stats
        perf = data['performance']
        print(f"Status: {'RUNNING' if perf['running'] else 'STOPPED'}")
        print(f"Last Update: {perf['last_update']}")
        print(f"Update Count: {perf['update_count']}")
        print(".3f")
        print(f"Symbols Tracked: {perf['symbols_tracked']}")
        print(f"Buffer Size: {perf['buffer_size']}")

        # Signal stats
        sig = data['signal_stats']
        print(f"\nSignal Statistics (Last 20):")
        print(f"Total Signals: {sig['total_signals']}")
        print(".3f")
        print(".3f")
        print(".3f")
        print(f"High Confidence Signals: {sig['high_confidence_signals']}")
        print(f"Signal Distribution: {sig['signal_distribution']}")

        # Symbol stats
        print(f"\nSymbol Status:")
        for symbol, stats in data['symbol_stats'].items():
            signal_type = {0: 'AVOID', 1: 'WEAK BUY', 2: 'STRONG BUY'}.get(stats['latest_signal'], 'UNKNOWN')
            print(f"  {symbol}: {signal_type} (Conf: {stats['avg_confidence']:.2f}, Count: {stats['signal_count']})")

        # Latest signals
        print(f"\nLatest Signals:")
        for signal in data['latest_signals']:
            timestamp = signal['timestamp'].strftime('%H:%M:%S') if hasattr(signal['timestamp'], 'strftime') else str(signal['timestamp'])
            signal_type = {0: 'AVOID', 1: 'WEAK BUY', 2: 'STRONG BUY'}.get(signal['combined_signal'], 'UNKNOWN')
            print(f"  {timestamp} {signal['symbol']}: {signal_type} (Conf: {signal['confidence_score']:.2f})")

        print("="*60)

# Convenience functions for easy setup
def create_real_time_generator(system: DualSignalTradingSystem = None) -> RealTimeSignalGenerator:
    """Create a real-time signal generator with default settings"""
    if system is None:
        system = DualSignalTradingSystem()

    config = StreamingConfig()
    return RealTimeSignalGenerator(system, config)

def setup_alert_handlers(generator: RealTimeSignalGenerator):
    """Setup default alert handlers"""

    def console_alert_handler(alert_type: str, message: str, signal_data: Dict[str, Any]):
        """Print alerts to console"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def log_alert_handler(alert_type: str, message: str, signal_data: Dict[str, Any]):
        """Log alerts to file"""
        logging.info(f"ALERT: {alert_type} - {message}")

    generator.alert_system.add_alert_handler(console_alert_handler)
    generator.alert_system.add_alert_handler(log_alert_handler)

# Example usage and testing
def example_real_time_system():
    """
    Example of setting up and running the real-time signal generation system
    """
    from sweet_spot_danger_zone_system import example_usage

    # Get trained system
    system, signals, results = example_usage()

    # Create real-time generator
    generator = create_real_time_generator(system)

    # Setup alerts
    setup_alert_handlers(generator)

    # Add sample market data (replace with real data feed)
    dates = pd.date_range('2024-01-01', periods=200, freq='5min')
    np.random.seed(42)

    # Generate synthetic price data
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.005, len(dates))))
    volume = np.random.lognormal(15, 0.5, len(dates))

    sample_data = pd.DataFrame({
        'open': price,
        'high': price * (1 + np.random.normal(0, 0.002, len(dates))),
        'low': price * (1 + np.random.normal(0, -0.002, len(dates))),
        'close': price,
        'volume': volume
    }, index=dates)

    # Add data for multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    for symbol in symbols:
        # Add some symbol-specific variation
        variation = np.random.normal(0, 0.001, len(sample_data))
        symbol_data = sample_data.copy()
        symbol_data['close'] *= (1 + variation.cumsum())
        symbol_data['open'] *= (1 + variation.cumsum())
        symbol_data['high'] *= (1 + variation.cumsum())
        symbol_data['low'] *= (1 + variation.cumsum())

        generator.add_market_data(symbol, symbol_data)

    # Create dashboard
    dashboard = SignalDashboard(generator)

    # Start streaming
    generator.start_streaming()

    try:
        # Run for a few minutes to demonstrate
        for i in range(10):
            time.sleep(2)  # Wait 2 seconds

            # Simulate price updates
            for symbol in symbols:
                new_price = generator.market_data[symbol]['close'].iloc[-1] * (1 + np.random.normal(0, 0.001))
                generator.update_price(symbol, new_price)

            # Print dashboard
            if i % 3 == 0:  # Print every 6 seconds
                dashboard.print_dashboard()

    finally:
        # Stop streaming
        generator.stop_streaming()

    return generator, dashboard

if __name__ == "__main__":
    generator, dashboard = example_real_time_system()