"""
Real-Time Market Data Streaming Module

This module provides real-time market data streaming capabilities for TensorTrade,
integrating with IBKR TWS/Gateway for live data feeds and order execution.

Author: TensorTrade Development Team
Created: August 17, 2025
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import threading
from queue import Queue, Empty

# Import IB API components (assuming ib_insync is available)
try:
    from ib_insync import IB, Stock, Contract, Ticker, BarData
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logging.warning("ib_insync not available. Real-time streaming will use simulation mode.")

from .db_utils import DatabaseManager


@dataclass
class MarketTick:
    """Individual market tick data"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    tick_type: str  # 'trade', 'bid', 'ask'


@dataclass
class MarketBar:
    """OHLCV bar data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    count: int


class BarAggregator:
    """Aggregates ticks into OHLCV bars"""
    
    def __init__(self, bar_interval: int = 60):
        """
        Initialize bar aggregator
        
        Args:
            bar_interval: Bar interval in seconds (default 60 for 1-minute bars)
        """
        self.bar_interval = bar_interval
        self.current_bars: Dict[str, Dict] = {}
        self.completed_bars: Queue = Queue()
        self.logger = logging.getLogger(__name__)
        
    def add_tick(self, tick: MarketTick) -> Optional[MarketBar]:
        """
        Add tick to aggregation and return completed bar if any
        
        Args:
            tick: Market tick data
            
        Returns:
            Completed bar if bar interval finished, None otherwise
        """
        symbol = tick.symbol
        bar_timestamp = self._get_bar_timestamp(tick.timestamp)
        
        # Initialize new bar if needed
        if symbol not in self.current_bars or self.current_bars[symbol]['timestamp'] != bar_timestamp:
            # Complete previous bar if exists
            if symbol in self.current_bars:
                completed_bar = self._complete_bar(symbol)
                if completed_bar:
                    self.completed_bars.put(completed_bar)
            
            # Start new bar
            self.current_bars[symbol] = {
                'timestamp': bar_timestamp,
                'open': tick.price,
                'high': tick.price,
                'low': tick.price,
                'close': tick.price,
                'volume': 0,
                'vwap_sum': 0,
                'count': 0
            }
        
        # Update current bar
        bar = self.current_bars[symbol]
        bar['high'] = max(bar['high'], tick.price)
        bar['low'] = min(bar['low'], tick.price)
        bar['close'] = tick.price
        bar['volume'] += tick.volume
        bar['vwap_sum'] += tick.price * tick.volume
        bar['count'] += 1
        
        return None
    
    def _get_bar_timestamp(self, timestamp: datetime) -> datetime:
        """Get bar timestamp by rounding down to bar interval"""
        seconds = timestamp.second - (timestamp.second % self.bar_interval)
        return timestamp.replace(second=seconds, microsecond=0)
    
    def _complete_bar(self, symbol: str) -> Optional[MarketBar]:
        """Complete current bar for symbol"""
        if symbol not in self.current_bars:
            return None
            
        bar_data = self.current_bars[symbol]
        
        # Calculate VWAP
        vwap = bar_data['vwap_sum'] / bar_data['volume'] if bar_data['volume'] > 0 else bar_data['close']
        
        return MarketBar(
            symbol=symbol,
            timestamp=bar_data['timestamp'],
            open=bar_data['open'],
            high=bar_data['high'],
            low=bar_data['low'],
            close=bar_data['close'],
            volume=bar_data['volume'],
            vwap=vwap,
            count=bar_data['count']
        )
    
    def get_completed_bars(self) -> List[MarketBar]:
        """Get all completed bars"""
        bars = []
        while not self.completed_bars.empty():
            try:
                bars.append(self.completed_bars.get_nowait())
            except Empty:
                break
        return bars


class IBKRRealTimeStreamer:
    """Real-time market data streaming from IBKR"""
    
    def __init__(self, 
                 host: str = '127.0.0.1',
                 port: int = 7497,
                 client_id: int = 1,
                 bar_interval: int = 60):
        """
        Initialize IBKR real-time streamer
        
        Args:
            host: TWS/Gateway host
            port: TWS/Gateway port (7497 for TWS, 7496 for paper trading)
            client_id: Unique client ID
            bar_interval: Bar aggregation interval in seconds
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.bar_interval = bar_interval
        
        self.ib = None
        self.is_connected = False
        self.streaming_symbols: List[str] = []
        self.bar_aggregator = BarAggregator(bar_interval)
        self.tick_handlers: List[Callable] = []
        self.bar_handlers: List[Callable] = []
        
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.latest_ticks: Dict[str, MarketTick] = {}
        self.latest_bars: Dict[str, MarketBar] = {}
        
        # Threading
        self.streaming_thread = None
        self.is_streaming = False
        
    async def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway"""
        if not IB_AVAILABLE:
            self.logger.warning("IB API not available, using simulation mode")
            return False
            
        try:
            self.ib = IB()
            await self.ib.connectAsync(self.host, self.port, self.client_id)
            self.is_connected = True
            self.logger.info(f"Connected to IBKR at {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.ib and self.is_connected:
            self.ib.disconnect()
            self.is_connected = False
            self.logger.info("Disconnected from IBKR")
    
    def add_tick_handler(self, handler: Callable[[MarketTick], None]):
        """Add tick data handler"""
        self.tick_handlers.append(handler)
    
    def add_bar_handler(self, handler: Callable[[MarketBar], None]):
        """Add bar data handler"""
        self.bar_handlers.append(handler)
    
    async def start_streaming(self, symbols: List[str]):
        """Start real-time data streaming for symbols"""
        if not self.is_connected:
            self.logger.error("Not connected to IBKR")
            return False
            
        self.streaming_symbols = symbols
        self.is_streaming = True
        
        # Create contracts and request market data
        for symbol in symbols:
            contract = Stock(symbol, 'SMART', 'USD')
            await self.ib.qualifyContractsAsync(contract)
            
            # Request real-time market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            ticker.updateEvent += self._on_ticker_update
            
        self.logger.info(f"Started streaming for symbols: {symbols}")
        return True
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.is_streaming = False
        if self.ib and self.is_connected:
            self.ib.cancelMktData()
        self.logger.info("Stopped real-time streaming")
    
    def _on_ticker_update(self, ticker: Ticker):
        """Handle ticker updates from IBKR"""
        if not self.is_streaming:
            return
            
        symbol = ticker.contract.symbol
        now = datetime.now()
        
        # Create tick from ticker data
        if ticker.last and ticker.lastSize:
            tick = MarketTick(
                symbol=symbol,
                timestamp=now,
                price=ticker.last,
                volume=ticker.lastSize,
                bid=ticker.bid if ticker.bid else ticker.last,
                ask=ticker.ask if ticker.ask else ticker.last,
                tick_type='trade'
            )
            
            # Store latest tick
            self.latest_ticks[symbol] = tick
            
            # Process through bar aggregator
            completed_bar = self.bar_aggregator.add_tick(tick)
            if completed_bar:
                self.latest_bars[symbol] = completed_bar
                # Notify bar handlers
                for handler in self.bar_handlers:
                    try:
                        handler(completed_bar)
                    except Exception as e:
                        self.logger.error(f"Error in bar handler: {e}")
            
            # Notify tick handlers
            for handler in self.tick_handlers:
                try:
                    handler(tick)
                except Exception as e:
                    self.logger.error(f"Error in tick handler: {e}")
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all streaming symbols"""
        return {symbol: tick.price for symbol, tick in self.latest_ticks.items()}
    
    def get_latest_bars(self) -> Dict[str, MarketBar]:
        """Get latest completed bars"""
        return self.latest_bars.copy()


class SimulationStreamer:
    """Simulation streamer for testing when IBKR not available"""
    
    def __init__(self, symbols: List[str], bar_interval: int = 60):
        self.symbols = symbols
        self.bar_interval = bar_interval
        self.is_streaming = False
        self.tick_handlers: List[Callable] = []
        self.bar_handlers: List[Callable] = []
        self.logger = logging.getLogger(__name__)
        
        # Simulation data
        self.current_prices = {symbol: 100.0 for symbol in symbols}
        self.bar_aggregator = BarAggregator(bar_interval)
        
    async def connect(self) -> bool:
        """Simulate connection"""
        self.logger.info("Using simulation mode - connected")
        return True
    
    def disconnect(self):
        """Simulate disconnection"""
        self.logger.info("Simulation mode - disconnected")
    
    def add_tick_handler(self, handler: Callable[[MarketTick], None]):
        """Add tick handler"""
        self.tick_handlers.append(handler)
    
    def add_bar_handler(self, handler: Callable[[MarketBar], None]):
        """Add bar handler"""
        self.bar_handlers.append(handler)
    
    async def start_streaming(self, symbols: List[str]):
        """Start simulation streaming"""
        self.symbols = symbols
        self.is_streaming = True
        self.logger.info(f"Started simulation streaming for: {symbols}")
        
        # Start simulation thread
        self.streaming_thread = threading.Thread(target=self._simulation_loop)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        
        return True
    
    def stop_streaming(self):
        """Stop simulation streaming"""
        self.is_streaming = False
        self.logger.info("Stopped simulation streaming")
    
    def _simulation_loop(self):
        """Simulation loop generating fake tick data"""
        while self.is_streaming:
            for symbol in self.symbols:
                # Generate random price movement
                price_change = np.random.normal(0, 0.001) * self.current_prices[symbol]
                self.current_prices[symbol] += price_change
                
                # Create simulation tick
                tick = MarketTick(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=self.current_prices[symbol],
                    volume=np.random.randint(100, 1000),
                    bid=self.current_prices[symbol] * 0.999,
                    ask=self.current_prices[symbol] * 1.001,
                    tick_type='trade'
                )
                
                # Process tick
                completed_bar = self.bar_aggregator.add_tick(tick)
                if completed_bar:
                    for handler in self.bar_handlers:
                        try:
                            handler(completed_bar)
                        except Exception as e:
                            self.logger.error(f"Error in bar handler: {e}")
                
                for handler in self.tick_handlers:
                    try:
                        handler(tick)
                    except Exception as e:
                        self.logger.error(f"Error in tick handler: {e}")
            
            time.sleep(1)  # 1 second between updates
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Get current simulation prices"""
        return self.current_prices.copy()


class TensorTradeRealTimeEngine:
    """Main real-time engine coordinating streaming and RL decisions"""
    
    def __init__(self, 
                 symbols: List[str],
                 db_manager: DatabaseManager,
                 use_simulation: bool = False):
        """
        Initialize real-time engine
        
        Args:
            symbols: List of symbols to stream
            db_manager: Database manager instance
            use_simulation: Use simulation mode instead of IBKR
        """
        self.symbols = symbols
        self.db_manager = db_manager
        self.use_simulation = use_simulation
        
        # Initialize streamer
        if use_simulation or not IB_AVAILABLE:
            self.streamer = SimulationStreamer(symbols)
        else:
            self.streamer = IBKRRealTimeStreamer()
        
        # Add handlers
        self.streamer.add_tick_handler(self._on_tick_received)
        self.streamer.add_bar_handler(self._on_bar_received)
        
        self.logger = logging.getLogger(__name__)
        
        # RL integration
        self.rl_model = None
        self.feature_engine = None
        self.last_features = None
        self.decision_handlers: List[Callable] = []
        
        # Performance tracking
        self.tick_count = 0
        self.bar_count = 0
        self.decision_count = 0
        self.start_time = None
    
    async def start(self):
        """Start real-time engine"""
        self.start_time = datetime.now()
        
        # Connect to data source
        connected = await self.streamer.connect()
        if not connected:
            self.logger.error("Failed to connect to data source")
            return False
        
        # Start streaming
        success = await self.streamer.start_streaming(self.symbols)
        if success:
            self.logger.info("TensorTrade real-time engine started successfully")
        
        return success
    
    def stop(self):
        """Stop real-time engine"""
        self.streamer.stop_streaming()
        self.streamer.disconnect()
        
        # Log performance stats
        if self.start_time:
            runtime = datetime.now() - self.start_time
            self.logger.info(f"Engine runtime: {runtime}")
            self.logger.info(f"Processed: {self.tick_count} ticks, {self.bar_count} bars, {self.decision_count} decisions")
    
    def set_rl_model(self, model):
        """Set RL model for decision making"""
        self.rl_model = model
        self.logger.info("RL model attached to real-time engine")
    
    def set_feature_engine(self, feature_engine):
        """Set feature engineering component"""
        self.feature_engine = feature_engine
        self.logger.info("Feature engine attached to real-time engine")
    
    def add_decision_handler(self, handler: Callable[[np.ndarray, List[str]], None]):
        """Add handler for RL decisions"""
        self.decision_handlers.append(handler)
    
    def _on_tick_received(self, tick: MarketTick):
        """Handle incoming tick data"""
        self.tick_count += 1
        
        # Store tick in database (optional - might be too frequent)
        # self.db_manager.store_tick(tick)
        
        # Log occasionally
        if self.tick_count % 100 == 0:
            self.logger.debug(f"Processed {self.tick_count} ticks")
    
    def _on_bar_received(self, bar: MarketBar):
        """Handle completed bar data"""
        self.bar_count += 1
        
        # Store bar in database
        self._store_bar_in_db(bar)
        
        # Trigger RL decision if model available
        if self.rl_model and self.feature_engine:
            self._make_rl_decision(bar)
        
        self.logger.debug(f"Processed bar: {bar.symbol} at {bar.timestamp}, Price: {bar.close:.2f}")
    
    def _store_bar_in_db(self, bar: MarketBar):
        """Store bar data in database"""
        try:
            # Convert to database format
            bar_data = {
                'symbol': bar.symbol,
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'vwap': bar.vwap
            }
            
            # Insert into tt_prices table
            self.db_manager.insert_price_data([bar_data])
            
        except Exception as e:
            self.logger.error(f"Error storing bar in database: {e}")
    
    def _make_rl_decision(self, bar: MarketBar):
        """Make RL decision based on new bar data"""
        try:
            # Get recent data for features
            recent_data = self.db_manager.get_recent_data(
                symbols=self.symbols,
                lookback_periods=100  # Get last 100 bars for features
            )
            
            if recent_data.empty:
                return
            
            # Generate features
            features = self.feature_engine.compute_features(recent_data)
            if features is None or features.empty:
                return
            
            # Make prediction
            actions = self.rl_model.predict(features.values[-1].reshape(1, -1))
            
            # Notify decision handlers
            for handler in self.decision_handlers:
                try:
                    handler(actions[0], self.symbols)
                except Exception as e:
                    self.logger.error(f"Error in decision handler: {e}")
            
            self.decision_count += 1
            self.logger.debug(f"Made RL decision #{self.decision_count}: {actions[0]}")
            
        except Exception as e:
            self.logger.error(f"Error making RL decision: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        return {
            'runtime_seconds': runtime.total_seconds(),
            'ticks_processed': self.tick_count,
            'bars_processed': self.bar_count,
            'decisions_made': self.decision_count,
            'ticks_per_second': self.tick_count / max(runtime.total_seconds(), 1),
            'bars_per_minute': self.bar_count / max(runtime.total_seconds() / 60, 1),
            'decisions_per_hour': self.decision_count / max(runtime.total_seconds() / 3600, 1)
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_real_time_engine():
        """Test the real-time engine"""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Initialize database
        db_manager = DatabaseManager()
        
        # Create real-time engine (using simulation)
        engine = TensorTradeRealTimeEngine(
            symbols=symbols,
            db_manager=db_manager,
            use_simulation=True
        )
        
        # Add decision handler
        def on_decision(actions, symbols):
            print(f"RL Decision: {dict(zip(symbols, actions))}")
        
        engine.add_decision_handler(on_decision)
        
        # Start engine
        print("Starting real-time engine...")
        await engine.start()
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Stop engine
        engine.stop()
        
        # Print stats
        stats = engine.get_performance_stats()
        print("Performance Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Run test
    asyncio.run(test_real_time_engine())
