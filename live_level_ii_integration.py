#!/usr/bin/env python3
"""
Level II Data Integration with Live IBKR Connection

This module connects to live IBKR Gateway using existing infrastructure
and integrates Level II data with Week 2 AI models.

Features:
- Live Level II market data streaming
- Real-time order book analysis
- PostgreSQL data storage
- AI model signal generation

Author: GitHub Copilot
Date: August 31, 2025
"""

import sys
import os
import logging
import time
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
from queue import Queue, Empty

# Add paths for imports
sys.path.append(r'c:\Users\nzcon\VSPython\TradeAppComponents_fresh')

try:
    # Import IBKR connection
    from ibkr_api.connect_me import get_managed_ibkr_connection
    from ibkr_api.client_id_registry import get_component_client_id
    ibkr_available = True
except ImportError as e:
    print(f"[WARNING] IBKR connection not available: {e}")
    ibkr_available = False

try:
    # Import database manager
    from modules.database.railway_db_manager import RailwayPostgreSQLManager
    database_available = True
except ImportError as e:
    print(f"[WARNING] Database not available: {e}")
    database_available = False

# Import our standalone AI models
from week2_level_ii_standalone_models import (
    LevelIIEnhancedPPOTrader,
    LevelIIEnhancedGeneticOptimizer, 
    LevelIIEnhancedSpectrumAnalyzer,
    LevelIIData,
    TradingSignal,
    generate_mock_level_ii_data
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveLevelIICollector:
    """
    Live Level II Data Collector using IBKR Gateway
    
    Features:
    - Real-time order book streaming
    - Order flow analysis
    - Market microstructure calculation
    - Database storage integration
    """
    
    def __init__(self, symbols: List[str], config: Dict[str, Any]):
        self.symbols = symbols
        self.config = config
        self.ib = None
        self.is_running = False
        self.data_queue = Queue()
        
        # Components
        self.db_manager = None
        self.ai_models = {}
        
        # Storage
        self.level_ii_data = {}
        self.order_book_history = {symbol: [] for symbol in symbols}
        
        logger.info(f"[SUCCESS] Initialized Level II Collector for symbols: {symbols}")
    
    def connect_to_ibkr(self) -> bool:
        """Connect to IBKR Gateway using existing infrastructure"""
        
        if not ibkr_available:
            logger.error("[ERROR] IBKR connection not available")
            return False
        
        try:
            logger.info("[PROCESSING] Connecting to IBKR Gateway...")
            
            # Use the existing managed connection
            self.ib = get_managed_ibkr_connection("level_ii_collector")
            
            if self.ib and self.ib.isConnected():
                logger.info(f"[SUCCESS] Connected to IBKR Gateway - Client ID: {self.ib.client.clientId}")
                return True
            else:
                logger.error("[ERROR] Failed to establish IBKR connection")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] IBKR connection failed: {str(e)}")
            return False
    
    def connect_to_database(self) -> bool:
        """Connect to Railway PostgreSQL database"""
        
        if not database_available:
            logger.warning("[WARNING] Database not available - using mock storage")
            return False
        
        try:
            logger.info("[PROCESSING] Connecting to Railway PostgreSQL...")
            self.db_manager = RailwayPostgreSQLManager()
            logger.info("[SUCCESS] Connected to Railway PostgreSQL database")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Database connection failed: {str(e)}")
            return False
    
    def initialize_ai_models(self):
        """Initialize the enhanced AI models"""
        
        logger.info("[PROCESSING] Initializing Level II Enhanced AI Models...")
        
        for symbol in self.symbols:
            model_config = {
                'symbol': symbol,
                'position_size_base': self.config.get('position_size_base', 100),
                'min_liquidity_score': self.config.get('min_liquidity_score', 0.3),
                'max_spread_threshold': self.config.get('max_spread_threshold', 0.005)
            }
            
            self.ai_models[symbol] = {
                'ppo_trader': LevelIIEnhancedPPOTrader(model_config),
                'genetic_optimizer': LevelIIEnhancedGeneticOptimizer(model_config),
                'spectrum_analyzer': LevelIIEnhancedSpectrumAnalyzer(model_config)
            }
        
        logger.info(f"[SUCCESS] Initialized AI models for {len(self.symbols)} symbols")
    
    def setup_market_data_subscriptions(self):
        """Subscribe to Level II market data for all symbols"""
        
        if not self.ib or not self.ib.isConnected():
            logger.error("[ERROR] No IBKR connection available for market data")
            return False
        
        try:
            from ib_insync import Stock
            
            logger.info("[PROCESSING] Setting up Level II market data subscriptions...")
            
            for symbol in self.symbols:
                # Create contract
                contract = Stock(symbol, 'SMART', 'USD')
                
                # Request market data with Level II
                self.ib.reqMktData(contract, '', False, False)
                
                # Request market depth (Level II order book)
                self.ib.reqMktDepth(contract, 10, False)  # 10 levels deep
                
                logger.info(f"[SUCCESS] Subscribed to Level II data for {symbol}")
            
            # Set up event handlers
            self.ib.pendingTickersEvent += self.on_ticker_update
            
            # Check if the DOM event exists (different versions of ib_insync)
            if hasattr(self.ib, 'domTickersEvent'):
                self.ib.domTickersEvent += self.on_dom_update
            elif hasattr(self.ib, 'domBidAskEvent'):
                self.ib.domBidAskEvent += self.on_dom_update
            else:
                logger.warning("[WARNING] DOM events not available - will use basic ticker data only")
            
            logger.info("[SUCCESS] Level II market data subscriptions active")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to setup market data subscriptions: {str(e)}")
            return False
    
    def on_ticker_update(self, tickers):
        """Handle ticker updates"""
        for ticker in tickers:
            if ticker.contract.symbol in self.symbols:
                # Process basic ticker data
                self.process_ticker_data(ticker)
    
    def on_dom_update(self, dom_tickers=None):
        """Handle Depth of Market (Level II) updates"""
        
        # Handle different event signatures
        if dom_tickers is None:
            # Check if we can get DOM data directly
            if hasattr(self.ib, 'domTickers') and self.ib.domTickers:
                dom_tickers = list(self.ib.domTickers.values())
            else:
                return
        
        # Process each DOM ticker
        if isinstance(dom_tickers, list):
            for dom_ticker in dom_tickers:
                if hasattr(dom_ticker, 'contract') and dom_ticker.contract.symbol in self.symbols:
                    self.process_level_ii_update(dom_ticker.contract.symbol, dom_ticker)
        else:
            # Single DOM ticker
            if hasattr(dom_tickers, 'contract') and dom_tickers.contract.symbol in self.symbols:
                self.process_level_ii_update(dom_tickers.contract.symbol, dom_tickers)
    
    def process_ticker_data(self, ticker):
        """Process basic ticker data"""
        symbol = ticker.contract.symbol
        
        # Helper function to handle NaN values
        def safe_float(value, default=None):
            try:
                if value is None or (hasattr(value, '__iter__') and len(str(value)) == 0):
                    return default
                import math
                if math.isnan(float(value)) or float(value)  0 and size > 0:
                                bid_levels.append({
                                    'price': price,
                                    'size': size,
                                    'market_maker': getattr(bid, 'marketMaker', '')
                                })
                        except (ValueError, TypeError) as e:
                            logger.debug(f"[WARNING] Invalid bid data for {symbol}: {e}")
                            continue
                    
                    # Process ask levels
                    for ask in dom_ticker.domAsks:
                        try:
                            price = float(ask.price) if hasattr(ask, 'price') else 0
                            size = int(float(ask.size)) if hasattr(ask, 'size') else 0
                            
                            if price > 0 and size > 0:
                                ask_levels.append({
                                    'price': price,
                                    'size': size,
                                    'market_maker': getattr(ask, 'marketMaker', '')
                                })
                        except (ValueError, TypeError) as e:
                            logger.debug(f"[WARNING] Invalid ask data for {symbol}: {e}")
                            continue
            
            # If no DOM levels, create synthetic ones from basic ticker
            if not bid_levels or not ask_levels:
                self.create_synthetic_dom_levels(symbol, bid_levels, ask_levels)
            
            # Sort levels
            bid_levels.sort(key=lambda x: x['price'], reverse=True)  # Highest bid first
            ask_levels.sort(key=lambda x: x['price'])  # Lowest ask first
            
            if len(bid_levels) > 0 and len(ask_levels) > 0:
                # Calculate Level II metrics
                level_ii_data = self.calculate_level_ii_metrics(symbol, bid_levels, ask_levels)
                
                # Store data
                self.level_ii_data[symbol]['level_ii'] = level_ii_data
                
                # Add to history
                self.order_book_history[symbol].append(level_ii_data)
                if len(self.order_book_history[symbol]) > 100:  # Keep last 100 updates
                    self.order_book_history[symbol].pop(0)
                
                # Generate AI signals
                self.generate_ai_signals(symbol, level_ii_data)
                
                # Store in database
                if self.db_manager:
                    self.store_level_ii_data(level_ii_data)
                
        except Exception as e:
            logger.error(f"[ERROR] Processing Level II update for {symbol}: {str(e)}")
    
    def create_synthetic_dom_levels(self, symbol: str, bid_levels: list, ask_levels: list):
        """Create synthetic DOM levels from basic ticker data when Level II is not available"""
        
        if symbol in self.level_ii_data and 'basic' in self.level_ii_data[symbol]:
            basic_data = self.level_ii_data[symbol]['basic']
            
            if basic_data['bid'] and basic_data['ask']:
                bid_price = basic_data['bid']
                ask_price = basic_data['ask']
                
                # Create synthetic levels around the BBO
                import random
                
                if not bid_levels:
                    for i in range(5):
                        price = bid_price - (i * 0.01)
                        size = random.randint(100, 1000)
                        bid_levels.append({
                            'price': price,
                            'size': size,
                            'market_maker': 'SYNTHETIC'
                        })
                
                if not ask_levels:
                    for i in range(5):
                        price = ask_price + (i * 0.01)
                        size = random.randint(100, 1000)
                        ask_levels.append({
                            'price': price,
                            'size': size,
                            'market_maker': 'SYNTHETIC'
                        })
    
    def calculate_level_ii_metrics(self, symbol: str, bid_levels: List[Dict], ask_levels: List[Dict]) -> LevelIIData:
        """Calculate comprehensive Level II metrics"""
        
        # Basic spread calculation
        best_bid = bid_levels[0]['price'] if bid_levels else 0
        best_ask = ask_levels[0]['price'] if ask_levels else 0
        spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
        
        # Order imbalance (top 5 levels)
        top_bid_size = sum(level['size'] for level in bid_levels[:5])
        top_ask_size = sum(level['size'] for level in ask_levels[:5])
        total_size = top_bid_size + top_ask_size
        depth_imbalance = top_bid_size / total_size if total_size > 0 else 0.5
        
        # Liquidity score
        total_liquidity = top_bid_size + top_ask_size
        liquidity_score = min(1.0, total_liquidity / 10000)  # Normalize to 0-1
        
        # Order flow analysis (simplified - would need tick-by-tick data for full analysis)
        order_flow = {
            'aggressive_buy_ratio': 0.5,  # Would calculate from actual trade data
            'net_flow_normalized': (top_bid_size - top_ask_size) / total_size if total_size > 0 else 0,
            'institutional_flow': 0.1  # Would detect from large order sizes
        }
        
        # Microstructure analysis
        microstructure = {
            'price_impact': spread / best_bid if best_bid > 0 else 0,
            'institutional_activity': sum(1 for level in bid_levels + ask_levels if level['size'] > 1000) / len(bid_levels + ask_levels),
            'effective_spread': spread
        }
        
        return LevelIIData(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            spread=spread,
            depth_imbalance=depth_imbalance,
            liquidity_score=liquidity_score,
            order_flow=order_flow,
            microstructure=microstructure
        )
    
    def generate_ai_signals(self, symbol: str, level_ii_data: LevelIIData):
        """Generate trading signals using AI models"""
        
        if symbol not in self.ai_models:
            return
        
        try:
            models = self.ai_models[symbol]
            signals = []
            
            # PPO Trader signals
            ppo_signal = models['ppo_trader'].generate_signal(level_ii_data, {})
            if ppo_signal:
                signals.append(ppo_signal)
                logger.info(f"[DATA] PPO Signal: {ppo_signal.action} {ppo_signal.symbol} - {ppo_signal.reason}")
            
            # Spectrum Analyzer signals
            spectrum_signals = models['spectrum_analyzer'].generate_spectrum_signals(level_ii_data)
            signals.extend(spectrum_signals)
            
            for signal in spectrum_signals:
                logger.info(f"[DATA] Spectrum Signal: {signal.action} {signal.symbol} - {signal.reason}")
            
            # Store signals
            for signal in signals:
                self.store_trading_signal(signal)
                
        except Exception as e:
            logger.error(f"[ERROR] Generating AI signals for {symbol}: {str(e)}")
    
    def store_level_ii_data(self, level_ii_data: LevelIIData):
        """Store Level II data in database"""
        
        if not self.db_manager:
            return
        
        try:
            # This would use the actual database schema
            # For now, just log the data
            logger.debug(f"[DATA] Storing Level II data for {level_ii_data.symbol}")
            
        except Exception as e:
            logger.error(f"[ERROR] Storing Level II data: {str(e)}")
    
    def store_trading_signal(self, signal: TradingSignal):
        """Store trading signal"""
        
        try:
            # For now, just log the signal
            logger.info(f"[SIGNAL] {signal.action} {signal.quantity} {signal.symbol} @ {signal.price} (confidence: {signal.confidence:.2f})")
            
        except Exception as e:
            logger.error(f"[ERROR] Storing trading signal: {str(e)}")
    
    def start_data_collection(self):
        """Start the Level II data collection process"""
        
        logger.info("="*80)
        logger.info("[STARTING] Live Level II Data Collection")
        logger.info("="*80)
        
        # Connect to IBKR
        if not self.connect_to_ibkr():
            logger.error("[ERROR] Failed to connect to IBKR - using mock data")
            self.run_mock_data_mode()
            return
        
        # Connect to database
        self.connect_to_database()
        
        # Initialize AI models
        self.initialize_ai_models()
        
        # Setup market data subscriptions
        if not self.setup_market_data_subscriptions():
            logger.error("[ERROR] Failed to setup market data - using mock data")
            self.run_mock_data_mode()
            return
        
        # Start the main loop
        logger.info("[SUCCESS] Starting live Level II data collection...")
        self.is_running = True
        
        try:
            while self.is_running:
                # Keep the connection alive and process events
                self.ib.sleep(1)  # Process IB events for 1 second
                
                # Optional: Process any queued data
                try:
                    while not self.data_queue.empty():
                        item = self.data_queue.get_nowait()
                        logger.info(f"[DATA] Processed queued item: {item}")
                except Empty:
                    pass
                
        except KeyboardInterrupt:
            logger.info("[WARNING] Received interrupt signal")
        except Exception as e:
            logger.error(f"[ERROR] Data collection error: {str(e)}")
        finally:
            self.stop_data_collection()
    
    def run_mock_data_mode(self):
        """Run in mock data mode for testing"""
        
        logger.info("[WARNING] Running in MOCK DATA MODE")
        logger.info("[PROCESSING] Initializing AI models...")
        
        self.initialize_ai_models()
        
        logger.info("[PROCESSING] Generating mock Level II data...")
        
        try:
            # Generate mock signals for demonstration
            for i in range(10):
                for symbol in self.symbols:
                    # Generate mock Level II data
                    mock_data = generate_mock_level_ii_data(symbol)
                    
                    # Generate AI signals
                    self.generate_ai_signals(symbol, mock_data)
                    
                    time.sleep(0.5)  # Simulate real-time updates
                
                logger.info(f"[DATA] Processed mock data cycle {i+1}/10")
                time.sleep(2)
            
            logger.info("[SUCCESS] Mock data mode completed")
            
        except Exception as e:
            logger.error(f"[ERROR] Mock data mode failed: {str(e)}")
    
    def stop_data_collection(self):
        """Stop the data collection process"""
        
        logger.info("[PROCESSING] Stopping Level II data collection...")
        self.is_running = False
        
        if self.ib and self.ib.isConnected():
            logger.info("[PROCESSING] Disconnecting from IBKR...")
            self.ib.disconnect()
            logger.info("[SUCCESS] Disconnected from IBKR")
        
        logger.info("[SUCCESS] Level II data collection stopped")

def main():
    """Main function to start Level II data integration"""
    
    logger.info("="*80)
    logger.info("[STARTING] Week 2 Level II Data Integration with Live IBKR")
    logger.info("="*80)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Configuration
    config = {
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'position_size_base': 100,
        'min_liquidity_score': 0.3,
        'max_spread_threshold': 0.005,
        'update_frequency': 1.0  # seconds
    }
    
    try:
        # Initialize collector
        collector = LiveLevelIICollector(config['symbols'], config)
        
        # Start data collection
        collector.start_data_collection()
        
        logger.info("")
        logger.info("="*80)
        logger.info("[SUCCESS] Level II Data Integration Completed")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"[ERROR] Level II integration failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
