#!/usr/bin/env python3
"""
Level II Data Integration for Week 2 AI Models

Synchronous IBKR Level II data streaming and storage for AI model training.
Uses existing Railway PostgreSQL infrastructure and synchronous IBKR connection.

Features:
- Level II order book streaming
- Order flow analysis
- Market microstructure detection
- PostgreSQL storage with AI-ready schema
- Synchronous architecture (no async/await mixing)

Author: GitHub Copilot
Date: August 31, 2025
"""

import sys
import os
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from collections import deque
import json

# Add the TradeAppComponents_fresh path for imports
sys.path.append(r'c:\Users\nzcon\VSPython\TradeAppComponents_fresh')

from ib_insync import IB, Stock, util
from ibkr_api.connect_me import get_managed_ibkr_connection
from modules.database.railway_db_manager import RailwayPostgreSQLManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LevelIIDataCollector:
    """
    Synchronous Level II data collector for AI model training
    
    Connects to IBKR Gateway and streams Level II order book data
    to Railway PostgreSQL for AI model consumption.
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['SPY', 'QQQ', 'TSLA', 'AAPL', 'NVDA']
        
        # Initialize database connection
        logger.info("[STARTING] Initializing Railway PostgreSQL connection...")
        self.db_manager = RailwayPostgreSQLManager()
        
        # Initialize IBKR connection
        logger.info("[STARTING] Initializing IBKR Gateway connection...")
        self.ib = get_managed_ibkr_connection("level_ii_collector")
        
        if not self.ib or not self.ib.isConnected():
            raise RuntimeError("Failed to connect to IBKR Gateway")
        
        logger.info(f"[SUCCESS] Connected to IBKR Gateway - Client ID: {self.ib.client.clientId}")
        
        # Data storage
        self.order_book_data = {}
        self.contracts = {}
        
        # Initialize database schema
        self._create_level_ii_tables()
        self._setup_contracts()
    
    def _create_level_ii_tables(self):
        """Create Level II data tables in PostgreSQL"""
        logger.info("[PROCESSING] Creating Level II database tables...")
        
        # Create schema for Level II data
        create_schema_sql = """
        CREATE SCHEMA IF NOT EXISTS level_ii_data;
        """
        
        # Order book snapshots table
        order_book_table_sql = """
        CREATE TABLE IF NOT EXISTS level_ii_data.order_book_snapshots (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            
            -- Bid side (top 10 levels)
            bid_levels JSONB NOT NULL,
            total_bid_volume BIGINT,
            avg_bid_price DECIMAL(12,4),
            
            -- Ask side (top 10 levels)
            ask_levels JSONB NOT NULL,
            total_ask_volume BIGINT,
            avg_ask_price DECIMAL(12,4),
            
            -- Spread analysis
            spread DECIMAL(12,6),
            spread_bps DECIMAL(8,2),
            midpoint DECIMAL(12,4),
            
            -- Order imbalance metrics
            order_imbalance DECIMAL(8,4),
            size_imbalance DECIMAL(8,4),
            
            -- Liquidity metrics
            liquidity_score DECIMAL(8,4),
            market_depth_score DECIMAL(8,4),
            
            -- AI model features
            volatility_indicator DECIMAL(8,4),
            momentum_signal DECIMAL(8,4),
            institutional_flow DECIMAL(8,4),
            
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_order_book_symbol_time 
        ON level_ii_data.order_book_snapshots(symbol, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_order_book_imbalance 
        ON level_ii_data.order_book_snapshots(order_imbalance) 
        WHERE ABS(order_imbalance) > 0.3;
        """
        
        # Order flow analysis table
        order_flow_table_sql = """
        CREATE TABLE IF NOT EXISTS level_ii_data.order_flow_analysis (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            
            -- Volume analysis
            aggressive_buy_volume BIGINT DEFAULT 0,
            aggressive_sell_volume BIGINT DEFAULT 0,
            passive_buy_volume BIGINT DEFAULT 0,
            passive_sell_volume BIGINT DEFAULT 0,
            
            -- Flow metrics
            net_flow BIGINT,
            flow_intensity DECIMAL(8,4),
            buy_pressure DECIMAL(8,4),
            sell_pressure DECIMAL(8,4),
            
            -- Institutional detection
            block_trades_detected INTEGER DEFAULT 0,
            avg_trade_size DECIMAL(12,2),
            large_order_ratio DECIMAL(6,4),
            
            -- AI features
            flow_acceleration DECIMAL(8,4),
            regime_classification VARCHAR(50),
            pattern_strength DECIMAL(6,4),
            
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_flow_symbol_time 
        ON level_ii_data.order_flow_analysis(symbol, timestamp DESC);
        """
        
        # Market microstructure table
        microstructure_table_sql = """
        CREATE TABLE IF NOT EXISTS level_ii_data.market_microstructure (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            
            -- Spread components
            quoted_spread DECIMAL(12,6),
            effective_spread DECIMAL(12,6),
            realized_spread DECIMAL(12,6),
            
            -- Price impact
            temporary_impact DECIMAL(10,6),
            permanent_impact DECIMAL(10,6),
            adverse_selection DECIMAL(10,6),
            
            -- Market quality
            price_improvement DECIMAL(10,6),
            fill_rate DECIMAL(6,4),
            execution_shortfall DECIMAL(10,6),
            
            -- Regime indicators
            volatility_regime VARCHAR(20),
            liquidity_regime VARCHAR(20),
            trend_strength DECIMAL(6,4),
            
            -- AI model inputs
            market_efficiency DECIMAL(6,4),
            information_content DECIMAL(6,4),
            noise_ratio DECIMAL(6,4),
            
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_microstructure_symbol_time 
        ON level_ii_data.market_microstructure(symbol, timestamp DESC);
        """
        
        try:
            with self.db_manager.get_session() as session:
                session.execute(create_schema_sql)
                session.execute(order_book_table_sql)
                session.execute(order_flow_table_sql)
                session.execute(microstructure_table_sql)
                session.commit()
                logger.info("[SUCCESS] Level II database tables created")
        except Exception as e:
            logger.error(f"[ERROR] Failed to create Level II tables: {e}")
            raise
    
    def _setup_contracts(self):
        """Setup IBKR contracts for Level II data"""
        logger.info("[PROCESSING] Setting up IBKR contracts...")
        
        for symbol in self.symbols:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            self.contracts[symbol] = contract
            self.order_book_data[symbol] = deque(maxlen=1000)
            
        logger.info(f"[SUCCESS] Configured contracts for {len(self.symbols)} symbols")
    
    def start_level_ii_streaming(self, num_rows: int = 10):
        """Start Level II market depth streaming for all symbols"""
        logger.info(f"[STARTING] Level II streaming for {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            contract = self.contracts[symbol]
            
            # Request Level II market depth
            try:
                ticker = self.ib.reqMktDepth(contract, numRows=num_rows, isSmartDepth=True)
                
                # Set up event handler for market depth updates
                ticker.updateEvent += self._create_depth_handler(symbol)
                
                logger.info(f"[SUCCESS] Started Level II streaming for {symbol}")
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to start Level II for {symbol}: {e}")
    
    def _create_depth_handler(self, symbol: str):
        """Create market depth update handler for symbol"""
        def on_depth_update(ticker):
            try:
                self._process_depth_update(symbol, ticker)
            except Exception as e:
                logger.error(f"[ERROR] Processing depth update for {symbol}: {e}")
        
        return on_depth_update
    
    def _process_depth_update(self, symbol: str, ticker):
        """Process Level II market depth update"""
        current_time = datetime.now(timezone.utc)
        
        # Extract bid and ask levels
        bid_levels = [(bid.price, bid.size) for bid in ticker.domBids[:10]]
        ask_levels = [(ask.price, ask.size) for ask in ticker.domAsks[:10]]
        
        if not bid_levels or not ask_levels:
            return
        
        # Calculate basic metrics
        best_bid = bid_levels[0][0] if bid_levels else 0
        best_ask = ask_levels[0][0] if ask_levels else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0
        midpoint = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        
        # Calculate order imbalance (top 5 levels)
        total_bid_size = sum(size for _, size in bid_levels[:5])
        total_ask_size = sum(size for _, size in ask_levels[:5])
        
        order_imbalance = 0
        if total_ask_size > 0:
            order_imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
        
        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(bid_levels, ask_levels)
        
        # Prepare data for storage
        order_book_data = {
            'symbol': symbol,
            'timestamp': current_time,
            'bid_levels': json.dumps(bid_levels),
            'ask_levels': json.dumps(ask_levels),
            'total_bid_volume': total_bid_size,
            'total_ask_volume': total_ask_size,
            'avg_bid_price': sum(price for price, _ in bid_levels) / len(bid_levels) if bid_levels else 0,
            'avg_ask_price': sum(price for price, _ in ask_levels) / len(ask_levels) if ask_levels else 0,
            'spread': spread,
            'spread_bps': (spread / midpoint * 10000) if midpoint > 0 else 0,
            'midpoint': midpoint,
            'order_imbalance': order_imbalance,
            'size_imbalance': self._calculate_size_imbalance(bid_levels, ask_levels),
            'liquidity_score': liquidity_score,
            'market_depth_score': self._calculate_depth_score(bid_levels, ask_levels),
            'volatility_indicator': self._calculate_volatility_indicator(symbol),
            'momentum_signal': self._calculate_momentum_signal(symbol, order_imbalance),
            'institutional_flow': self._detect_institutional_flow(bid_levels, ask_levels)
        }
        
        # Store in database
        self._store_order_book_data(order_book_data)
        
        # Store in memory for real-time analysis
        self.order_book_data[symbol].append(order_book_data)
        
        # Log significant imbalances
        if abs(order_imbalance) > 0.3:
            signal_type = "BULLISH" if order_imbalance > 0 else "BEARISH"
            logger.info(f"[SIGNAL] {signal_type} imbalance detected for {symbol}: {order_imbalance:.3f}")
    
    def _calculate_liquidity_score(self, bid_levels: List[Tuple], ask_levels: List[Tuple]) -> float:
        """Calculate liquidity score based on order book depth"""
        if not bid_levels or not ask_levels:
            return 0.0
        
        # Weight deeper levels less
        weights = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        weighted_bid_size = sum(size * weights[i] for i, (_, size) in enumerate(bid_levels[:10]))
        weighted_ask_size = sum(size * weights[i] for i, (_, size) in enumerate(ask_levels[:10]))
        
        return float((weighted_bid_size + weighted_ask_size) / 2)
    
    def _calculate_size_imbalance(self, bid_levels: List[Tuple], ask_levels: List[Tuple]) -> float:
        """Calculate size-weighted imbalance"""
        if not bid_levels or not ask_levels:
            return 0.0
        
        bid_weight = sum(price * size for price, size in bid_levels[:5])
        ask_weight = sum(price * size for price, size in ask_levels[:5])
        
        total_weight = bid_weight + ask_weight
        if total_weight == 0:
            return 0.0
        
        return float((bid_weight - ask_weight) / total_weight)
    
    def _calculate_depth_score(self, bid_levels: List[Tuple], ask_levels: List[Tuple]) -> float:
        """Calculate market depth score"""
        if len(bid_levels)  100])
        ask_depth = len([level for level in ask_levels if level[1] > 100])
        
        return float((bid_depth + ask_depth) / 20)  # Normalize to 0-1
    
    def _calculate_volatility_indicator(self, symbol: str) -> float:
        """Calculate short-term volatility indicator"""
        if symbol not in self.order_book_data or len(self.order_book_data[symbol])  float:
        """Calculate momentum signal from order flow"""
        if symbol not in self.order_book_data or len(self.order_book_data[symbol]) = 3:
            trend = sum(recent_imbalances[-3:]) / 3 - sum(recent_imbalances[-6:-3]) / 3
            return float(trend)
        
        return 0.0
    
    def _detect_institutional_flow(self, bid_levels: List[Tuple], ask_levels: List[Tuple]) -> float:
        """Detect institutional order flow patterns"""
        if not bid_levels or not ask_levels:
            return 0.0
        
        # Look for large orders (potential institutional activity)
        large_bid_orders = sum(1 for _, size in bid_levels if size >= 10000)
        large_ask_orders = sum(1 for _, size in ask_levels if size >= 10000)
        
        total_orders = len(bid_levels) + len(ask_levels)
        if total_orders == 0:
            return 0.0
        
        institutional_ratio = (large_bid_orders + large_ask_orders) / total_orders
        return float(institutional_ratio)
    
    def _store_order_book_data(self, data: Dict):
        """Store order book data in PostgreSQL"""
        insert_sql = """
        INSERT INTO level_ii_data.order_book_snapshots 
        (symbol, timestamp, bid_levels, ask_levels, total_bid_volume, total_ask_volume,
         avg_bid_price, avg_ask_price, spread, spread_bps, midpoint, order_imbalance,
         size_imbalance, liquidity_score, market_depth_score, volatility_indicator,
         momentum_signal, institutional_flow)
        VALUES (%(symbol)s, %(timestamp)s, %(bid_levels)s, %(ask_levels)s, %(total_bid_volume)s,
                %(total_ask_volume)s, %(avg_bid_price)s, %(avg_ask_price)s, %(spread)s,
                %(spread_bps)s, %(midpoint)s, %(order_imbalance)s, %(size_imbalance)s,
                %(liquidity_score)s, %(market_depth_score)s, %(volatility_indicator)s,
                %(momentum_signal)s, %(institutional_flow)s)
        """
        
        try:
            with self.db_manager.get_session() as session:
                session.execute(insert_sql, data)
                session.commit()
        except Exception as e:
            logger.error(f"[ERROR] Failed to store order book data: {e}")
    
    def get_ai_model_features(self, symbol: str, lookback_minutes: int = 5) -> Dict:
        """Get Level II features for AI model training"""
        cutoff_time = datetime.now(timezone.utc).replace(minute=datetime.now().minute - lookback_minutes)
        
        query_sql = """
        SELECT 
            AVG(order_imbalance) as avg_imbalance,
            STDDEV(order_imbalance) as imbalance_volatility,
            AVG(liquidity_score) as avg_liquidity,
            AVG(spread_bps) as avg_spread_bps,
            AVG(momentum_signal) as avg_momentum,
            AVG(institutional_flow) as avg_institutional_flow,
            COUNT(*) as sample_count
        FROM level_ii_data.order_book_snapshots 
        WHERE symbol = %(symbol)s AND timestamp >= %(cutoff_time)s
        """
        
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(query_sql, {
                    'symbol': symbol, 
                    'cutoff_time': cutoff_time
                }).fetchone()
                
                if result and result[6] > 0:  # sample_count > 0
                    return {
                        'avg_order_imbalance': float(result[0] or 0),
                        'imbalance_volatility': float(result[1] or 0),
                        'avg_liquidity': float(result[2] or 0),
                        'avg_spread_bps': float(result[3] or 0),
                        'avg_momentum': float(result[4] or 0),
                        'avg_institutional_flow': float(result[5] or 0),
                        'sample_count': int(result[6])
                    }
        except Exception as e:
            logger.error(f"[ERROR] Failed to get AI features for {symbol}: {e}")
        
        return {}
    
    def stop_streaming(self):
        """Stop Level II streaming and cleanup"""
        logger.info("[PROCESSING] Stopping Level II streaming...")
        
        for symbol in self.symbols:
            try:
                contract = self.contracts[symbol]
                self.ib.cancelMktDepth(contract)
                logger.info(f"[SUCCESS] Stopped Level II streaming for {symbol}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to stop streaming for {symbol}: {e}")
        
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("[SUCCESS] Disconnected from IBKR Gateway")
    
    def run_collection(self, duration_minutes: int = 60):
        """Run Level II data collection for specified duration"""
        logger.info(f"[STARTING] Level II data collection for {duration_minutes} minutes...")
        
        try:
            # Start streaming
            self.start_level_ii_streaming()
            
            # Run for specified duration
            end_time = time.time() + (duration_minutes * 60)
            sample_count = 0
            
            logger.info("[PROCESSING] Data collection in progress...")
            
            while time.time() < end_time:
                # Let the event loop process updates
                self.ib.sleep(1)
                sample_count += 1
                
                # Log progress every 60 seconds
                if sample_count % 60 == 0:
                    elapsed_minutes = (time.time() - (end_time - duration_minutes * 60)) / 60
                    logger.info(f"[PROCESSING] Collection progress: {elapsed_minutes:.1f}/{duration_minutes} minutes")
                    
                    # Log current imbalances
                    for symbol in self.symbols:
                        if symbol in self.order_book_data and self.order_book_data[symbol]:
                            latest = self.order_book_data[symbol][-1]
                            imbalance = latest['order_imbalance']
                            logger.info(f"[DATA] {symbol}: Imbalance={imbalance:.3f}, Liquidity={latest['liquidity_score']:.1f}")
            
            logger.info(f"[SUCCESS] Level II data collection completed - {duration_minutes} minutes")
            
        except KeyboardInterrupt:
            logger.info("[WARNING] Data collection interrupted by user")
        except Exception as e:
            logger.error(f"[ERROR] Data collection failed: {e}")
        finally:
            self.stop_streaming()


def main():
    """Main execution function"""
    print("=== Level II Data Integration for Week 2 AI Models ===")
    print("[STARTING] IBKR Level II Data Collection System")
    
    # High-volume symbols for Level II analysis
    symbols = ['SPY', 'QQQ', 'TSLA', 'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN']
    
    try:
        # Initialize collector
        collector = LevelIIDataCollector(symbols)
        
        # Run data collection
        print(f"[PROCESSING] Starting data collection for {len(symbols)} symbols...")
        print("Press Ctrl+C to stop collection early")
        
        # Collect data for 30 minutes initially
        collector.run_collection(duration_minutes=30)
        
        # Test AI model feature extraction
        print("[PROCESSING] Testing AI model feature extraction...")
        for symbol in symbols[:3]:  # Test first 3 symbols
            features = collector.get_ai_model_features(symbol)
            if features:
                print(f"[DATA] {symbol} AI Features: {features}")
        
        print("[SUCCESS] Level II data integration completed successfully!")
        
    except Exception as e:
        logger.error(f"[ERROR] Level II integration failed: {e}")
        print(f"[ERROR] Failed to run Level II integration: {e}")


if __name__ == "__main__":
    main()
