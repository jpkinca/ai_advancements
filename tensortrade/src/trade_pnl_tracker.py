"""
trade_pnl_tracker.py
===================
Real-time P&L tracking for TensorTrade model trades.

Provides comprehensive trade-level profit/loss analysis including:
- Position entry/exit tracking
- Realized and unrealized P&L calculation
- Trade performance metrics
- Integration with existing database schema
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from db_utils import get_engine


@dataclass
class TradeEntry:
    """Individual trade record with complete P&L information"""
    trade_id: str
    episode_id: int
    symbol: str
    entry_timestamp: datetime
    exit_timestamp: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float  # Positive for long, negative for short
    realized_pnl: Optional[float]
    unrealized_pnl: Optional[float]
    holding_period_minutes: Optional[int]
    trade_type: str  # 'LONG', 'SHORT', 'COVER'
    commission_cost: float = 0.0
    slippage_cost: float = 0.0
    net_pnl: Optional[float] = None
    
    def __post_init__(self):
        if self.exit_price and self.entry_price:
            self.realized_pnl = self.calculate_realized_pnl()
            self.net_pnl = self.realized_pnl - self.commission_cost - self.slippage_cost
            
        if self.exit_timestamp and self.entry_timestamp:
            delta = self.exit_timestamp - self.entry_timestamp
            self.holding_period_minutes = int(delta.total_seconds() / 60)
    
    def calculate_realized_pnl(self) -> float:
        """Calculate realized P&L based on entry/exit prices"""
        if not self.exit_price:
            return 0.0
        
        if self.quantity > 0:  # Long position
            return (self.exit_price - self.entry_price) * abs(self.quantity)
        else:  # Short position
            return (self.entry_price - self.exit_price) * abs(self.quantity)
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L based on current market price"""
        if self.exit_price:  # Already closed
            return 0.0
            
        if self.quantity > 0:  # Long position
            return (current_price - self.entry_price) * abs(self.quantity)
        else:  # Short position
            return (self.entry_price - current_price) * abs(self.quantity)


@dataclass
class PositionSnapshot:
    """Current position state for a symbol"""
    symbol: str
    episode_id: int
    timestamp: datetime
    total_quantity: float
    average_cost: float
    market_value: float
    unrealized_pnl: float
    cost_basis: float
    open_trades: List[TradeEntry]


@dataclass
class PnLSummary:
    """Comprehensive P&L summary for episode or period"""
    episode_id: int
    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_realized_pnl: float
    total_unrealized_pnl: float
    total_commission: float
    total_slippage: float
    net_pnl: float
    gross_pnl: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_holding_period_minutes: float
    total_volume_traded: float


class TradePnLTracker:
    """
    Comprehensive P&L tracking for TensorTrade model decisions.
    
    Tracks individual trades, position changes, and provides real-time
    performance analytics for reinforcement learning trading decisions.
    """
    
    def __init__(self, engine: Engine, episode_id: int):
        self.engine = engine
        self.episode_id = episode_id
        self.active_positions: Dict[str, PositionSnapshot] = {}
        self.closed_trades: List[TradeEntry] = []
        self.trade_counter = 0
        self.commission_per_trade = 1.0  # Default $1 per trade
        self.slippage_bps = 2.0  # Default 2 basis points slippage
        
        # Create enhanced tables if they don't exist
        self._ensure_enhanced_tables()
    
    def _ensure_enhanced_tables(self):
        """Create enhanced P&L tracking tables"""
        with self.engine.begin() as conn:
            # Enhanced trades table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tt_trades (
                    id SERIAL PRIMARY KEY,
                    trade_id TEXT UNIQUE NOT NULL,
                    episode_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_timestamp TIMESTAMP NOT NULL,
                    exit_timestamp TIMESTAMP,
                    entry_price DOUBLE PRECISION NOT NULL,
                    exit_price DOUBLE PRECISION,
                    quantity DOUBLE PRECISION NOT NULL,
                    realized_pnl DOUBLE PRECISION,
                    unrealized_pnl DOUBLE PRECISION,
                    holding_period_minutes INTEGER,
                    trade_type TEXT NOT NULL,
                    commission_cost DOUBLE PRECISION DEFAULT 0.0,
                    slippage_cost DOUBLE PRECISION DEFAULT 0.0,
                    net_pnl DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Portfolio snapshots table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tt_portfolio_snapshots (
                    id SERIAL PRIMARY KEY,
                    episode_id INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    symbol TEXT NOT NULL,
                    total_quantity DOUBLE PRECISION NOT NULL,
                    average_cost DOUBLE PRECISION NOT NULL,
                    market_value DOUBLE PRECISION NOT NULL,
                    unrealized_pnl DOUBLE PRECISION NOT NULL,
                    cost_basis DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Indexes for performance
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tt_trades_episode_symbol 
                ON tt_trades (episode_id, symbol)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tt_portfolio_episode_timestamp 
                ON tt_portfolio_snapshots (episode_id, timestamp)
            """))
    
    def process_action(self, symbol: str, new_weight: float, current_price: float, 
                      timestamp: datetime, portfolio_value: float = 100000.0):
        """
        Process RL model action and update position tracking.
        
        Args:
            symbol: Trading symbol
            new_weight: Target portfolio weight (-1 to 1)
            current_price: Current market price
            timestamp: Action timestamp
            portfolio_value: Total portfolio value for position sizing
        """
        try:
            # Calculate target position size in shares
            target_value = new_weight * portfolio_value
            target_quantity = target_value / current_price if current_price > 0 else 0
            
            # Get current position
            current_position = self.active_positions.get(symbol)
            current_quantity = current_position.total_quantity if current_position else 0.0
            
            # Calculate quantity change
            quantity_change = target_quantity - current_quantity
            
            if abs(quantity_change) > 0.01:  # Minimum trade threshold
                # Calculate costs
                commission = self.commission_per_trade
                slippage = abs(quantity_change * current_price * self.slippage_bps / 10000)
                
                if quantity_change > 0:  # Increasing position (buy)
                    self._open_or_add_position(symbol, quantity_change, current_price, 
                                            timestamp, commission, slippage)
                else:  # Decreasing position (sell)
                    self._close_or_reduce_position(symbol, abs(quantity_change), 
                                                 current_price, timestamp, commission, slippage)
                
                # Update portfolio snapshot
                self._update_portfolio_snapshot(symbol, current_price, timestamp)
                
        except Exception as e:
            print(f"Error processing action for {symbol}: {e}")
    
    def _open_or_add_position(self, symbol: str, quantity: float, price: float,
                             timestamp: datetime, commission: float, slippage: float):
        """Open new position or add to existing position"""
        self.trade_counter += 1
        trade_id = f"{self.episode_id}_{symbol}_{self.trade_counter}"
        
        trade_type = "LONG" if quantity > 0 else "SHORT"
        
        trade = TradeEntry(
            trade_id=trade_id,
            episode_id=self.episode_id,
            symbol=symbol,
            entry_timestamp=timestamp,
            exit_timestamp=None,
            entry_price=price,
            exit_price=None,
            quantity=quantity,
            realized_pnl=None,
            unrealized_pnl=0.0,
            holding_period_minutes=None,
            trade_type=trade_type,
            commission_cost=commission,
            slippage_cost=slippage
        )
        
        # Update active position
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            # Calculate new average cost
            total_cost = (position.cost_basis + 
                         (quantity * price) + commission + slippage)
            total_quantity = position.total_quantity + quantity
            new_avg_cost = total_cost / abs(total_quantity) if total_quantity != 0 else price
            
            position.total_quantity = total_quantity
            position.average_cost = new_avg_cost
            position.cost_basis = total_cost
            position.open_trades.append(trade)
        else:
            # New position
            self.active_positions[symbol] = PositionSnapshot(
                symbol=symbol,
                episode_id=self.episode_id,
                timestamp=timestamp,
                total_quantity=quantity,
                average_cost=price,
                market_value=quantity * price,
                unrealized_pnl=0.0,
                cost_basis=(quantity * price) + commission + slippage,
                open_trades=[trade]
            )
        
        # Persist to database
        self._persist_trade(trade)
    
    def _close_or_reduce_position(self, symbol: str, quantity_to_close: float,
                                 price: float, timestamp: datetime, 
                                 commission: float, slippage: float):
        """Close or reduce existing position"""
        if symbol not in self.active_positions:
            return
        
        position = self.active_positions[symbol]
        trades_to_close = []
        remaining_to_close = quantity_to_close
        
        # Close trades FIFO (First In, First Out)
        for trade in position.open_trades:
            if remaining_to_close  0 else -close_quantity,
                realized_pnl=None,  # Will be calculated in __post_init__
                unrealized_pnl=0.0,
                holding_period_minutes=None,  # Will be calculated in __post_init__
                trade_type="COVER" if trade.quantity = trade_quantity:
                # Fully closed
                trades_to_close.append(trade)
                remaining_to_close -= trade_quantity
            else:
                # Partially closed - reduce original trade quantity
                trade.quantity = trade.quantity - (close_quantity if trade.quantity > 0 else -close_quantity)
                remaining_to_close = 0
            
            self.closed_trades.append(closed_trade)
            self._persist_trade(closed_trade)
        
        # Remove fully closed trades
        for trade in trades_to_close:
            position.open_trades.remove(trade)
        
        # Update position
        position.total_quantity -= quantity_to_close
        if abs(position.total_quantity)  Dict[str, float]:
        """Get current unrealized P&L for all positions"""
        pnl_by_symbol = {}
        
        for symbol, position in self.active_positions.items():
            current_price = current_prices.get(symbol, position.average_cost)
            unrealized_pnl = 0.0
            
            for trade in position.open_trades:
                unrealized_pnl += trade.calculate_unrealized_pnl(current_price)
            
            pnl_by_symbol[symbol] = unrealized_pnl
        
        return pnl_by_symbol
    
    def get_pnl_summary(self) -> PnLSummary:
        """Generate comprehensive P&L summary"""
        closed_trades = self.closed_trades
        
        if not closed_trades:
            return PnLSummary(
                episode_id=self.episode_id,
                period_start=datetime.now(),
                period_end=datetime.now(),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_realized_pnl=0.0,
                total_unrealized_pnl=0.0,
                total_commission=0.0,
                total_slippage=0.0,
                net_pnl=0.0,
                gross_pnl=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                average_holding_period_minutes=0.0,
                total_volume_traded=0.0
            )
        
        # Calculate metrics
        realized_pnls = [trade.realized_pnl for trade in closed_trades if trade.realized_pnl]
        winning_trades = [pnl for pnl in realized_pnls if pnl > 0]
        losing_trades = [pnl for pnl in realized_pnls if pnl  0 else float('inf'),
            average_win=sum(winning_trades) / len(winning_trades) if winning_trades else 0.0,
            average_loss=sum(losing_trades) / len(losing_trades) if losing_trades else 0.0,
            largest_win=max(winning_trades) if winning_trades else 0.0,
            largest_loss=min(losing_trades) if losing_trades else 0.0,
            average_holding_period_minutes=np.mean([
                trade.holding_period_minutes for trade in closed_trades 
                if trade.holding_period_minutes
            ]) if closed_trades else 0.0,
            total_volume_traded=sum(abs(trade.quantity * trade.entry_price) for trade in closed_trades)
        )
    
    def print_pnl_report(self):
        """Print formatted P&L report"""
        summary = self.get_pnl_summary()
        
        print("\n" + "="*80)
        print(f"📊 TRADE P&L SUMMARY - Episode {self.episode_id}")
        print("="*80)
        print(f"Period: {summary.period_start.strftime('%Y-%m-%d %H:%M')} to {summary.period_end.strftime('%Y-%m-%d %H:%M')}")
        print()
        print("💰 PROFIT & LOSS")
        print(f"   Realized P&L:    ${summary.total_realized_pnl:>12,.2f}")
        print(f"   Unrealized P&L:  ${summary.total_unrealized_pnl:>12,.2f}")
        print(f"   Commission:      ${summary.total_commission:>12,.2f}")
        print(f"   Slippage:        ${summary.total_slippage:>12,.2f}")
        print(f"   Net P&L:         ${summary.net_pnl:>12,.2f}")
        print()
        print("📈 PERFORMANCE METRICS")
        print(f"   Win Rate:               {summary.win_rate:>8.1%}")
        print(f"   Profit Factor:          {summary.profit_factor:>8.2f}")
        print(f"   Avg Holding Time:       {summary.average_holding_period_minutes:>8.0f} min")
        print()
        print("🎯 TRADE STATISTICS")
        print(f"   Total Trades:           {summary.total_trades:>8}")
        print(f"   Winning Trades:         {summary.winning_trades:>8}")
        print(f"   Losing Trades:          {summary.losing_trades:>8}")
        print(f"   Largest Win:     ${summary.largest_win:>12,.2f}")
        print(f"   Largest Loss:    ${summary.largest_loss:>12,.2f}")
        print(f"   Average Win:     ${summary.average_win:>12,.2f}")
        print(f"   Average Loss:    ${summary.average_loss:>12,.2f}")
        print(f"   Volume Traded:   ${summary.total_volume_traded:>12,.2f}")
        print("="*80)
