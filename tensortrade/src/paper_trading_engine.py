"""
Paper Trading Engine for TensorTrade

This module provides a comprehensive paper trading environment for testing
RL trading strategies without risking real capital. Includes realistic
market simulation, execution costs, and performance tracking.

Author: TensorTrade Development Team
Created: August 17, 2025
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import uuid
import time

from .db_utils import DatabaseManager
from .enhanced_risk_module import EnhancedRiskManager, Position, Portfolio, RiskLimits


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    
    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        return self.status == OrderStatus.FILLED


@dataclass
class Fill:
    """Order fill/execution"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    
    @property
    def gross_amount(self) -> float:
        """Gross trade amount"""
        return self.quantity * self.price
    
    @property
    def net_amount(self) -> float:
        """Net trade amount after commission"""
        return self.gross_amount - self.commission


@dataclass
class Trade:
    """Completed trade (can consist of multiple fills)"""
    trade_id: str
    symbol: str
    side: OrderSide
    total_quantity: float
    avg_price: float
    total_commission: float
    timestamp: datetime
    fills: List[Fill] = field(default_factory=list)
    
    @property
    def gross_amount(self) -> float:
        return self.total_quantity * self.avg_price
    
    @property
    def net_amount(self) -> float:
        return self.gross_amount - self.total_commission


class CommissionStructure:
    """Commission/fee calculation"""
    
    def __init__(self,
                 per_share: float = 0.005,  # $0.005 per share
                 minimum: float = 1.0,      # $1.00 minimum
                 maximum: Optional[float] = None):
        self.per_share = per_share
        self.minimum = minimum
        self.maximum = maximum
    
    def calculate(self, quantity: float, price: float) -> float:
        """Calculate commission for trade"""
        commission = abs(quantity) * self.per_share
        commission = max(commission, self.minimum)
        
        if self.maximum:
            commission = min(commission, self.maximum)
        
        return commission


class MarketSimulator:
    """Simulates market conditions and price movements"""
    
    def __init__(self, 
                 spread_bps: float = 5.0,      # 5 basis points spread
                 slippage_bps: float = 2.0,    # 2 basis points slippage
                 volatility: float = 0.02):    # 2% daily volatility
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.volatility = volatility
        
        # Market state
        self.current_prices: Dict[str, float] = {}
        self.bid_ask_spreads: Dict[str, Tuple[float, float]] = {}
        
    def set_market_price(self, symbol: str, price: float):
        """Set current market price for symbol"""
        self.current_prices[symbol] = price
        
        # Calculate bid/ask spread
        spread_dollar = price * (self.spread_bps / 10000)
        bid = price - spread_dollar / 2
        ask = price + spread_dollar / 2
        self.bid_ask_spreads[symbol] = (bid, ask)
    
    def get_execution_price(self, 
                          symbol: str, 
                          side: OrderSide, 
                          quantity: float) -> float:
        """Get realistic execution price including spread and slippage"""
        if symbol not in self.current_prices:
            raise ValueError(f"No market price for {symbol}")
        
        market_price = self.current_prices[symbol]
        bid, ask = self.bid_ask_spreads[symbol]
        
        # Start with bid/ask
        if side == OrderSide.BUY:
            base_price = ask
        else:
            base_price = bid
        
        # Add slippage based on quantity (simplified model)
        slippage_factor = min(abs(quantity) / 1000, 0.1)  # Max 10% slippage impact
        slippage_bps_adjusted = self.slippage_bps * slippage_factor
        slippage_dollar = market_price * (slippage_bps_adjusted / 10000)
        
        if side == OrderSide.BUY:
            execution_price = base_price + slippage_dollar
        else:
            execution_price = base_price - slippage_dollar
        
        return execution_price
    
    def can_fill_order(self, order: Order) -> bool:
        """Check if order can be filled at current market conditions"""
        if order.symbol not in self.current_prices:
            return False
        
        market_price = self.current_prices[order.symbol]
        
        if order.order_type == OrderType.MARKET:
            return True
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                return market_price = order.price
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                return market_price >= order.stop_price
            else:
                return market_price  Optional[str]:
        """
        Place trading order
        
        Returns:
            Order ID if order placed successfully, None if rejected
        """
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Create order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        # Risk validation
        if self.risk_manager:
            execution_price = self._estimate_execution_price(order)
            is_valid, violations = self.risk_manager.validate_trade_request(
                symbol, quantity if side == OrderSide.BUY else -quantity, execution_price
            )
            
            if not is_valid:
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"Order rejected: {', '.join(violations)}")
                self._notify_order_callbacks(order)
                return None
        
        # Cash validation for buy orders
        if side == OrderSide.BUY and order_type == OrderType.MARKET:
            estimated_cost = self._estimate_order_cost(order)
            if estimated_cost > self.cash:
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"Insufficient cash for order: need ${estimated_cost:.2f}, have ${self.cash:.2f}")
                self._notify_order_callbacks(order)
                return None
        
        # Position validation for sell orders
        if side == OrderSide.SELL:
            current_position = self.positions.get(symbol)
            if not current_position or current_position.quantity  bool:
        """Cancel pending order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        self.logger.info(f"Order cancelled: {order_id}")
        self._notify_order_callbacks(order)
        
        return True
    
    def _estimate_execution_price(self, order: Order) -> float:
        """Estimate execution price for order"""
        if order.order_type == OrderType.MARKET:
            return self.market_simulator.get_execution_price(
                order.symbol, order.side, order.quantity
            )
        elif order.order_type == OrderType.LIMIT:
            return order.price
        else:
            # For stop orders, use current market price as estimate
            return self.market_simulator.current_prices.get(order.symbol, 0.0)
    
    def _estimate_order_cost(self, order: Order) -> float:
        """Estimate total cost of order including commission"""
        execution_price = self._estimate_execution_price(order)
        gross_cost = order.quantity * execution_price
        commission = self.commission_structure.calculate(order.quantity, execution_price)
        return gross_cost + commission
    
    def _process_pending_orders(self):
        """Process all pending orders against current market prices"""
        for order in self.orders.values():
            if order.status == OrderStatus.PENDING:
                self._try_fill_order(order)
    
    def _try_fill_order(self, order: Order):
        """Try to fill an order"""
        if not self.market_simulator.can_fill_order(order):
            return
        
        # Calculate execution price
        execution_price = self.market_simulator.get_execution_price(
            order.symbol, order.side, order.remaining_quantity
        )
        
        # Calculate commission
        commission = self.commission_structure.calculate(
            order.remaining_quantity, execution_price
        )
        
        # Check cash for buy orders
        if order.is_buy:
            total_cost = order.remaining_quantity * execution_price + commission
            if total_cost > self.cash:
                self.logger.warning(f"Insufficient cash to fill order {order.order_id}")
                return
        
        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.remaining_quantity,
            price=execution_price,
            commission=commission,
            timestamp=datetime.now()
        )
        
        # Update order
        order.filled_quantity += fill.quantity
        order.filled_price = execution_price  # Simplified - should be weighted average
        order.commission += commission
        order.status = OrderStatus.FILLED
        order.filled_at = fill.timestamp
        
        # Store fill
        self.fills.append(fill)
        
        # Update positions and cash
        self._process_fill(fill)
        
        # Create trade record
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol=order.symbol,
            side=order.side,
            total_quantity=fill.quantity,
            avg_price=execution_price,
            total_commission=commission,
            timestamp=fill.timestamp,
            fills=[fill]
        )
        
        self.trades.append(trade)
        self.total_trades += 1
        
        # Update statistics
        self.total_commission += commission
        
        # Callbacks
        self._notify_order_callbacks(order)
        self._notify_fill_callbacks(fill)
        self._notify_trade_callbacks(trade)
        
        self.logger.info(f"Order filled: {order.order_id} - {fill.quantity} @ ${execution_price:.2f}")
    
    def _process_fill(self, fill: Fill):
        """Process fill and update positions/cash"""
        symbol = fill.symbol
        
        if fill.side == OrderSide.BUY:
            # Buy order
            self.cash -= fill.net_amount
            
            if symbol in self.positions:
                # Update existing position
                pos = self.positions[symbol]
                total_cost = pos.quantity * pos.avg_price + fill.gross_amount
                total_quantity = pos.quantity + fill.quantity
                new_avg_price = total_cost / total_quantity
                
                pos.quantity = total_quantity
                pos.avg_price = new_avg_price
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=fill.quantity,
                    avg_price=fill.price,
                    current_price=fill.price,
                    market_value=fill.gross_amount,
                    unrealized_pnl=0.0,
                    sector=self._get_sector(symbol)
                )
        
        else:
            # Sell order
            self.cash += fill.net_amount
            
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.quantity -= fill.quantity
                
                # Remove position if quantity is zero
                if pos.quantity  str:
        """Get sector for symbol (simplified mapping)"""
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
            'JPM': 'Financials', 'BAC': 'Financials', 'V': 'Financials',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare'
        }
        return sector_map.get(symbol, 'Unknown')
    
    def _notify_order_callbacks(self, order: Order):
        """Notify order callbacks"""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Error in order callback: {e}")
    
    def _notify_fill_callbacks(self, fill: Fill):
        """Notify fill callbacks"""
        for callback in self.fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                self.logger.error(f"Error in fill callback: {e}")
    
    def _notify_trade_callbacks(self, trade: Trade):
        """Notify trade callbacks"""
        for callback in self.trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                self.logger.error(f"Error in trade callback: {e}")
    
    def _update_risk_manager(self):
        """Update risk manager with current portfolio state"""
        if self.risk_manager:
            total_value = self.get_total_portfolio_value()
            self.risk_manager.update_portfolio_state(
                positions=self.positions,
                cash=self.cash,
                total_value=total_value
            )
    
    def get_total_portfolio_value(self) -> float:
        """Get total portfolio value"""
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        total_value = self.get_total_portfolio_value()
        total_pnl = total_value - self.initial_capital
        total_pnl_pct = total_pnl / self.initial_capital
        
        return {
            'cash': self.cash,
            'position_value': sum(pos.market_value for pos in self.positions.values()),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'num_positions': len(self.positions),
            'total_trades': self.total_trades,
            'total_commission': self.total_commission
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if len(self.equity_curve)  0 else 0
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'avg_commission_per_trade': self.total_commission / max(self.total_trades, 1)
        }
    
    def reset(self):
        """Reset paper trading engine to initial state"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.fills.clear()
        self.trades.clear()
        self.equity_curve = [(datetime.now(), self.initial_capital)]
        self.daily_pnl.clear()
        
        # Reset statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        
        self.logger.info("Paper trading engine reset")


# Example usage and testing
if __name__ == "__main__":
    # Test paper trading engine
    engine = PaperTradingEngine(initial_capital=100000)
    
    # Add callbacks
    def on_order(order: Order):
        print(f"Order: {order.status.value} - {order.side.value} {order.quantity} {order.symbol}")
    
    def on_fill(fill: Fill):
        print(f"Fill: {fill.quantity} {fill.symbol} @ ${fill.price:.2f}")
    
    engine.add_order_callback(on_order)
    engine.add_fill_callback(on_fill)
    
    # Update market prices
    prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2800.0}
    engine.update_market_prices(prices)
    
    # Place some orders
    engine.place_order('AAPL', OrderSide.BUY, 100, OrderType.MARKET)
    engine.place_order('MSFT', OrderSide.BUY, 50, OrderType.MARKET)
    
    # Update prices and check portfolio
    prices = {'AAPL': 155.0, 'MSFT': 305.0, 'GOOGL': 2850.0}
    engine.update_market_prices(prices)
    
    # Print portfolio summary
    summary = engine.get_portfolio_summary()
    print("\nPortfolio Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Print performance metrics
    metrics = engine.get_performance_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
