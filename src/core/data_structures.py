"""
AI Trading Advancements - Core Data Structures

This module defines the core data structures used throughout the AI trading system.
All classes use Decimal for financial calculations and Eastern timezone (EST/EDT)
for all timestamps, following NYSE/NASDAQ timezone standards.

Author: AI Trading Development Team
Date: August 31, 2025
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum
import json

# Import timezone utilities
from .timezone_utils import now_eastern, to_eastern, EASTERN_TZ


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class MarketCondition(Enum):
    """Market condition types."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


class TimeFrame(Enum):
    """Time frame options for analysis."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass
class MarketData:
    """
    Market data structure for OHLCV data.
    
    All timestamps are automatically converted to Eastern timezone (EST/EDT)
    to match NYSE/NASDAQ trading hours and market data standards.
    """
    symbol: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: int
    timeframe: TimeFrame
    
    def __post_init__(self):
        """Ensure timestamp is in Eastern timezone."""
        # Convert to Eastern timezone for all market data
        self.timestamp = to_eastern(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open_price': str(self.open_price),
            'high_price': str(self.high_price),
            'low_price': str(self.low_price),
            'close_price': str(self.close_price),
            'volume': self.volume,
            'timeframe': self.timeframe.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create instance from dictionary."""
        return cls(
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            open_price=Decimal(data['open_price']),
            high_price=Decimal(data['high_price']),
            low_price=Decimal(data['low_price']),
            close_price=Decimal(data['close_price']),
            volume=data['volume'],
            timeframe=TimeFrame(data['timeframe'])
        )


@dataclass
class TechnicalIndicators:
    """Technical indicators for a symbol."""
    symbol: str
    timestamp: datetime
    sma_20: Optional[Decimal] = None
    sma_50: Optional[Decimal] = None
    sma_200: Optional[Decimal] = None
    ema_12: Optional[Decimal] = None
    ema_26: Optional[Decimal] = None
    rsi: Optional[Decimal] = None
    macd: Optional[Decimal] = None
    macd_signal: Optional[Decimal] = None
    macd_histogram: Optional[Decimal] = None
    bollinger_upper: Optional[Decimal] = None
    bollinger_middle: Optional[Decimal] = None
    bollinger_lower: Optional[Decimal] = None
    atr: Optional[Decimal] = None
    volume_sma: Optional[Decimal] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with string representations of Decimals."""
        result = {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat()
        }
        
        for key, value in self.__dict__.items():
            if key not in ['symbol', 'timestamp'] and value is not None:
                result[key] = str(value) if isinstance(value, Decimal) else value
        
        return result


@dataclass
class SentimentData:
    """Sentiment analysis data structure."""
    symbol: str
    timestamp: datetime
    source: str  # 'twitter', 'reddit', 'news', etc.
    sentiment_score: Decimal  # -1.0 to 1.0
    confidence: Decimal  # 0.0 to 1.0
    mention_count: int
    raw_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'sentiment_score': str(self.sentiment_score),
            'confidence': str(self.confidence),
            'mention_count': self.mention_count,
            'raw_text': self.raw_text
        }


@dataclass
class TradingSignal:
    """
    AI-generated trading signal.
    
    All timestamps are automatically converted to Eastern timezone (EST/EDT)
    to match NYSE/NASDAQ trading hours and market data standards.
    """
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    confidence: Decimal  # 0.0 to 1.0
    entry_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    position_size: Optional[Decimal] = None
    reasoning: Optional[str] = None
    model_version: Optional[str] = None
    
    # Additional metadata
    technical_score: Optional[Decimal] = None
    sentiment_score: Optional[Decimal] = None
    market_condition: Optional[MarketCondition] = None
    risk_score: Optional[Decimal] = None
    
    def __post_init__(self):
        """Ensure timestamp is in Eastern timezone."""
        # Convert to Eastern timezone for all trading signals
        self.timestamp = to_eastern(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'signal_type': self.signal_type.value,
            'confidence': str(self.confidence)
        }
        
        # Add optional fields if they exist
        for key, value in self.__dict__.items():
            if key not in ['symbol', 'timestamp', 'signal_type', 'confidence'] and value is not None:
                if isinstance(value, Decimal):
                    result[key] = str(value)
                elif isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value
        
        return result


@dataclass
class PortfolioPosition:
    """
    Portfolio position data structure.
    
    All timestamps are automatically converted to Eastern timezone (EST/EDT)
    to match NYSE/NASDAQ trading hours and market data standards.
    """
    symbol: str
    quantity: Decimal
    average_cost: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal('0.0')
    timestamp: datetime = field(default_factory=now_eastern)
    
    def __post_init__(self):
        """Ensure timestamp is in Eastern timezone."""
        # Convert to Eastern timezone for all portfolio positions
        self.timestamp = to_eastern(self.timestamp)
    
    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost basis."""
        return self.quantity * self.average_cost
    
    @property
    def pnl_percentage(self) -> Decimal:
        """Calculate PnL percentage."""
        if self.total_cost == 0:
            return Decimal('0.0')
        return (self.unrealized_pnl / self.total_cost) * Decimal('100.0')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'symbol': self.symbol,
            'quantity': str(self.quantity),
            'average_cost': str(self.average_cost),
            'current_price': str(self.current_price),
            'market_value': str(self.market_value),
            'unrealized_pnl': str(self.unrealized_pnl),
            'realized_pnl': str(self.realized_pnl),
            'pnl_percentage': str(self.pnl_percentage),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class BacktestResult:
    """Backtesting result data structure."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    total_return: Decimal
    annual_return: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    profit_factor: Decimal
    
    @property
    def net_profit(self) -> Decimal:
        """Calculate net profit."""
        return self.final_capital - self.initial_capital
    
    @property
    def return_percentage(self) -> Decimal:
        """Calculate return percentage."""
        if self.initial_capital == 0:
            return Decimal('0.0')
        return (self.net_profit / self.initial_capital) * Decimal('100.0')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'strategy_name': self.strategy_name,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': str(self.initial_capital),
            'final_capital': str(self.final_capital),
            'net_profit': str(self.net_profit),
            'total_return': str(self.total_return),
            'return_percentage': str(self.return_percentage),
            'annual_return': str(self.annual_return),
            'sharpe_ratio': str(self.sharpe_ratio),
            'max_drawdown': str(self.max_drawdown),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': str(self.win_rate),
            'profit_factor': str(self.profit_factor)
        }


@dataclass
class ModelMetrics:
    """AI model performance metrics."""
    model_name: str
    model_version: str
    training_date: datetime
    accuracy: Decimal
    precision: Decimal
    recall: Decimal
    f1_score: Decimal
    validation_loss: Decimal
    training_samples: int
    validation_samples: int
    test_samples: int
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'training_date': self.training_date.isoformat(),
            'accuracy': str(self.accuracy),
            'precision': str(self.precision),
            'recall': str(self.recall),
            'f1_score': str(self.f1_score),
            'validation_loss': str(self.validation_loss),
            'training_samples': self.training_samples,
            'validation_samples': self.validation_samples,
            'test_samples': self.test_samples,
            'hyperparameters': self.hyperparameters
        }


class DataValidator:
    """Utility class for data validation."""
    
    @staticmethod
    def validate_market_data(data: MarketData) -> bool:
        """Validate market data structure."""
        if data.open_price < min(data.open_price, data.close_price):
            return False
            
        if data.volume < 0:
            return False
            
        return True
    
    @staticmethod
    def validate_signal(signal: TradingSignal) -> bool:
        """Validate trading signal structure."""
        if not (Decimal('0.0') <= signal.confidence <= Decimal('1.0')):
            return False
        
        if signal.entry_price is not None and signal.entry_price <= 0:
            return False
            
        if signal.stop_loss is not None and signal.stop_loss <= 0:
            return False
            
        if signal.take_profit is not None and signal.take_profit <= 0:
            return False
            
        return True


# Export commonly used types
__all__ = [
    'SignalType', 'MarketCondition', 'TimeFrame',
    'MarketData', 'TechnicalIndicators', 'SentimentData',
    'TradingSignal', 'PortfolioPosition', 'BacktestResult',
    'ModelMetrics', 'DataValidator'
]


if __name__ == "__main__":
    # Example usage and testing
    import time
    
    print("\n=== AI Trading Core Data Structures Test ===")
    
    # Test MarketData
    market_data = MarketData(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        open_price=Decimal('150.00'),
        high_price=Decimal('152.50'),
        low_price=Decimal('149.75'),
        close_price=Decimal('151.25'),
        volume=1000000,
        timeframe=TimeFrame.DAY_1
    )
    
    print(f"[SUCCESS] MarketData created for {market_data.symbol}")
    print(f"[DATA] Price: ${market_data.close_price}")
    print(f"[DATA] Valid: {DataValidator.validate_market_data(market_data)}")
    
    # Test TradingSignal
    signal = TradingSignal(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        signal_type=SignalType.BUY,
        confidence=Decimal('0.85'),
        entry_price=Decimal('151.00'),
        stop_loss=Decimal('148.00'),
        take_profit=Decimal('156.00'),
        reasoning="Strong technical breakout with positive sentiment"
    )
    
    print(f"[SUCCESS] TradingSignal created: {signal.signal_type.value}")
    print(f"[DATA] Confidence: {signal.confidence}")
    print(f"[DATA] Valid: {DataValidator.validate_signal(signal)}")
    
    # Test JSON serialization
    signal_dict = signal.to_dict()
    print(f"[SUCCESS] JSON serialization: {len(signal_dict)} fields")
    
    print("\n[SUCCESS] Core data structures test completed")
