"""
AI Trading Advancements - Base Classes

This module provides abstract base classes and interfaces for the AI trading system.
All classes follow the modular, reusable design principles and ASCII-only standards.

Author: AI Trading Development Team
Date: August 31, 2025
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from decimal import Decimal
from datetime import datetime, timezone
import logging
import asyncio
from pathlib import Path

from .data_structures import (
    MarketData, TradingSignal, BacktestResult, ModelMetrics,
    PortfolioPosition, SentimentData, TechnicalIndicators,
    SignalType, MarketCondition, TimeFrame
)

logger = logging.getLogger(__name__)


class BaseDataProvider(ABC):
    """
    Abstract base class for market data providers.
    Supports both historical and real-time data fetching.
    """
    
    def __init__(self, name: str):
        """Initialize data provider with name."""
        self.name = name
        self.logger = logger.getChild(f"DataProvider.{name}")
        self.is_connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data provider."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data provider."""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime
    ) -> List[MarketData]:
        """Fetch historical market data."""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current market price for symbol."""
        pass
    
    @abstractmethod
    async def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is available."""
        try:
            available_symbols = await self.get_symbols()
            return symbol.upper() in [s.upper() for s in available_symbols]
        except Exception as e:
            self.logger.error(f"[ERROR] Symbol validation failed: {e}")
            return False


class BaseIndicatorCalculator(ABC):
    """
    Abstract base class for technical indicator calculations.
    Provides standardized interface for all technical analysis.
    """
    
    def __init__(self, name: str):
        """Initialize indicator calculator."""
        self.name = name
        self.logger = logger.getChild(f"Indicator.{name}")
    
    @abstractmethod
    def calculate(self, data: List[MarketData]) -> TechnicalIndicators:
        """Calculate technical indicators from market data."""
        pass
    
    @abstractmethod
    def get_required_periods(self) -> int:
        """Get minimum number of data points required."""
        pass
    
    def validate_data(self, data: List[MarketData]) -> bool:
        """Validate input data for calculation."""
        if not data:
            self.logger.error("[ERROR] No data provided for indicator calculation")
            return False
        
        if len(data) < 1:
            self.logger.error(f"[ERROR] Mixed symbols in data: {symbols}")
            return False
        
        return True


class BaseAIModel(ABC):
    """
    Abstract base class for AI/ML models.
    Provides standardized interface for model training, prediction, and evaluation.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """Initialize AI model."""
        self.name = name
        self.version = version
        self.logger = logger.getChild(f"AIModel.{name}")
        self.is_trained = False
        self.model_path: Optional[Path] = None
        self.metrics: Optional[ModelMetrics] = None
    
    @abstractmethod
    async def train(
        self,
        training_data: List[MarketData],
        validation_data: Optional[List[MarketData]] = None,
        **kwargs
    ) -> ModelMetrics:
        """Train the model with provided data."""
        pass
    
    @abstractmethod
    async def predict(self, data: List[MarketData]) -> List[TradingSignal]:
        """Generate trading signals from market data."""
        pass
    
    @abstractmethod
    async def evaluate(self, test_data: List[MarketData]) -> ModelMetrics:
        """Evaluate model performance on test data."""
        pass
    
    @abstractmethod
    async def save_model(self, path: Path) -> bool:
        """Save trained model to disk."""
        pass
    
    @abstractmethod
    async def load_model(self, path: Path) -> bool:
        """Load trained model from disk."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metrics."""
        return {
            'name': self.name,
            'version': self.version,
            'is_trained': self.is_trained,
            'model_path': str(self.model_path) if self.model_path else None,
            'metrics': self.metrics.to_dict() if self.metrics else None
        }


class BaseSentimentAnalyzer(ABC):
    """
    Abstract base class for sentiment analysis.
    Supports multiple data sources and sentiment scoring methods.
    """
    
    def __init__(self, name: str):
        """Initialize sentiment analyzer."""
        self.name = name
        self.logger = logger.getChild(f"Sentiment.{name}")
    
    @abstractmethod
    async def analyze_sentiment(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.HOUR_1
    ) -> List[SentimentData]:
        """Analyze sentiment for given symbol."""
        pass
    
    @abstractmethod
    async def get_sentiment_score(self, text: str) -> Tuple[Decimal, Decimal]:
        """Get sentiment score and confidence for text."""
        pass
    
    def aggregate_sentiment(self, sentiment_data: List[SentimentData]) -> Decimal:
        """Aggregate multiple sentiment scores into single score."""
        if not sentiment_data:
            return Decimal('0.0')
        
        total_score = Decimal('0.0')
        total_weight = Decimal('0.0')
        
        for data in sentiment_data:
            weight = data.confidence * Decimal(str(data.mention_count))
            total_score += data.sentiment_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return Decimal('0.0')
        
        return total_score / total_weight


class BaseTradingStrategy(ABC):
    """
    Abstract base class for trading strategies.
    Integrates AI models, technical analysis, and risk management.
    """
    
    def __init__(self, name: str, risk_per_trade: Decimal = Decimal('0.01')):
        """Initialize trading strategy."""
        self.name = name
        self.risk_per_trade = risk_per_trade
        self.logger = logger.getChild(f"Strategy.{name}")
        self.models: List[BaseAIModel] = []
        self.indicators: List[BaseIndicatorCalculator] = []
        self.sentiment_analyzers: List[BaseSentimentAnalyzer] = []
    
    @abstractmethod
    async def generate_signals(
        self,
        market_data: List[MarketData],
        portfolio: List[PortfolioPosition]
    ) -> List[TradingSignal]:
        """Generate trading signals based on strategy logic."""
        pass
    
    @abstractmethod
    async def calculate_position_size(
        self,
        signal: TradingSignal,
        portfolio_value: Decimal,
        risk_per_trade: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate appropriate position size for signal."""
        pass
    
    def add_model(self, model: BaseAIModel) -> None:
        """Add AI model to strategy."""
        self.models.append(model)
        self.logger.info(f"[SUCCESS] Added model: {model.name}")
    
    def add_indicator(self, indicator: BaseIndicatorCalculator) -> None:
        """Add technical indicator to strategy."""
        self.indicators.append(indicator)
        self.logger.info(f"[SUCCESS] Added indicator: {indicator.name}")
    
    def add_sentiment_analyzer(self, analyzer: BaseSentimentAnalyzer) -> None:
        """Add sentiment analyzer to strategy."""
        self.sentiment_analyzers.append(analyzer)
        self.logger.info(f"[SUCCESS] Added sentiment analyzer: {analyzer.name}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy configuration information."""
        return {
            'name': self.name,
            'risk_per_trade': str(self.risk_per_trade),
            'models': [model.name for model in self.models],
            'indicators': [indicator.name for indicator in self.indicators],
            'sentiment_analyzers': [analyzer.name for analyzer in self.sentiment_analyzers]
        }


class BaseBacktester(ABC):
    """
    Abstract base class for backtesting trading strategies.
    Provides standardized backtesting framework with performance metrics.
    """
    
    def __init__(self, initial_capital: Decimal = Decimal('100000.00')):
        """Initialize backtester."""
        self.initial_capital = initial_capital
        self.logger = logger.getChild("Backtester")
    
    @abstractmethod
    async def run_backtest(
        self,
        strategy: BaseTradingStrategy,
        data: List[MarketData],
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Run backtest for strategy on historical data."""
        pass
    
    @abstractmethod
    def calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        portfolio_values: List[Decimal]
    ) -> BacktestResult:
        """Calculate performance metrics from backtest results."""
        pass
    
    def validate_backtest_data(
        self,
        data: List[MarketData],
        start_date: datetime,
        end_date: datetime
    ) -> bool:
        """Validate backtest data completeness and consistency."""
        if not data:
            self.logger.error("[ERROR] No data provided for backtesting")
            return False
        
        # Check date range
        data_dates = [d.timestamp for d in data]
        if min(data_dates) > start_date or max(data_dates)  86400 * 7:  # More than 7 days gap
                self.logger.warning(f"[WARNING] Large gap in data: {gap/86400:.1f} days")
        
        return True


class BaseRiskManager(ABC):
    """
    Abstract base class for risk management.
    Provides position sizing, stop-loss, and portfolio risk controls.
    """
    
    def __init__(self, max_risk_per_trade: Decimal = Decimal('0.02')):
        """Initialize risk manager."""
        self.max_risk_per_trade = max_risk_per_trade
        self.logger = logger.getChild("RiskManager")
    
    @abstractmethod
    async def validate_trade(
        self,
        signal: TradingSignal,
        portfolio: List[PortfolioPosition],
        portfolio_value: Decimal
    ) -> bool:
        """Validate if trade meets risk management criteria."""
        pass
    
    @abstractmethod
    async def calculate_stop_loss(
        self,
        signal: TradingSignal,
        market_data: List[MarketData]
    ) -> Decimal:
        """Calculate appropriate stop-loss level."""
        pass
    
    @abstractmethod
    async def calculate_position_size(
        self,
        signal: TradingSignal,
        portfolio_value: Decimal,
        risk_amount: Decimal
    ) -> Decimal:
        """Calculate position size based on risk parameters."""
        pass
    
    def calculate_portfolio_risk(self, portfolio: List[PortfolioPosition]) -> Decimal:
        """Calculate total portfolio risk exposure."""
        total_risk = Decimal('0.0')
        for position in portfolio:
            position_risk = abs(position.unrealized_pnl) / position.market_value
            total_risk += position_risk
        return total_risk


# Export base classes
__all__ = [
    'BaseDataProvider', 'BaseIndicatorCalculator', 'BaseAIModel',
    'BaseSentimentAnalyzer', 'BaseTradingStrategy', 'BaseBacktester',
    'BaseRiskManager'
]


if __name__ == "__main__":
    # Example usage and interface demonstration
    print("\n=== AI Trading Base Classes Interface Test ===")
    
    # Demonstrate interface design
    class MockDataProvider(BaseDataProvider):
        async def connect(self): return True
        async def disconnect(self): pass
        async def get_historical_data(self, symbol, timeframe, start_date, end_date): return []
        async def get_current_price(self, symbol): return Decimal('100.00')
        async def get_symbols(self): return ['AAPL', 'MSFT', 'GOOGL']
    
    # Test interface
    provider = MockDataProvider("MockProvider")
    print(f"[SUCCESS] Data provider created: {provider.name}")
    print(f"[DATA] Connected: {provider.is_connected}")
    
    # Test async functionality
    async def test_provider():
        connected = await provider.connect()
        price = await provider.get_current_price("AAPL")
        symbols = await provider.get_symbols()
        
        print(f"[SUCCESS] Connection test: {connected}")
        print(f"[DATA] Mock price: ${price}")
        print(f"[DATA] Available symbols: {len(symbols)}")
    
    # Run async test
    asyncio.run(test_provider())
    
    print("\n[SUCCESS] Base classes interface test completed")
