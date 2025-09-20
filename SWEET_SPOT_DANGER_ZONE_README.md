# Sweet Spot & Danger Zone Detection System

A comprehensive AI-powered algorithmic trading system that combines opportunity identification with risk management through dual-signal generation.

## Overview

This system implements advanced machine learning techniques for real-time trading signal generation, featuring:

- **Sweet Spot Detection**: ML-based opportunity identification using Random Forest and technical analysis
- **Danger Zone Detection**: Risk-focused capital preservation using XGBoost and statistical measures
- **Dual Signal System**: Integrated decision making combining both opportunity and risk signals
- **Real-Time Processing**: Streaming signal generation with low latency
- **Advanced Backtesting**: Walk-forward validation and comprehensive performance metrics
- **Unified Architecture**: Complete integration with TensorTrade ecosystem

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Market Data    │───▶│ Signal Generation│───▶│ Portfolio Mgmt  │
│   (Real-time)   │    │  (ML Models)     │    │  (Risk Control) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Backtesting   │    │   Real-time      │    │   Live Trading  │
│   Framework     │    │   Streaming      │    │   Execution     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Core Components

### 1. Dual Signal Trading System (`sweet_spot_danger_zone_system.py`)

**Features:**
- SweetSpotDetector: Opportunity identification using Random Forest
- DangerZoneDetector: Risk assessment using XGBoost
- Feature engineering with 40+ technical indicators
- Asymmetric risk management
- Confidence scoring and position sizing

**Key Classes:**
- `SweetSpotDetector`: ML model for opportunity detection
- `DangerZoneDetector`: ML model for risk assessment
- `DualSignalTradingSystem`: Integrated signal generation

### 2. Advanced Backtesting Framework (`backtesting_framework.py`)

**Features:**
- Walk-forward validation
- Comprehensive performance metrics
- Risk-adjusted return analysis
- Monte Carlo simulation
- Visualization and reporting

**Key Classes:**
- `AdvancedBacktester`: Main backtesting engine
- `WalkForwardValidator`: Robust validation framework
- `PerformanceMetrics`: Detailed performance analysis

### 3. Real-Time Signal Generation (`real_time_signals.py`)

**Features:**
- Streaming signal processing
- Thread-safe signal buffering
- Configurable alert system
- Low-latency updates
- Performance monitoring

**Key Classes:**
- `RealTimeSignalGenerator`: Streaming signal engine
- `SignalBuffer`: Thread-safe signal storage
- `AlertSystem`: Real-time notification system

### 4. Unified Trading System (`unified_trading_system.py`)

**Features:**
- Complete system integration
- Portfolio management
- Risk management
- Multi-feed data integration
- Live trading execution

**Key Classes:**
- `UnifiedTradingSystem`: Complete trading platform
- `PortfolioManager`: Position and capital management
- `DataFeedManager`: Multi-source data integration

## Quick Start

### Installation

```bash
# Install required packages
pip install numpy pandas scikit-learn xgboost ta-lib matplotlib seaborn scipy

# For TA-Lib on Windows (if needed)
# Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.24-cp311-cp311-win_amd64.whl
```

### Basic Usage

```python
from sweet_spot_danger_zone_system import DualSignalTradingSystem
from backtesting_framework import AdvancedBacktester

# Create and train the system
system = DualSignalTradingSystem()
system.train_models(X_train, y_train_sweet, y_train_danger)

# Generate signals
signals = system.generate_signals(market_data)

# Run backtest
backtester = AdvancedBacktester()
results = backtester.run_backtest(system, historical_data, initial_capital=100000)
```

### Real-Time Trading

```python
from real_time_signals import RealTimeSignalGenerator, create_real_time_generator
from unified_trading_system import create_unified_system

# Create real-time system
system = create_unified_system(100000.0)

# Add market data
for symbol, data in historical_data.items():
    system.real_time_generator.add_market_data(symbol, data)

# Start trading
system.start_trading()

# Monitor performance
status = system.get_system_status()
print(f"Portfolio Value: ${status['portfolio']['current_capital']:.2f}")
```

## Configuration

### Signal Generation Parameters

```python
from sweet_spot_danger_zone_system import SweetSpotConfig, DangerZoneConfig

# Configure Sweet Spot detector
sweet_config = SweetSpotConfig(
    n_estimators=200,
    max_depth=10,
    min_samples_split=50,
    feature_importance_threshold=0.01
)

# Configure Danger Zone detector
danger_config = DangerZoneConfig(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.1,
    risk_threshold=0.7
)

system = DualSignalTradingSystem(sweet_config, danger_config)
```

### Real-Time Streaming Configuration

```python
from real_time_signals import StreamingConfig

config = StreamingConfig(
    update_interval=1.0,  # Update every second
    lookback_window=100,  # 100 periods for feature calculation
    enable_alerts=True,
    alert_thresholds={
        'strong_buy': 0.8,
        'weak_buy': 0.6,
        'danger': 0.7,
        'confidence': 0.75
    }
)
```

### Backtesting Configuration

```python
from backtesting_framework import BacktestConfig

config = BacktestConfig(
    initial_capital=100000,
    commission=0.001,  # 0.1% per trade
    slippage=0.0005,   # 0.05% slippage
    enable_walk_forward=True,
    train_window=252,  # 1 year training
    test_window=21     # 1 month testing
)
```

## Feature Engineering

The system uses 40+ technical indicators organized in categories:

### Trend Indicators
- Moving Averages (SMA, EMA, WMA)
- MACD (MACD, Signal, Histogram)
- ADX (Directional Movement)
- Ichimoku Cloud components

### Momentum Indicators
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Williams %R
- ROC (Rate of Change)

### Volatility Indicators
- Bollinger Bands
- ATR (Average True Range)
- Standard Deviation
- Historical Volatility

### Volume Indicators
- Volume Moving Averages
- OBV (On Balance Volume)
- Volume Rate of Change
- Accumulation/Distribution

### Statistical Measures
- Z-Score
- Percentile Ranks
- Rolling Statistics
- Correlation Measures

## Signal Interpretation

### Combined Signal Values
- **2**: Strong Buy (High opportunity, Low risk)
- **1**: Weak Buy (Moderate opportunity, Acceptable risk)
- **0**: Avoid (Low opportunity or High risk)

### Confidence Scoring
- **0.0-0.3**: Low confidence - Consider avoiding
- **0.3-0.6**: Medium confidence - Monitor closely
- **0.6-0.8**: High confidence - Good trading opportunity
- **0.8-1.0**: Very high confidence - Strong signal

### Risk Assessment
- **Sweet Probability**: Likelihood of profitable opportunity
- **Danger Probability**: Likelihood of adverse market conditions
- **Position Size**: Recommended position size (0.0-1.0 scale)

## Performance Metrics

### Risk-Adjusted Returns
- Sharpe Ratio: Risk-adjusted return measure
- Sortino Ratio: Downside risk-adjusted return
- Calmar Ratio: Maximum drawdown-adjusted return

### Risk Metrics
- Maximum Drawdown: Largest peak-to-trough decline
- Value at Risk (VaR): Potential loss at confidence level
- Expected Shortfall: Average loss beyond VaR

### Trade Statistics
- Win Rate: Percentage of profitable trades
- Profit Factor: Gross profit / Gross loss
- Average Win/Loss: Mean profitable/unprofitable trade
- Trade Frequency: Number of trades per period

## Backtesting Methodology

### Walk-Forward Validation
1. **Training Window**: Initial period for model training
2. **Testing Window**: Out-of-sample performance evaluation
3. **Rolling Window**: Move forward and retrain periodically
4. **Performance Tracking**: Monitor stability and adaptability

### Monte Carlo Simulation
- Generate multiple random scenarios
- Assess strategy robustness
- Calculate probability distributions
- Stress test under various conditions

## Real-Time Operation

### Signal Generation Flow
1. **Data Ingestion**: Receive real-time market data
2. **Feature Calculation**: Compute technical indicators
3. **Model Inference**: Generate Sweet Spot and Danger Zone predictions
4. **Signal Combination**: Create integrated trading decision
5. **Risk Management**: Apply position sizing and limits
6. **Execution**: Send orders to broker (paper/live trading)

### Alert System
- **Strong Buy Alerts**: High-confidence opportunity signals
- **Danger Alerts**: Risk threshold breaches
- **Performance Alerts**: System health monitoring
- **Custom Alerts**: User-defined conditions

## Integration with TensorTrade

### Data Flow
```
Market Data → Feature Engineering → ML Models → Signals → TensorTrade Strategy → Execution
```

### Strategy Implementation
```python
from tensortrade.strategies import TradingStrategy
from unified_trading_system import UnifiedTradingSystem

class DualSignalStrategy(TradingStrategy):
    def __init__(self, unified_system):
        self.system = unified_system

    def should_buy(self, symbol):
        signals = self.system.real_time_generator.get_signals_for_symbol(symbol, 1)
        if signals:
            return signals[0]['combined_signal'] >= 1
        return False

    def should_sell(self, symbol):
        signals = self.system.real_time_generator.get_signals_for_symbol(symbol, 1)
        if signals:
            return signals[0]['combined_signal'] == 0
        return False
```

## Example Results

### Backtest Performance (Sample)
```
Total Return: +24.7%
Annual Return: +18.3%
Sharpe Ratio: 1.45
Max Drawdown: -8.2%
Win Rate: 62.4%
Profit Factor: 1.73
Total Trades: 247
```

### Signal Accuracy (Sample)
```
Sweet Spot Detection:
- Precision: 71.2%
- Recall: 68.9%
- F1-Score: 0.70

Danger Zone Detection:
- Precision: 78.3%
- Recall: 72.1%
- F1-Score: 0.75

Combined System:
- Signal Confidence: 0.76 (avg)
- Risk-Adjusted Return: 1.42 (Sharpe)
```

## Best Practices

### Model Training
1. Use sufficient historical data (2+ years)
2. Include various market conditions
3. Validate on out-of-sample data
4. Regular model retraining

### Risk Management
1. Start with small position sizes
2. Implement stop-loss orders
3. Monitor maximum drawdown
4. Diversify across assets

### System Monitoring
1. Track signal quality metrics
2. Monitor model performance drift
3. Log all trades and decisions
4. Regular system health checks

## Troubleshooting

### Common Issues

**Low Signal Confidence:**
- Check feature calculation accuracy
- Verify model training data quality
- Review market condition coverage

**Poor Backtest Results:**
- Adjust feature selection
- Modify model hyperparameters
- Review transaction costs assumptions

**Real-Time Performance:**
- Optimize feature calculation
- Reduce update frequency if needed
- Check data feed latency

### Performance Optimization
- Use vectorized operations
- Implement parallel processing
- Cache frequently used data
- Optimize memory usage

## Future Enhancements

### Planned Features
- Multi-timeframe analysis
- Alternative data integration
- Deep learning models
- Portfolio optimization
- Advanced order types

### Research Areas
- Quantum-enhanced ML models
- Blockchain-based execution
- Sentiment analysis integration
- Cross-market arbitrage

## Support and Documentation

For detailed API documentation, see individual module docstrings and example scripts.

### Key Files
- `sweet_spot_danger_zone_system.py`: Core ML system
- `backtesting_framework.py`: Backtesting and validation
- `real_time_signals.py`: Streaming signal generation
- `unified_trading_system.py`: Complete platform integration

### Example Scripts
- `example_usage()`: Basic system demonstration
- `example_real_time_system()`: Real-time operation example
- `example_unified_system()`: Complete system integration

---

**Disclaimer**: This system is for educational and research purposes. Always test thoroughly before live trading. Past performance does not guarantee future results.