# Using the Sweet Spot & Danger Zone System

## Quick Start with Real Data

```python
import pandas as pd
from sweet_spot_danger_zone_system import create_dual_signal_system

# 1. Load your market data (OHLCV format)
# Replace with your data source (CSV, database, API)
df = pd.read_csv('your_market_data.csv', index_col='timestamp', parse_dates=True)
# Ensure columns: open, high, low, close, volume

# 2. Create and train the system
system = create_dual_signal_system()
metrics = system.train(df)

print("Training completed:")
print(f"Sweet Spot Accuracy: {metrics['sweet_spot']['val_accuracy']:.3f}")
print(f"Danger Zone Accuracy: {metrics['danger_zone']['val_accuracy']:.3f}")

# 3. Generate trading signals
signals = system.generate_signals(df)

# 4. View recent signals
recent_signals = signals.tail(10)
print(recent_signals[['combined_signal', 'position_size', 'confidence_score']])

# 5. Run backtest
backtest_results = system.backtest(df, initial_capital=100000)
print(f"Total Return: {backtest_results['total_return']:.2%}")
print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")

# 6. Save models for later use
system.save_models('my_trained_models')
```

## Signal Interpretation

- `combined_signal`: 0=Avoid, 1=Weak Buy, 2=Strong Buy
- `position_size`: Suggested allocation (0-1 scale)
- `confidence_score`: Overall confidence in the signal

## Customization Options

### Change Model Types
```python
from sweet_spot_danger_zone_system import SweetSpotConfig, DangerZoneConfig, DualSignalTradingSystem

# Custom configurations
sweet_config = SweetSpotConfig(
    model_type='random_forest',  # or 'xgboost', 'gradient_boosting'
    n_estimators=300,
    min_sweetness_threshold=0.7
)

danger_config = DangerZoneConfig(
    model_type='xgboost',
    max_danger_threshold=0.3
)

system = DualSignalTradingSystem(sweet_config, danger_config)
```

### Individual Components
```python
from sweet_spot_danger_zone_system import create_sweet_spot_detector, create_danger_zone_detector

# Use only sweet spot detection
sweet_detector = create_sweet_spot_detector('xgboost')
sweet_detector.train(df)
predictions, probs = sweet_detector.predict(df)
```

## Data Requirements

Your DataFrame must have:
- DatetimeIndex
- Columns: `open`, `high`, `low`, `close`, `volume` (optional but recommended)

## Performance Notes

- Training time: 1-5 minutes depending on data size
- Memory usage: ~500MB for 4 years of daily data
- Prediction speed: ~1000 rows/second

## Troubleshooting

- **Low accuracy**: Try different model types or adjust feature parameters
- **Memory errors**: Reduce `n_estimators` or use smaller datasets
- **No signals**: Check data quality and feature engineering

## Advanced Usage

- Load saved models: `system.load_models('my_models')`
- Feature importance: `system.get_feature_importance()`
- Custom features: Extend `create_sweet_features()` or `create_danger_features()`