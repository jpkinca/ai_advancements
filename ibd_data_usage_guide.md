# Using IBD Stock Data with Sweet Spot & Danger Zone System

Assuming your IBD data is in CSV format with columns: date, open, high, low, close, volume

## Step 1: Load Your Data

```python
import pandas as pd
from sweet_spot_danger_zone_system import create_dual_signal_system

# Example: Load data for one stock (replace with your file path)
stock_data = pd.read_csv('path/to/IBD_stock_AAPL.csv', parse_dates=['date'], index_col='date')
# Ensure column names match: open, high, low, close, volume

print(stock_data.head())
print(stock_data.columns)
```

## Step 2: Verify Data Format

Your DataFrame should look like:
```
                     open    high     low   close    volume
date
2020-01-01 00:00:00  100.0  105.0   95.0  102.0  1000000
2020-01-02 00:00:00  102.0  108.0   98.0  105.0  1200000
...
```

## Step 3: Run the System

```python
# Create and train the system
system = create_dual_signal_system()
metrics = system.train(stock_data)

print("Training Metrics:")
print(f"Sweet Spot Accuracy: {metrics['sweet_spot']['val_accuracy']:.3f}")
print(f"Danger Zone Accuracy: {metrics['danger_zone']['val_accuracy']:.3f}")

# Generate signals
signals = system.generate_signals(stock_data)

# View recent signals
print(signals.tail(10)[['combined_signal', 'position_size', 'confidence_score']])

# Run backtest
results = system.backtest(stock_data, initial_capital=10000)
print(f"Stock Performance: {results['total_return']:.2%} return, Sharpe {results['sharpe_ratio']:.2f}")
```

## For Multiple Stocks (50 IBD Stocks)

```python
import os

# Assume all CSV files are in a directory
data_dir = 'path/to/ibd_stocks/'
stock_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

results_summary = {}

for stock_file in stock_files[:5]:  # Test with first 5 stocks
    stock_name = stock_file.split('.')[0]
    stock_data = pd.read_csv(os.path.join(data_dir, stock_file), parse_dates=['date'], index_col='date')
    
    # Run system
    system = create_dual_signal_system()
    system.train(stock_data)
    signals = system.generate_signals(stock_data)
    results = system.backtest(stock_data)
    
    results_summary[stock_name] = {
        'total_return': results['total_return'],
        'sharpe_ratio': results['sharpe_ratio'],
        'win_rate': results['win_rate']
    }

# Print summary
for stock, metrics in results_summary.items():
    print(f"{stock}: Return {metrics['total_return']:.1%}, Sharpe {metrics['sharpe_ratio']:.2f}")
```

## Data Quality Checks

Before running, ensure:
- No missing values in OHLCV columns
- Reasonable price ranges (not pennies or millions)
- Sufficient data points (minimum 200-500 rows for training)
- Consistent date frequency (daily preferred)

## If Data Format Issues

If your CSV has different column names:
```python
# Rename columns
stock_data = stock_data.rename(columns={
    'Date': 'date',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})
```

## Expected Results

With good IBD stock data, you should see:
- Training accuracies: 50-70% (depending on market conditions)
- Backtest returns: Variable, but system should outperform buy-and-hold in trending markets
- Signals: Clear buy/avoid recommendations with position sizing

Run this on one stock first, then scale to all 50. If you share a sample data file or error, I can help debug! What's the format of your IBD data files?