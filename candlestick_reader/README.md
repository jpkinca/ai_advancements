# Candlestick AI Analysis System

A comprehensive AI-powered candlestick pattern recognition and analysis system that integrates with volume-price action (VPA) and multimodal fusion for advanced trading signals.

## Features

- **AI-Powered Pattern Recognition**: Uses YOLOv8 for object detection and Vision Transformers for advanced pattern classification
- **50+ Candlestick Patterns**: Comprehensive pattern library including bullish, bearish, and continuation patterns
- **VPA Integration**: Combines candlestick analysis with volume-price action signals
- **Multimodal Fusion**: Integrates chart images, technical indicators, and pattern analysis
- **Real-time Processing**: Celery-based task system for real-time signal generation
- **FAISS Similarity Matching**: Pattern similarity search for historical analysis
- **Comprehensive Testing**: Full test suite with performance benchmarks

## Architecture

```
candlestick_reader/
├── candlestick_analyzer.py      # Main AI analysis engine
├── integration.py               # System integration with VPA/multimodal
├── model_training.py            # Training pipeline for YOLO/ViT models
├── tasks.py                     # Celery tasks for real-time processing
├── test_candlestick_system.py   # Comprehensive test suite
└── __init__.py
```

## Installation

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install ultralytics transformers mplfinance pandas numpy
pip install celery redis faiss-cpu pillow
```

### Model Setup

1. **Download Pre-trained Models** (optional):
   ```python
   from candlestick_reader.model_training import CandlestickModelTrainer
   trainer = CandlestickModelTrainer()
   trainer.download_pretrained_models()
   ```

2. **Train Custom Models**:
   ```python
   trainer.train_yolo_model()
   trainer.train_vit_model()
   ```

## Quick Start

### Basic Usage

```python
from candlestick_reader.candlestick_analyzer import CandlestickAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = CandlestickAnalyzer()

# Load your OHLCV data
df = pd.read_csv('your_market_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Analyze candlestick patterns
result = analyzer.analyze_candlestick_signal(df, 'AAPL', chart_path=None)

print(f"Signal: {result['signal']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Patterns: {result['patterns']}")
```

### Integrated System Usage

```python
from candlestick_reader.integration import IntegratedCandlestickSystem
import asyncio

async def main():
    # Initialize integrated system
    system = IntegratedCandlestickSystem()

    # Comprehensive analysis
    result = await system.analyze_symbol_comprehensive('AAPL', '5min', 50)

    print(f"Combined Signal: {result['signal']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Trade Action: {result['trade_recommendation']['action']}")

asyncio.run(main())
```

### Real-time Processing with Celery

```python
from candlestick_reader.tasks import generate_signal_task

# Queue a signal generation task
result = generate_signal_task.delay('AAPL', '5min')
print(result.get())  # Wait for result
```

## Pattern Library

The system recognizes 50+ candlestick patterns including:

### Bullish Patterns
- Hammer, Bullish Engulfing, Morning Star
- Bullish Harami, Piercing Pattern, Three White Soldiers
- And many more...

### Bearish Patterns
- Shooting Star, Bearish Engulfing, Evening Star
- Bearish Harami, Dark Cloud Cover, Three Black Crows
- And many more...

### Continuation Patterns
- Doji, Spinning Top, High Wave
- And others...

## API Reference

### CandlestickAnalyzer

#### `analyze_candlestick_signal(df, symbol, chart_path=None)`
Analyzes candlestick patterns and generates trading signals.

**Parameters:**
- `df`: OHLCV DataFrame
- `symbol`: Stock symbol
- `chart_path`: Path to chart image (optional)

**Returns:** Dictionary with signal, confidence, patterns, and analysis

#### `detect_candlestick_patterns(df)`
Detects candlestick patterns in the data.

**Parameters:**
- `df`: OHLCV DataFrame

**Returns:** Dictionary with detected patterns and confidence scores

#### `generate_candlestick_chart(df, symbol, timeframe)`
Generates candlestick chart image.

**Parameters:**
- `df`: OHLCV DataFrame
- `symbol`: Stock symbol
- `timeframe`: Data timeframe

**Returns:** Path to generated chart image

### IntegratedCandlestickSystem

#### `analyze_symbol_comprehensive(symbol, timeframe, lookback_periods)`
Performs comprehensive analysis combining candlesticks, VPA, and multimodal fusion.

**Parameters:**
- `symbol`: Stock symbol
- `timeframe`: Data timeframe
- `lookback_periods`: Number of periods to analyze

**Returns:** Complete analysis results

#### `generate_trading_signal(symbol, timeframe)`
Generates complete trading signal with position sizing and risk management.

**Parameters:**
- `symbol`: Stock symbol
- `timeframe`: Data timeframe

**Returns:** Trading signal with action, position size, stop loss, take profit

## Celery Tasks

### Starting Workers

```bash
# Start Celery worker
celery -A candlestick_reader.tasks worker --loglevel=info --queues=candlestick

# Start Celery beat for periodic tasks
celery -A candlestick_reader.tasks beat --loglevel=info
```

### Available Tasks

- `candlestick.analyze_symbol`: Analyze single symbol
- `candlestick.generate_signal`: Generate trading signal
- `candlestick.batch_analyze`: Analyze multiple symbols
- `candlestick.real_time_monitor`: Monitor symbols in real-time
- `candlestick.update_models`: Update AI models

### Example Task Usage

```python
from candlestick_reader.tasks import analyze_symbol_task, generate_signal_task

# Single symbol analysis
result = analyze_symbol_task.delay('AAPL', '5min', 50)
analysis = result.get()

# Generate trading signal
signal = generate_signal_task.delay('AAPL', '5min')
trade_signal = signal.get()

# Batch analysis
symbols = ['AAPL', 'MSFT', 'GOOGL']
batch_result = batch_analyze_task.delay(symbols, '5min', 50)
results = batch_result.get()
```

## Model Training

### Data Generation

```python
from candlestick_reader.model_training import CandlestickDataGenerator

generator = CandlestickDataGenerator()
patterns = ['hammer', 'shooting_star', 'doji', 'engulfing']

# Generate training data
training_data = generator.generate_pattern_data(patterns, samples_per_pattern=1000)

# Generate training charts
generator.generate_training_dataset(patterns, samples_per_pattern=1000)
```

### Model Training

```python
from candlestick_reader.model_training import CandlestickModelTrainer

trainer = CandlestickModelTrainer()

# Train YOLO model
yolo_results = trainer.train_yolo_model(
    data_yaml='data/candlestick_data.yaml',
    epochs=100,
    batch_size=16
)

# Train Vision Transformer
vit_results = trainer.train_vit_model(
    train_dir='data/train',
    val_dir='data/val',
    epochs=50,
    batch_size=8
)
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest candlestick_reader/test_candlestick_system.py -v

# Run specific test class
python -m pytest candlestick_reader/test_candlestick_system.py::TestCandlestickAnalyzer -v

# Run performance benchmarks
python candlestick_reader/test_candlestick_system.py
```

### System Validation

```python
from candlestick_reader.test_candlestick_system import run_system_validation

# Validate system components
validation_results = run_system_validation()
print(f"System Status: {validation_results['overall_status']}")
```

## Configuration

### Celery Configuration

Edit `celeryconfig.py` to configure:

- Redis broker URL
- Worker concurrency
- Task queues and routing
- Periodic task schedules

### Model Configuration

Models can be configured in the respective classes:

- `CandlestickAnalyzer`: Pattern thresholds, model paths
- `IntegratedCandlestickSystem`: Signal weights, confidence thresholds
- `CandlestickModelTrainer`: Training hyperparameters

## Performance

### Benchmarks

The system has been benchmarked with the following performance:

- **Pattern Detection**: ~0.1-0.5 seconds for 1000 data points
- **Chart Generation**: ~1-3 seconds for 1000 data points
- **Signal Generation**: ~2-5 seconds per symbol (including all components)
- **Batch Processing**: ~10-30 seconds for 10 symbols

### Optimization Tips

1. **Use GPU**: Enable CUDA for model inference
2. **Batch Processing**: Process multiple symbols together
3. **Caching**: Cache chart images and analysis results
4. **Async Processing**: Use Celery for non-blocking analysis

## Integration Examples

### With Existing Trading System

```python
from candlestick_reader.integration import IntegratedCandlestickSystem
from your_trading_system import TradingEngine

class EnhancedTradingEngine(TradingEngine):
    def __init__(self):
        super().__init__()
        self.candlestick_system = IntegratedCandlestickSystem()

    async def analyze_and_trade(self, symbol):
        # Get candlestick signal
        signal = await self.candlestick_system.generate_trading_signal(symbol)

        if signal['action'] != 'HOLD':
            # Execute trade based on signal
            self.execute_trade(
                symbol=symbol,
                action=signal['action'],
                quantity=signal['position_size'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )
```

### With Web Dashboard

```python
from flask import Flask, jsonify
from candlestick_reader.tasks import generate_signal_task

app = Flask(__name__)

@app.route('/api/signal/<symbol>')
def get_signal(symbol):
    # Queue async task
    result = generate_signal_task.delay(symbol, '5min')

    # Return task ID for polling
    return jsonify({'task_id': result.id})

@app.route('/api/result/<task_id>')
def get_result(task_id):
    # Get task result
    from celery.result import AsyncResult
    result = AsyncResult(task_id)

    if result.ready():
        return jsonify(result.get())
    else:
        return jsonify({'status': 'pending'})
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure model files exist and are accessible
2. **Memory Issues**: Reduce batch sizes or use smaller models
3. **Celery Connection Errors**: Check Redis connection and configuration
4. **Chart Generation Errors**: Install matplotlib and ensure display settings

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor Celery tasks:

```bash
# Using Flower
pip install flower
celery -A candlestick_reader.tasks flower
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the test examples

## Roadmap

- [ ] Additional pattern recognition models
- [ ] Real-time streaming data integration
- [ ] Advanced risk management features
- [ ] Multi-timeframe analysis
- [ ] Pattern prediction models
- [ ] Integration with additional data sources