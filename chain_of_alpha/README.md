# Chain-of-Alpha MVP: AI-Driven Alpha Factor Discovery

A standalone implementation of the Chain-of-Alpha framework for automated alpha factor generation and optimization using large language models and quantitative backtesting.

## Overview

This MVP implements the complete Chain-of-Alpha pipeline:

1. **Data Acquisition** - Fetch and preprocess market data
2. **Factor Generation** - Use LLMs to create alpha factors
3. **Backtesting** - Quantitatively evaluate factor performance
4. **Optimization** - Iteratively improve factors based on results
5. **Evaluation & Export** - Comprehensive analysis and reporting

## Features

- **Modular Architecture** - Clean separation of concerns with independent components
- **Multiple LLM Support** - Local Llama models, Grok API, OpenAI API, or mock for testing
- **Comprehensive Backtesting** - Full quantitative evaluation with risk metrics
- **Iterative Optimization** - AI-driven factor improvement based on performance
- **Rich Export Options** - JSON, CSV, HTML reports, and performance charts
- **Production Ready** - Error handling, logging, and configuration management

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Settings

Edit `config.py` to set your preferences:

```python
# Choose your LLM
LLM_CONFIG = {
    'llm_model': 'mock',  # 'llama-3-8b', 'grok', 'openai', or 'mock'
    'llm_api_key': 'your-api-key-here',  # For API models
}

# Set data parameters
DATA_CONFIG = {
    'tickers': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
    'start_date': '2020-01-01',
    'end_date': '2024-01-01',
}
```

### 3. Run the MVP

```bash
python chain_of_alpha_mvp.py
```

## Configuration Options

### LLM Models

- **mock**: Testing mode with predefined responses
- **llama-3-8b**: Local Llama model (requires ~8GB RAM)
- **grok**: xAI Grok API
- **openai**: OpenAI GPT models

### Data Sources

- **yfinance**: Yahoo Finance (free, reliable)
- **alpha_vantage**: Alpha Vantage API (premium features)

### Export Formats

- **json**: Complete results with all metadata
- **csv**: Factor summary table
- **html**: Interactive web report
- **png**: Performance visualization charts

## Output Structure

```
results/
├── chain_of_alpha_results_20241201_143022.json    # Complete results
├── factor_summary_20241201_143022.csv            # Factor table
├── chain_of_alpha_report_20241201_143022.html    # Web report
├── performance_charts_20241201_143022.png        # Charts
└── chain_of_alpha.log                             # Execution log
```

## Architecture

```
ChainOfAlphaMVP
├── DataAcquisition        # Market data fetching & preprocessing
├── LLMInterface          # Unified LLM API (multiple providers)
├── FactorGenerationChain # AI-powered factor creation
├── BacktestingEngine     # Quantitative performance evaluation
├── FactorOptimizationChain # Iterative factor improvement
└── EvaluationExportModule  # Results analysis & export
```

## Key Metrics

The system evaluates factors using:

- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk only
- **Calmar Ratio**: Drawdown-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits / gross losses

## Example Output

```
CHAIN-OF-ALPHA MVP EXECUTION SUMMARY
============================================================
Top Factor Sharpe Ratio: 1.234
Average Sharpe Ratio: 0.856
Top Factor Total Return: 45.67%
Factors Generated: 10
Factors Optimized: 5
Execution Time: 0:02:34
Exported Files: 4
  - results/chain_of_alpha_results_20241201_143022.json
  - results/factor_summary_20241201_143022.csv
  - results/chain_of_alpha_report_20241201_143022.html
  - results/performance_charts_20241201_143022.png
============================================================
```

## Advanced Usage

### Custom Configuration

```python
from chain_of_alpha_mvp import ChainOfAlphaMVP

config = {
    'tickers': ['CUSTOM_STOCKS'],
    'llm_model': 'openai',
    'llm_api_key': 'sk-your-key',
    'num_factors': 20,
    'optimization_iterations': 5,
}

mvp = ChainOfAlphaMVP(config)
results = mvp.run_mvp()
```

### Component-Level Access

```python
from src.data_acquisition import DataAcquisition
from src.factor_generation import FactorGenerationChain

# Use individual components
data_acq = DataAcquisition(config)
market_data = data_acq.fetch_data()

factor_gen = FactorGenerationChain(config, llm_interface)
factors = factor_gen.generate_factors(market_data, num_factors=15)
```

## Troubleshooting

### Common Issues

1. **LLM Import Errors**: Install transformers with `pip install transformers torch`
2. **VectorBT Errors**: Install with `pip install vectorbt`
3. **Memory Issues**: Use smaller models or reduce `num_factors`
4. **API Rate Limits**: Add delays or use local models

### Debug Mode

Set logging to DEBUG in `config.py`:

```python
LOGGING_CONFIG = {
    'level': 'DEBUG',
}
```

## Performance Tips

- **Use Mock Mode** for testing: Fast execution, no dependencies
- **Local Models** for privacy: No API costs, but requires GPU
- **API Models** for quality: Best results, but has costs
- **Batch Processing**: Run multiple configurations overnight

## Contributing

This is a standalone MVP implementation. For the full Chain-of-Alpha framework, see the research paper: [Chain-of-Alpha: Unlocking the Power of Proprietary Alpha Generation](https://arxiv.org/abs/2508.06312)

## License

See LICENSE file for details.