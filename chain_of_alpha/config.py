"""
Configuration file for Chain-of-Alpha MVP

This file contains all configuration parameters for the Chain-of-Alpha framework.
"""

# Data Configuration
DATA_CONFIG = {
    'tickers': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
        'NFLX', 'AMD', 'INTC', 'SPY', 'QQQ'
    ],
    'start_date': '2020-01-01',
    'end_date': '2024-01-01',
    'data_source': 'yfinance',  # 'yfinance' or 'alpha_vantage'
}

# LLM Configuration
LLM_CONFIG = {
    'llm_model': 'llama',  # 'llama-3-8b', 'grok', 'openai', or 'mock' for testing
    'llm_api_key': None,  # Set your API key here for Grok/OpenAI
    'huggingface_token': None,  # Set your HF token here for Llama models
    'temperature': 0.7,
    'max_tokens': 1000,
    'model_path': None,  # For local models
}

# Factor Generation Configuration
GENERATION_CONFIG = {
    'num_factors': 10,  # Number of factors to generate
    'factor_complexity': 'medium',  # 'simple', 'medium', 'complex'
    'include_technical': True,
    'include_volume': True,
    'include_price': True,
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'initial_capital': 10000,
    'commission': 0.001,  # 0.1% per trade
    'slippage': 0.001,   # 0.1% slippage
    'benchmark_ticker': 'SPY',
    'rebalance_freq': 'D',  # Daily rebalancing
    'factor_threshold': 0.0,  # Signal threshold
}

# Optimization Configuration
OPTIMIZATION_CONFIG = {
    'max_iterations': 3,
    'optimization_method': 'iterative',  # 'iterative' or 'genetic'
    'selection_criteria': 'sharpe_ratio',  # 'sharpe_ratio', 'sortino', 'calmar'
    'risk_penalty': 0.1,  # Penalty for high volatility
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'metrics': ['sharpe', 'sortino', 'calmar', 'max_drawdown', 'win_rate', 'profit_factor'],
    'benchmark_comparison': True,
    'robustness_tests': ['out_of_sample', 'different_periods'],
    'statistical_tests': ['t_test', 'normality_test'],
}

# Export Configuration
EXPORT_CONFIG = {
    'output_dir': 'results',
    'export_formats': ['json', 'csv', 'html', 'png'],
    'save_factors': True,
    'save_backtests': True,
    'save_charts': True,
    'timestamp_results': True,
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'file_logging': True,
    'log_file': 'chain_of_alpha.log',
    'console_logging': True,
}

# Complete Configuration Dictionary
CONFIG = {
    **DATA_CONFIG,
    **LLM_CONFIG,
    **GENERATION_CONFIG,
    **BACKTEST_CONFIG,
    **OPTIMIZATION_CONFIG,
    **EVALUATION_CONFIG,
    **EXPORT_CONFIG,
    **LOGGING_CONFIG,
}