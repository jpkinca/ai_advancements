# Week 2 Advanced AI Trading Implementations

This directory contains the complete Week 2 implementation of advanced AI techniques for algorithmic trading. All modules are modular, compartmentalized, and designed to work independently without modifying existing code.

## 🎯 Implementation Overview

### Week 2 Focus Areas
- **Advanced Reinforcement Learning**: PPO with sophisticated trading environments + Multi-Agent Systems
- **Genetic Optimization**: Parameter optimization and portfolio allocation using genetic algorithms  
- **Sparse Spectrum Methods**: Fourier analysis, Wavelet transforms, and Compressed Sensing

## 📁 Module Structure

```
src/
├── reinforcement_learning/
│   ├── __init__.py                    # Module exports and factory functions
│   ├── ppo_advanced.py               # Advanced PPO with Actor-Critic networks
│   └── multi_agent_system.py        # Multi-agent ensemble with regime detection
├── genetic_optimization/
│   ├── __init__.py                    # Module exports and factory functions
│   ├── parameter_optimizer.py        # Genetic algorithm for strategy parameters
│   └── portfolio_genetics.py         # Portfolio allocation optimization
├── sparse_spectrum/
│   ├── __init__.py                    # Module exports and factory functions
│   ├── fourier_analysis.py          # Frequency domain analysis and harmonic patterns
│   ├── wavelet_analysis.py          # Multi-resolution time-frequency analysis
│   └── compressed_sensing.py        # Sparse representation and anomaly detection
└── core/                             # Shared data structures (existing)
    ├── data_structures.py
    └── base_classes.py
```

## 🚀 Quick Start

### Run Complete Demo
```bash
python week2_demo.py
```

### Individual Module Usage

#### Advanced Reinforcement Learning
```python
from src.reinforcement_learning import create_advanced_rl_model, create_multi_agent_trading_system

# Create PPO trading model
rl_config = {
    'environment': {'lookback_window': 20, 'transaction_cost': 0.001},
    'ppo': {'learning_rate': 3e-4, 'gamma': 0.99}
}
rl_model = create_advanced_rl_model(rl_config)

# Create multi-agent system
multi_agent_config = {
    'agents': {'n_trend_agents': 2, 'n_mean_reversion_agents': 2},
    'ensemble': {'voting_threshold': 0.6}
}
multi_agent_system = create_multi_agent_trading_system(multi_agent_config)
```

#### Genetic Optimization
```python
from src.genetic_optimization import create_genetic_optimizer, create_portfolio_genetic_optimizer

# Parameter optimization
param_config = {'population_size': 50, 'generations': 100, 'mutation_rate': 0.1}
genetic_optimizer = create_genetic_optimizer(param_config)

# Portfolio optimization
portfolio_config = {'population_size': 30, 'generations': 50, 'risk_free_rate': 0.02}
portfolio_optimizer = create_portfolio_genetic_optimizer(portfolio_config)
```

#### Sparse Spectrum Methods
```python
from src.sparse_spectrum import (
    create_spectral_trading_model,
    create_wavelet_trading_model, 
    create_compressed_sensing_model
)

# Fourier analysis
fourier_config = {'fourier': {'min_frequency': 0.01, 'max_frequency': 0.5}}
spectral_model = create_spectral_trading_model(fourier_config)

# Wavelet analysis
wavelet_config = {'wavelet': {'wavelet_type': 'db4', 'decomposition_levels': 5}}
wavelet_model = create_wavelet_trading_model(wavelet_config)

# Compressed sensing
cs_config = {'compressed_sensing': {'sparsity_level': 0.1, 'n_components': 50}}
cs_model = create_compressed_sensing_model(cs_config)
```

## 🔬 Technical Implementation Details

### 1. Advanced Reinforcement Learning

#### PPO Implementation (`ppo_advanced.py`)
- **Actor-Critic Architecture**: Separate networks for policy and value estimation
- **Generalized Advantage Estimation (GAE)**: Improved advantage calculation with λ-returns
- **Sophisticated Trading Environment**: Multi-dimensional state space including:
  - Price movements and technical indicators
  - Portfolio state and risk metrics
  - Market volatility and volume patterns
- **Custom Reward Function**: Risk-adjusted returns with transaction costs
- **Experience Buffer**: Efficient storage and sampling of trading experiences

#### Multi-Agent System (`multi_agent_system.py`)
- **Specialized Agents**: Different agents for trend-following, mean-reversion, and volatility trading
- **Market Regime Detection**: Automatic identification of market conditions
- **Meta-Agent Coordination**: Ensemble decision making with confidence weighting
- **Agent Communication**: Shared market state and coordinated actions
- **Dynamic Agent Selection**: Adaptive weighting based on recent performance

### 2. Genetic Optimization

#### Parameter Optimization (`parameter_optimizer.py`)
- **Flexible Gene Encoding**: Support for continuous and discrete parameters
- **Multi-Objective Fitness**: Sharpe ratio, max drawdown, and total return optimization
- **Advanced Crossover**: Blend crossover (BLX-α) for continuous parameters
- **Adaptive Mutation**: Self-adapting mutation rates based on population diversity
- **Strategy Evaluation**: Comprehensive backtesting with realistic transaction costs

#### Portfolio Genetics (`portfolio_genetics.py`)
- **Weight Vector Encoding**: Direct portfolio allocation representation
- **Risk-Return Optimization**: Sharpe ratio maximization with constraint handling
- **Constraint Management**: Long-only, max weight, and sector allocation constraints
- **Efficient Frontier**: Generation of optimal risk-return combinations
- **Covariance Matrix Integration**: Full correlation structure consideration

### 3. Sparse Spectrum Methods

#### Fourier Analysis (`fourier_analysis.py`)
- **Multi-Scale FFT**: Analysis across multiple time horizons
- **Harmonic Pattern Detection**: Identification of cyclical market patterns
- **Spectral Density Estimation**: Power spectrum analysis for dominant frequencies
- **Phase Analysis**: Market timing based on phase relationships
- **Noise Filtering**: Spectral filtering for signal enhancement

#### Wavelet Analysis (`wavelet_analysis.py`)
- **Multi-Resolution Analysis**: Decomposition across time and frequency
- **Adaptive Denoising**: Wavelet thresholding for noise reduction
- **Time-Frequency Signals**: Localized frequency analysis for market events
- **Trend-Cycle Separation**: Isolation of different market components
- **Real-Time Processing**: Efficient online wavelet computation

#### Compressed Sensing (`compressed_sensing.py`)
- **Sparse Feature Extraction**: L1-regularized feature selection from high-dimensional data
- **Dictionary Learning**: Adaptive basis discovery for market patterns
- **Anomaly Detection**: Sparse representation-based outlier identification
- **High-Frequency Patterns**: Compression of tick-level market microstructure
- **Reconstruction Error Analysis**: Quality assessment of sparse representations

## 🎯 Key Features

### Modular Design
- **Independent Modules**: Each module works standalone without dependencies on others
- **Factory Functions**: Consistent creation patterns with `create_*` functions
- **Configuration-Driven**: All parameters externalized through config dictionaries
- **Clean Interfaces**: Well-defined input/output contracts using data classes

### Production Ready
- **Comprehensive Error Handling**: Robust error management with graceful degradation
- **Logging Integration**: Structured logging with ASCII-only output for Windows compatibility
- **Type Annotations**: Full type hints for better IDE support and maintainability
- **Documentation**: Extensive docstrings and inline comments

### Performance Optimized
- **Vectorized Operations**: NumPy and SciPy for efficient numerical computation
- **Memory Management**: Efficient data structures and memory usage patterns
- **Parallel Processing**: Multi-core utilization where applicable
- **Caching**: Intelligent caching of expensive computations

## 📊 Integration Examples

### Signal Ensemble
```python
# Combine signals from multiple models
rl_signals = rl_model.predict(market_data)
genetic_signals = genetic_optimizer.generate_signals(market_data)
spectral_signals = spectral_model.predict(market_data)

# Ensemble combination
ensemble_signals = combine_signals([rl_signals, genetic_signals, spectral_signals])
```

### Multi-Asset Portfolio
```python
# Optimize portfolio across multiple assets
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
portfolio_data = {symbol: get_market_data(symbol) for symbol in symbols}

# Generate allocation using genetic optimization
optimal_allocation = portfolio_optimizer.optimize_portfolio(portfolio_data)

# Apply sparse spectrum analysis to each asset
asset_signals = {}
for symbol in symbols:
    asset_signals[symbol] = spectral_model.predict(portfolio_data[symbol])
```

## 🔧 Configuration Options

### Advanced RL Configuration
```python
rl_config = {
    'environment': {
        'lookback_window': 20,        # Historical data window
        'transaction_cost': 0.001,    # Trading costs
        'max_position': 1.0,          # Maximum position size
        'risk_penalty': 0.1           # Risk penalty coefficient
    },
    'ppo': {
        'learning_rate': 3e-4,        # Optimizer learning rate
        'gamma': 0.99,                # Discount factor
        'gae_lambda': 0.95,           # GAE lambda parameter
        'clip_epsilon': 0.2,          # PPO clipping parameter
        'entropy_coef': 0.01,         # Entropy regularization
        'value_coef': 0.5,            # Value loss coefficient
        'max_grad_norm': 0.5          # Gradient clipping
    },
    'network': {
        'hidden_sizes': [256, 128],   # Network architecture
        'activation': 'tanh',         # Activation function
        'use_batch_norm': True        # Batch normalization
    }
}
```

### Genetic Optimization Configuration
```python
genetic_config = {
    'population_size': 50,            # Population size
    'generations': 100,               # Number of generations
    'mutation_rate': 0.1,            # Mutation probability
    'crossover_rate': 0.8,           # Crossover probability
    'elitism_rate': 0.2,             # Elite preservation rate
    'tournament_size': 3,            # Tournament selection size
    'fitness_weights': {             # Multi-objective weights
        'sharpe_ratio': 0.4,
        'total_return': 0.3,
        'max_drawdown': 0.3
    }
}
```

### Sparse Spectrum Configuration
```python
spectrum_config = {
    'fourier': {
        'min_frequency': 0.01,        # Minimum frequency of interest
        'max_frequency': 0.5,         # Maximum frequency of interest
        'window_type': 'hamming',     # FFT window function
        'overlap_ratio': 0.5,         # Window overlap
        'significance_threshold': 0.05 # Statistical significance level
    },
    'wavelet': {
        'wavelet_type': 'db4',        # Wavelet basis function
        'decomposition_levels': 5,    # Number of decomposition levels
        'denoising_threshold': 0.1,   # Denoising threshold
        'threshold_mode': 'soft'      # Thresholding mode
    },
    'compressed_sensing': {
        'sparsity_level': 0.1,        # Target sparsity
        'l1_alpha': 0.01,            # L1 regularization strength
        'n_components': 50,           # Dictionary size
        'max_iter': 1000,            # Maximum iterations
        'tolerance': 1e-6            # Convergence tolerance
    }
}
```

## 🧪 Testing and Validation

### Unit Tests
Each module includes comprehensive unit tests covering:
- Individual function correctness
- Edge case handling
- Configuration validation
- Error condition management

### Integration Tests
End-to-end testing of complete workflows:
- Data pipeline processing
- Model training and prediction
- Signal generation and validation
- Performance benchmarking

### Backtesting Framework
Built-in backtesting capabilities:
- Historical performance evaluation
- Risk-adjusted return metrics
- Drawdown analysis
- Transaction cost modeling

## 📈 Performance Metrics

The implementation includes comprehensive performance tracking:

### Trading Performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Calmar Ratio**: Annual return over maximum drawdown

### Model Performance
- **Training Convergence**: Learning curve analysis
- **Prediction Accuracy**: Signal hit rate and timing
- **Computational Efficiency**: Runtime and memory usage
- **Statistical Significance**: Hypothesis testing for model effectiveness

## 🔄 Future Enhancements

The modular design enables easy extension:

### Advanced RL Enhancements
- Multi-asset environments
- Hierarchical reinforcement learning
- Model-based RL with environment prediction
- Meta-learning for strategy adaptation

### Genetic Algorithm Extensions
- Coevolutionary algorithms
- Niching and speciation
- Parallel genetic algorithms
- Hybrid optimization approaches

### Spectrum Analysis Additions
- Empirical mode decomposition
- Hilbert-Huang transform
- Multifractal analysis
- Singular spectrum analysis

## 📝 License and Usage

This implementation is designed for educational and research purposes. All modules are self-contained and can be used independently or in combination for algorithmic trading research and development.

---

**Note**: All implementations use ASCII-only output for Windows console compatibility. No emojis or Unicode characters are used in logging or output to prevent encoding errors.
