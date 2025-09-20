"""
Genetic Algorithm Parameter Optimization

This module implements genetic algorithms for trading strategy optimization:
- Hyperparameter optimization for AI models
- Technical indicator parameter evolution
- Portfolio allocation optimization
- Strategy performance evolution
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import random
from decimal import Decimal
from dataclasses import dataclass, field
from datetime import datetime
import logging
import copy
from abc import ABC, abstractmethod

# Remove problematic relative imports - use local definitions instead
from typing import Any, List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Local definitions to avoid import issues
@dataclass
class MarketDataLocal:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass 
class TradingSignalLocal:
    symbol: str
    signal_type: str
    confidence: float
    timestamp: datetime
    metadata: dict = None

# Use Any for MarketData type to avoid import issues
MarketData = Any

logger = logging.getLogger(__name__)

@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 50
    n_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_rate: float = 0.2
    tournament_size: int = 3
    convergence_threshold: float = 1e-6
    patience: int = 10

@dataclass
class Gene:
    """Individual gene representing a parameter."""
    name: str
    value: float
    min_value: float
    max_value: float
    gene_type: str = 'float'  # 'float', 'int', 'bool'
    
    def mutate(self, mutation_strength: float = 0.1):
        """Mutate gene value."""
        if self.gene_type == 'bool':
            if random.random()  'Gene':
        """Create copy of gene."""
        return Gene(
            name=self.name,
            value=self.value,
            min_value=self.min_value,
            max_value=self.max_value,
            gene_type=self.gene_type
        )

@dataclass
class Individual:
    """Individual in genetic algorithm population."""
    genes: List[Gene] = field(default_factory=list)
    fitness: float = 0.0
    age: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter dictionary from genes."""
        params = {}
        for gene in self.genes:
            if gene.gene_type == 'bool':
                params[gene.name] = bool(gene.value > 0.5)
            elif gene.gene_type == 'int':
                params[gene.name] = int(gene.value)
            else:
                params[gene.name] = gene.value
        return params
    
    def mutate(self, mutation_rate: float, adaptive_strength: bool = True):
        """Mutate individual."""
        # Adaptive mutation strength based on age and fitness
        if adaptive_strength:
            mutation_strength = mutation_rate * (1 + self.age * 0.01) * (1 / max(0.1, self.fitness))
        else:
            mutation_strength = mutation_rate
        
        for gene in self.genes:
            gene.mutate(mutation_strength)
        
        self.age += 1
    
    def copy(self) -> 'Individual':
        """Create copy of individual."""
        return Individual(
            genes=[gene.copy() for gene in self.genes],
            fitness=self.fitness,
            age=self.age,
            performance_metrics=self.performance_metrics.copy()
        )

class ParameterSpace:
    """Defines parameter space for optimization."""
    
    def __init__(self):
        self.parameters: Dict[str, Dict] = {}
    
    def add_parameter(self, name: str, min_val: float, max_val: float, 
                     param_type: str = 'float', default: Optional[float] = None):
        """Add parameter to optimization space."""
        self.parameters[name] = {
            'min': min_val,
            'max': max_val,
            'type': param_type,
            'default': default or (min_val + max_val) / 2
        }
        
        logger.debug(f"[DATA] Added parameter {name}: [{min_val}, {max_val}] ({param_type})")
    
    def create_random_individual(self) -> Individual:
        """Create random individual from parameter space."""
        genes = []
        for name, params in self.parameters.items():
            if params['type'] == 'bool':
                value = random.choice([0.0, 1.0])
            else:
                value = random.uniform(params['min'], params['max'])
                if params['type'] == 'int':
                    value = round(value)
            
            gene = Gene(
                name=name,
                value=value,
                min_value=params['min'],
                max_value=params['max'],
                gene_type=params['type']
            )
            genes.append(gene)
        
        return Individual(genes=genes)
    
    def create_individual_from_params(self, parameters: Dict[str, Any]) -> Individual:
        """Create individual from parameter dictionary."""
        genes = []
        for name, params in self.parameters.items():
            value = parameters.get(name, params['default'])
            
            gene = Gene(
                name=name,
                value=float(value),
                min_value=params['min'],
                max_value=params['max'],
                gene_type=params['type']
            )
            genes.append(gene)
        
        return Individual(genes=genes)

class FitnessEvaluator(ABC):
    """Abstract base class for fitness evaluation."""
    
    @abstractmethod
    def evaluate(self, individual: Individual, market_data: List[MarketData]) -> float:
        """Evaluate fitness of individual."""
        pass
    
    @abstractmethod
    def get_performance_metrics(self, individual: Individual, market_data: List[MarketData]) -> Dict[str, float]:
        """Get detailed performance metrics."""
        pass

class TradingStrategyEvaluator(FitnessEvaluator):
    """Evaluates trading strategy performance."""
    
    def __init__(self, 
                 strategy_function: Callable,
                 initial_balance: float = 100000.0,
                 transaction_cost: float = 0.001):
        self.strategy_function = strategy_function
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
    
    def evaluate(self, individual: Individual, market_data: List[MarketData]) -> float:
        """Evaluate trading strategy fitness."""
        try:
            # Get parameters from individual
            params = individual.get_parameters()
            
            # Run strategy simulation
            portfolio_values, trades = self._simulate_strategy(params, market_data)
            
            if len(portfolio_values)  0:
                sharpe = total_return / volatility
            else:
                sharpe = total_return
            
            # Maximum drawdown penalty
            max_dd = self._calculate_max_drawdown(portfolio_values)
            drawdown_penalty = max(0, max_dd - 0.1) * 5  # Penalty for >10% drawdown
            
            # Trade frequency penalty (avoid overtrading)
            trade_frequency = len(trades) / len(market_data)
            frequency_penalty = max(0, trade_frequency - 0.1) * 2
            
            # Final fitness score
            fitness = sharpe - drawdown_penalty - frequency_penalty
            
            return max(0.0, fitness)  # Ensure non-negative fitness
            
        except Exception as e:
            logger.warning(f"[WARNING] Strategy evaluation failed: {e}")
            return 0.0
    
    def get_performance_metrics(self, individual: Individual, market_data: List[MarketData]) -> Dict[str, float]:
        """Get detailed performance metrics."""
        try:
            params = individual.get_parameters()
            portfolio_values, trades = self._simulate_strategy(params, market_data)
            
            if len(portfolio_values)  0 else 0.0
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
            win_rate = winning_trades / len(trades) if trades else 0.0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'num_trades': len(trades),
                'win_rate': win_rate,
                'avg_trade_return': np.mean([trade['profit'] for trade in trades]) if trades else 0.0
            }
            
        except Exception as e:
            logger.warning(f"[WARNING] Metrics calculation failed: {e}")
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'num_trades': 0}
    
    def _simulate_strategy(self, params: Dict[str, Any], market_data: List[MarketData]) -> Tuple[List[float], List[Dict]]:
        """Simulate trading strategy with given parameters."""
        balance = self.initial_balance
        position = 0.0
        portfolio_values = [balance]
        trades = []
        
        for i, data in enumerate(market_data[1:], 1):
            current_price = float(data.close)
            
            # Get strategy signal
            signal = self.strategy_function(params, market_data[:i+1])
            
            if signal and signal != "HOLD":
                # Execute trade
                if signal == "BUY" and position = 0:
                    # Sell signal
                    if position > 0:  # Close long position
                        balance += position * current_price * (1 - self.transaction_cost)
                        profit = position * current_price - position * current_price  # Will be calculated properly with entry price
                        trades.append({'type': 'sell', 'price': current_price, 'profit': profit})
                    
                    # Open short position (simplified)
                    shares = (balance * 0.95) / current_price
                    balance += shares * current_price * (1 - self.transaction_cost)
                    position = -shares
            
            # Calculate portfolio value
            portfolio_value = balance + position * current_price
            portfolio_values.append(portfolio_value)
        
        return portfolio_values, trades
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values)  peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd

class GeneticOptimizer:
    """Genetic algorithm optimizer for trading strategies."""
    
    def __init__(self, 
                 parameter_space: ParameterSpace,
                 fitness_evaluator: FitnessEvaluator,
                 config: GeneticConfig = None):
        
        self.parameter_space = parameter_space
        self.fitness_evaluator = fitness_evaluator
        self.config = config or GeneticConfig()
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        
        logger.info("[SUCCESS] Genetic Optimizer initialized")
    
    def initialize_population(self, seed_individuals: Optional[List[Dict[str, Any]]] = None):
        """Initialize population with random individuals."""
        self.population = []
        
        # Add seed individuals if provided
        if seed_individuals:
            for params in seed_individuals:
                individual = self.parameter_space.create_individual_from_params(params)
                self.population.append(individual)
        
        # Fill remaining population randomly
        while len(self.population)  Dict[str, Any]:
        """Run genetic algorithm optimization."""
        
        if not self.population:
            self.initialize_population()
        
        optimization_results = {
            'best_fitness_history': [],
            'avg_fitness_history': [],
            'best_parameters': {},
            'best_performance_metrics': {},
            'convergence_generation': None
        }
        
        stagnation_counter = 0
        prev_best_fitness = -np.inf
        
        for generation in range(self.config.n_generations):
            self.generation = generation
            
            # Evaluate population
            self._evaluate_population(market_data)
            
            # Track best individual
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best.copy()
            
            # Record statistics
            fitness_values = [ind.fitness for ind in self.population]
            best_fitness = max(fitness_values)
            avg_fitness = np.mean(fitness_values)
            
            optimization_results['best_fitness_history'].append(best_fitness)
            optimization_results['avg_fitness_history'].append(avg_fitness)
            
            # Check for convergence
            if abs(best_fitness - prev_best_fitness) = self.config.patience:
                optimization_results['convergence_generation'] = generation
                logger.info(f"[SUCCESS] Convergence reached at generation {generation}")
                break
            
            prev_best_fitness = best_fitness
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"[PROCESSING] Generation {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
            
            # Create next generation
            if generation  List[Individual]:
        """Create next generation using selection, crossover, and mutation."""
        next_generation = []
        
        # Elitism: Keep best individuals
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        elite_count = int(self.config.population_size * self.config.elitism_rate)
        next_generation.extend([ind.copy() for ind in self.population[:elite_count]])
        
        # Generate offspring
        while len(next_generation)  Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, min(self.config.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single-point crossover for genes
        if len(parent1.genes) > 1:
            crossover_point = random.randint(1, len(parent1.genes) - 1)
            
            # Swap genes after crossover point
            for i in range(crossover_point, len(parent1.genes)):
                child1.genes[i].value = parent2.genes[i].value
                child2.genes[i].value = parent1.genes[i].value
        
        return child1, child2
    
    def get_population_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population)  0:
                        normalized_diff = abs(gene1.value - gene2.value) / range_size
                        distance += normalized_diff ** 2
                
                diversity_sum += np.sqrt(distance)
                comparison_count += 1
        
        return diversity_sum / comparison_count if comparison_count > 0 else 0.0

class TechnicalIndicatorOptimizer:
    """Specialized optimizer for technical indicator parameters."""
    
    def __init__(self, genetic_config: GeneticConfig = None):
        self.config = genetic_config or GeneticConfig()
        
    def optimize_moving_average_strategy(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Optimize moving average crossover strategy."""
        
        # Define parameter space
        param_space = ParameterSpace()
        param_space.add_parameter('short_ma_period', 5, 50, 'int')
        param_space.add_parameter('long_ma_period', 20, 200, 'int')
        param_space.add_parameter('position_size', 0.1, 1.0, 'float')
        param_space.add_parameter('stop_loss_pct', 0.01, 0.1, 'float')
        
        # Define strategy function
        def ma_strategy(params: Dict[str, Any], data: List[MarketData]) -> str:
            if len(data)  long_ma * 1.01:  # 1% threshold
                return "BUY"
            elif short_ma  Dict[str, Any]:
        """Optimize RSI-based trading strategy."""
        
        # Define parameter space
        param_space = ParameterSpace()
        param_space.add_parameter('rsi_period', 7, 30, 'int')
        param_space.add_parameter('oversold_threshold', 20, 40, 'float')
        param_space.add_parameter('overbought_threshold', 60, 80, 'float')
        param_space.add_parameter('position_size', 0.1, 1.0, 'float')
        
        # Define RSI strategy function
        def rsi_strategy(params: Dict[str, Any], data: List[MarketData]) -> str:
            if len(data)  0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-change)
            
            if len(gains) == 0:
                return "HOLD"
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Generate signal
            if rsi  params['overbought_threshold']:
                return "SELL"
            else:
                return "HOLD"
        
        # Create evaluator and optimizer
        evaluator = TradingStrategyEvaluator(rsi_strategy)
        optimizer = GeneticOptimizer(param_space, evaluator, self.config)
        
        # Run optimization
        results = optimizer.optimize(market_data)
        
        logger.info("[SUCCESS] RSI strategy optimization completed")
        return results

def create_genetic_optimizer(parameter_space: ParameterSpace, 
                           fitness_evaluator: FitnessEvaluator,
                           config: GeneticConfig = None) -> GeneticOptimizer:
    """Factory function to create genetic optimizer."""
    return GeneticOptimizer(parameter_space, fitness_evaluator, config)

# Alias for compatibility with demo script
class ParameterOptimizer:
    """
    Parameter Optimizer wrapper for GeneticOptimizer
    
    Provides a simplified interface compatible with the demo script.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # Convert config dict to GeneticConfig
        if config is None:
            config = {}
        
        self.genetic_config = GeneticConfig(
            population_size=config.get('population_size', 50),
            n_generations=config.get('generations', 100),
            mutation_rate=config.get('mutation_rate', 0.1),
            crossover_rate=config.get('crossover_rate', 0.8),
            elitism_rate=config.get('elite_size', 10) / config.get('population_size', 50),
            tournament_size=config.get('tournament_size', 5)
        )
        
        self.config = config
        logger.info("[SUCCESS] Parameter Optimizer wrapper initialized")
    
    def optimize_parameters(self, market_data: List, parameter_ranges: Dict[str, Tuple[float, float]], 
                          generations: int = None) -> Dict[str, Any]:
        """
        Optimize parameters using genetic algorithm.
        
        Args:
            market_data: Market data for backtesting
            parameter_ranges: Dictionary of parameter names and their (min, max) ranges
            generations: Number of generations to run
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"[OPTIMIZATION] Starting parameter optimization")
        
        # Create parameter space and add parameters
        param_space = ParameterSpace()  # FIXED: Remove the 'genes' argument
        
        for param_name, param_range in parameter_ranges.items():
            # Handle both tuple (min, max) and dict formats
            if isinstance(param_range, tuple) and len(param_range) == 2:
                min_val, max_val = param_range
                param_type = 'int' if isinstance(min_val, int) and isinstance(max_val, int) else 'float'
            elif isinstance(param_range, dict):
                min_val = param_range.get('min', 0)
                max_val = param_range.get('max', 1)
                param_type = param_range.get('type', 'float')
            else:
                logger.warning(f"[WARNING] Invalid parameter range format for {param_name}: {param_range}")
                continue
            
            param_space.add_parameter(param_name, min_val, max_val, param_type)
        
        # Create simple fitness evaluator
        def simple_strategy_fitness(individual: Individual, data: List) -> float:
            """Simple moving average crossover strategy fitness."""
            params = individual.get_parameters()
            
            # Get parameters with defaults
            sma_short = int(params.get('sma_short', 10))
            sma_long = int(params.get('sma_long', 30))
            
            # Extract prices
            prices = [float(getattr(md, 'close', 100.0)) for md in data]
            
            if len(prices)  sma_long_val:
                    # Enter long
                    position = 1
                    entry_price = prices[i]
                    trades += 1
                elif position == 1 and sma_short_val  float:
                return simple_strategy_fitness(individual, market_data)
            
            def get_performance_metrics(self, individual: Individual, market_data: List) -> Dict[str, float]:
                # Implement a basic version to satisfy the abstract method
                return {'fitness': individual.fitness}
        
        evaluator = SimpleEvaluator()
        
        # Override generations if specified
        if generations is not None:
            self.genetic_config.n_generations = generations
        
        try:
            # Create and run optimizer
            optimizer = GeneticOptimizer(param_space, evaluator, self.genetic_config)
            results = optimizer.optimize(market_data)
            
            # Convert results to expected format
            best_individual = optimizer.best_individual  # FIXED: Access from optimizer instance
            if best_individual:
                best_params = best_individual.get_parameters()
                best_fitness = best_individual.fitness
            else:
                best_params = {}
                best_fitness = 0.0
            
            optimization_results = {
                'best_parameters': best_params,
                'best_fitness': best_fitness,
                'total_generations': self.genetic_config.n_generations,
                'optimization_successful': best_fitness > 0
            }
            
            logger.info(f"[SUCCESS] Parameter optimization completed")
            logger.info(f"[RESULTS] Best fitness: {best_fitness:.4f}")
            logger.info(f"[RESULTS] Best parameters: {best_params}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"[ERROR] Parameter optimization failed: {e}")
            return {
                'best_parameters': {},
                'best_fitness': 0.0,
                'total_generations': 0,
                'optimization_successful': False,
                'error': str(e)
            }
