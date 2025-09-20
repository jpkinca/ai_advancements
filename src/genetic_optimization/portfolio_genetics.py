"""
Portfolio Genetics Module

This module implements genetic algorithms for portfolio optimization:
- Asset allocation optimization
- Risk parity evolution  
- Dynamic rebalancing strategies
- Multi-objective portfolio optimization
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import random
from decimal import Decimal
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..core.data_structures import MarketData, PortfolioPosition
from .parameter_optimizer import Individual, Gene, GeneticConfig, GeneticOptimizer, ParameterSpace, FitnessEvaluator

logger = logging.getLogger(__name__)

@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    max_position_size: float = 0.3  # Maximum weight per asset
    min_position_size: float = 0.01  # Minimum weight per asset
    max_leverage: float = 1.0  # Maximum leverage allowed
    max_sector_concentration: float = 0.5  # Maximum sector concentration
    transaction_costs: float = 0.001  # Transaction cost rate
    rebalance_threshold: float = 0.05  # Rebalancing threshold

@dataclass 
class RiskMetrics:
    """Portfolio risk metrics."""
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
class PortfolioIndividual(Individual):
    """Individual representing a portfolio allocation."""
    
    def __init__(self, asset_weights: List[Gene] = None, **kwargs):
        super().__init__(**kwargs)
        self.asset_weights = asset_weights or []
        self.portfolio_return: float = 0.0
        self.portfolio_risk: float = 0.0
        self.risk_metrics: RiskMetrics = RiskMetrics()
    
    def get_allocation(self) -> Dict[str, float]:
        """Get portfolio allocation as dictionary."""
        allocation = {}
        total_weight = sum(gene.value for gene in self.asset_weights)
        
        if total_weight > 0:
            for gene in self.asset_weights:
                allocation[gene.name] = gene.value / total_weight
        
        return allocation
    
    def normalize_weights(self):
        """Normalize weights to sum to 1."""
        total_weight = sum(gene.value for gene in self.asset_weights)
        
        if total_weight > 0:
            for gene in self.asset_weights:
                gene.value = gene.value / total_weight
    
    def apply_constraints(self, constraints: PortfolioConstraints):
        """Apply portfolio constraints."""
        # Enforce minimum and maximum position sizes
        for gene in self.asset_weights:
            gene.value = np.clip(gene.value, constraints.min_position_size, constraints.max_position_size)
        
        # Normalize after applying constraints
        self.normalize_weights()
        
        # Check leverage constraint
        total_leverage = sum(abs(gene.value) for gene in self.asset_weights)
        if total_leverage > constraints.max_leverage:
            scale_factor = constraints.max_leverage / total_leverage
            for gene in self.asset_weights:
                gene.value *= scale_factor

class PortfolioEvaluator(FitnessEvaluator):
    """Evaluates portfolio performance."""
    
    def __init__(self, 
                 asset_data: Dict[str, List[MarketData]],
                 constraints: PortfolioConstraints = None,
                 risk_free_rate: float = 0.02,
                 objective: str = 'sharpe'):  # 'sharpe', 'return', 'risk_parity', 'max_diversification'
        
        self.asset_data = asset_data
        self.constraints = constraints or PortfolioConstraints()
        self.risk_free_rate = risk_free_rate
        self.objective = objective
        
        # Calculate asset returns
        self.asset_returns = self._calculate_asset_returns()
        self.correlation_matrix = self._calculate_correlation_matrix()
        
        logger.info(f"[SUCCESS] Portfolio evaluator initialized with {len(asset_data)} assets")
    
    def evaluate(self, individual: Individual, market_data: List[MarketData] = None) -> float:
        """Evaluate portfolio fitness."""
        if not isinstance(individual, PortfolioIndividual):
            return 0.0
        
        # Apply constraints
        individual.apply_constraints(self.constraints)
        allocation = individual.get_allocation()
        
        if not allocation:
            return 0.0
        
        # Calculate portfolio metrics
        portfolio_returns = self._calculate_portfolio_returns(allocation)
        portfolio_risk = self._calculate_portfolio_risk(allocation)
        
        # Store metrics in individual
        individual.portfolio_return = np.mean(portfolio_returns) * 252  # Annualized
        individual.portfolio_risk = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        individual.risk_metrics = self._calculate_risk_metrics(portfolio_returns)
        
        # Calculate fitness based on objective
        if self.objective == 'sharpe':
            if individual.portfolio_risk > 0:
                fitness = (individual.portfolio_return - self.risk_free_rate) / individual.portfolio_risk
            else:
                fitness = 0.0
        elif self.objective == 'return':
            fitness = individual.portfolio_return
        elif self.objective == 'risk_parity':
            fitness = self._calculate_risk_parity_score(allocation)
        elif self.objective == 'max_diversification':
            fitness = self._calculate_diversification_ratio(allocation)
        else:
            fitness = individual.portfolio_return / max(individual.portfolio_risk, 0.01)
        
        return max(0.0, fitness)
    
    def get_performance_metrics(self, individual: Individual, market_data: List[MarketData] = None) -> Dict[str, float]:
        """Get detailed portfolio performance metrics."""
        if not isinstance(individual, PortfolioIndividual):
            return {}
        
        allocation = individual.get_allocation()
        portfolio_returns = self._calculate_portfolio_returns(allocation)
        
        return {
            'annual_return': individual.portfolio_return,
            'annual_volatility': individual.portfolio_risk,
            'sharpe_ratio': individual.risk_metrics.sharpe_ratio,
            'sortino_ratio': individual.risk_metrics.sortino_ratio,
            'calmar_ratio': individual.risk_metrics.calmar_ratio,
            'max_drawdown': individual.risk_metrics.max_drawdown,
            'var_95': individual.risk_metrics.var_95,
            'num_assets': len([w for w in allocation.values() if w > 0.01]),
            'concentration': max(allocation.values()) if allocation else 0.0
        }
    
    def _calculate_asset_returns(self) -> Dict[str, np.ndarray]:
        """Calculate returns for each asset."""
        asset_returns = {}
        
        for asset, data in self.asset_data.items():
            prices = [float(d.close) for d in data]
            returns = np.diff(prices) / prices[:-1]
            asset_returns[asset] = returns
        
        return asset_returns
    
    def _calculate_correlation_matrix(self) -> np.ndarray:
        """Calculate correlation matrix between assets."""
        assets = list(self.asset_returns.keys())
        n_assets = len(assets)
        
        if n_assets == 0:
            return np.eye(1)
        
        # Find minimum length to align returns
        min_length = min(len(returns) for returns in self.asset_returns.values())
        
        returns_matrix = np.array([
            self.asset_returns[asset][-min_length:] for asset in assets
        ])
        
        return np.corrcoef(returns_matrix)
    
    def _calculate_portfolio_returns(self, allocation: Dict[str, float]) -> np.ndarray:
        """Calculate portfolio returns given allocation."""
        if not allocation:
            return np.array([0.0])
        
        # Align returns data
        min_length = min(len(self.asset_returns[asset]) for asset in allocation.keys() 
                        if asset in self.asset_returns)
        
        portfolio_returns = np.zeros(min_length)
        
        for asset, weight in allocation.items():
            if asset in self.asset_returns and weight > 0:
                asset_returns = self.asset_returns[asset][-min_length:]
                portfolio_returns += weight * asset_returns
        
        return portfolio_returns
    
    def _calculate_portfolio_risk(self, allocation: Dict[str, float]) -> float:
        """Calculate portfolio volatility."""
        if not allocation:
            return 0.0
        
        assets = list(allocation.keys())
        weights = np.array([allocation[asset] for asset in assets])
        
        # Create covariance matrix
        asset_volatilities = {}
        for asset in assets:
            if asset in self.asset_returns:
                asset_volatilities[asset] = np.std(self.asset_returns[asset])
            else:
                asset_volatilities[asset] = 0.0
        
        vol_vector = np.array([asset_volatilities[asset] for asset in assets])
        
        # Portfolio variance = w^T * Cov * w
        if len(assets) > 1 and self.correlation_matrix.shape[0] >= len(assets):
            cov_matrix = np.outer(vol_vector, vol_vector) * self.correlation_matrix[:len(assets), :len(assets)]
            portfolio_variance = weights.T @ cov_matrix @ weights
        else:
            portfolio_variance = (weights @ vol_vector) ** 2
        
        return np.sqrt(max(0.0, portfolio_variance))
    
    def _calculate_risk_metrics(self, returns: np.ndarray) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        if len(returns) == 0:
            return RiskMetrics()
        
        annual_return = np.mean(returns) * 252
        annual_vol = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns  0 else 0.0
        sortino = (annual_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0.0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        # Calmar ratio
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
        
        return RiskMetrics(
            volatility=annual_vol,
            var_95=var_95,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar
        )
    
    def _calculate_risk_parity_score(self, allocation: Dict[str, float]) -> float:
        """Calculate risk parity score (how equally risk is distributed)."""
        if not allocation:
            return 0.0
        
        # Calculate risk contribution of each asset
        risk_contributions = {}
        total_portfolio_risk = self._calculate_portfolio_risk(allocation)
        
        if total_portfolio_risk == 0:
            return 0.0
        
        for asset, weight in allocation.items():
            if asset in self.asset_returns and weight > 0:
                asset_vol = np.std(self.asset_returns[asset])
                # Simplified risk contribution (weight * volatility / portfolio_risk)
                risk_contrib = (weight * asset_vol) / total_portfolio_risk
                risk_contributions[asset] = risk_contrib
        
        if not risk_contributions:
            return 0.0
        
        # Calculate how evenly distributed the risk is
        risk_values = list(risk_contributions.values())
        target_risk = 1.0 / len(risk_values)  # Equal risk target
        
        # Use negative variance from target as score (higher is better)
        variance_from_target = np.sum([(risk - target_risk) ** 2 for risk in risk_values])
        
        return 1.0 / (1.0 + variance_from_target)  # Convert to score between 0 and 1
    
    def _calculate_diversification_ratio(self, allocation: Dict[str, float]) -> float:
        """Calculate diversification ratio."""
        if not allocation:
            return 0.0
        
        assets = list(allocation.keys())
        weights = np.array([allocation[asset] for asset in assets])
        
        # Weighted average volatility
        weighted_vol = 0.0
        for asset, weight in allocation.items():
            if asset in self.asset_returns:
                asset_vol = np.std(self.asset_returns[asset])
                weighted_vol += weight * asset_vol
        
        # Portfolio volatility
        portfolio_vol = self._calculate_portfolio_risk(allocation)
        
        # Diversification ratio = weighted average vol / portfolio vol
        if portfolio_vol > 0:
            return weighted_vol / portfolio_vol
        else:
            return 1.0

class PortfolioGeneticOptimizer(GeneticOptimizer):
    """Specialized genetic optimizer for portfolio optimization."""
    
    def __init__(self, 
                 asset_names: List[str],
                 asset_data: Dict[str, List[MarketData]],
                 constraints: PortfolioConstraints = None,
                 config: GeneticConfig = None,
                 objective: str = 'sharpe'):
        
        self.asset_names = asset_names
        self.constraints = constraints or PortfolioConstraints()
        
        # Create parameter space for asset weights
        parameter_space = ParameterSpace()
        for asset in asset_names:
            parameter_space.add_parameter(
                f"weight_{asset}", 
                0.0, 
                self.constraints.max_position_size, 
                'float'
            )
        
        # Create evaluator
        fitness_evaluator = PortfolioEvaluator(
            asset_data=asset_data,
            constraints=constraints,
            objective=objective
        )
        
        super().__init__(parameter_space, fitness_evaluator, config)
        
        logger.info(f"[SUCCESS] Portfolio genetic optimizer initialized for {len(asset_names)} assets")
    
    def create_portfolio_individual(self, weights: Optional[Dict[str, float]] = None) -> PortfolioIndividual:
        """Create portfolio individual."""
        asset_weights = []
        
        for asset in self.asset_names:
            if weights and asset in weights:
                weight_value = weights[asset]
            else:
                weight_value = random.uniform(0.0, self.constraints.max_position_size)
            
            gene = Gene(
                name=asset,
                value=weight_value,
                min_value=0.0,
                max_value=self.constraints.max_position_size,
                gene_type='float'
            )
            asset_weights.append(gene)
        
        individual = PortfolioIndividual(asset_weights=asset_weights)
        individual.genes = asset_weights  # For compatibility with base class
        individual.normalize_weights()
        
        return individual
    
    def initialize_portfolio_population(self, seed_portfolios: Optional[List[Dict[str, float]]] = None):
        """Initialize population with portfolio individuals."""
        self.population = []
        
        # Add seed portfolios if provided
        if seed_portfolios:
            for weights in seed_portfolios:
                individual = self.create_portfolio_individual(weights)
                self.population.append(individual)
        
        # Add equal weight portfolio
        equal_weight = 1.0 / len(self.asset_names)
        equal_weights = {asset: equal_weight for asset in self.asset_names}
        equal_weight_individual = self.create_portfolio_individual(equal_weights)
        self.population.append(equal_weight_individual)
        
        # Fill remaining population randomly
        while len(self.population)  Dict[str, Any]:
        """Optimize portfolio allocation."""
        
        # Initialize population
        self.initialize_portfolio_population(seed_portfolios)
        
        # Run optimization (using dummy market_data since evaluator uses asset_data directly)
        results = self.optimize(market_data=[], validation_data=[])
        
        # Convert best individual to readable format
        if self.best_individual and isinstance(self.best_individual, PortfolioIndividual):
            results['best_allocation'] = self.best_individual.get_allocation()
            results['portfolio_metrics'] = {
                'annual_return': self.best_individual.portfolio_return,
                'annual_volatility': self.best_individual.portfolio_risk,
                'sharpe_ratio': self.best_individual.risk_metrics.sharpe_ratio,
                'max_drawdown': self.best_individual.risk_metrics.max_drawdown
            }
        
        logger.info("[SUCCESS] Portfolio optimization completed")
        return results
    
    def get_efficient_frontier(self, n_portfolios: int = 50) -> List[Dict[str, Any]]:
        """Generate efficient frontier portfolios."""
        efficient_portfolios = []
        
        # Generate portfolios with different risk/return profiles
        for i in range(n_portfolios):
            # Create individual with random weights
            individual = self.create_portfolio_individual()
            
            # Evaluate
            fitness = self.fitness_evaluator.evaluate(individual)
            metrics = self.fitness_evaluator.get_performance_metrics(individual)
            
            portfolio_data = {
                'allocation': individual.get_allocation(),
                'return': individual.portfolio_return,
                'risk': individual.portfolio_risk,
                'sharpe': individual.risk_metrics.sharpe_ratio,
                'fitness': fitness,
                'metrics': metrics
            }
            
            efficient_portfolios.append(portfolio_data)
        
        # Sort by risk
        efficient_portfolios.sort(key=lambda x: x['risk'])
        
        logger.info(f"[SUCCESS] Generated {len(efficient_portfolios)} efficient frontier portfolios")
        return efficient_portfolios

def create_portfolio_optimizer(asset_names: List[str],
                             asset_data: Dict[str, List[MarketData]],
                             constraints: PortfolioConstraints = None,
                             config: GeneticConfig = None,
                             objective: str = 'sharpe') -> PortfolioGeneticOptimizer:
    """Factory function to create portfolio genetic optimizer."""
    return PortfolioGeneticOptimizer(asset_names, asset_data, constraints, config, objective)
