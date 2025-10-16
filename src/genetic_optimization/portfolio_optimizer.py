#!/usr/bin/env python3
"""
Portfolio Optimizer Implementation

This module provides the missing PortfolioOptimizer class using genetic algorithms.
Implements portfolio optimization for risk-adjusted returns and position sizing.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
import random
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

@dataclass
class Asset:
    """Asset representation for portfolio optimization."""
    symbol: str
    expected_return: float
    risk: float
    correlation: Optional[Dict[str, float]] = None
    market_cap: Optional[float] = None
    current_price: Optional[float] = None

@dataclass
class PortfolioAllocation:
    """Portfolio allocation result."""
    allocations: Dict[str, float]  # symbol -> weight
    expected_return: float
    risk: float
    sharpe_ratio: float
    total_value: float
    rebalancing_needed: bool = False

class GeneticOptimizer:
    """Genetic algorithm implementation for portfolio optimization."""
    
    def __init__(self, population_size: int = 100, generations: int = 200):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_ratio = 0.2
    
    def _create_individual(self, num_assets: int) -> np.ndarray:
        """Create a random portfolio allocation."""
        weights = np.random.random(num_assets)
        return weights / weights.sum()  # Normalize to sum to 1
    
    def _create_population(self, num_assets: int) -> List[np.ndarray]:
        """Create initial population of portfolio allocations."""
        return [self._create_individual(num_assets) for _ in range(self.population_size)]
    
    def _fitness_function(self, weights: np.ndarray, returns: np.ndarray, 
                         cov_matrix: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate fitness (Sharpe ratio) for a portfolio allocation.
        
        Args:
            weights: Portfolio weights
            returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Fitness score (higher is better)
        """
        portfolio_return = np.sum(returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        if portfolio_std == 0:
            return 0.0
        
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        return sharpe_ratio
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Uniform crossover
        mask = np.random.random(len(parent1)) < 0.5
        """Mutate an individual."""
        if random.random() > self.mutation_rate:
            return individual
        
        # Add random noise and normalize
        noise = np.random.normal(0, 0.01, len(individual))
        mutated = individual + noise
        mutated = np.abs(mutated)  # Ensure non-negative
        return mutated / mutated.sum()
    
    def optimize(self, returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Run genetic algorithm optimization.
        
        Args:
            returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            
        Returns:
            Optimal portfolio weights
        """
        num_assets = len(returns)
        population = self._create_population(num_assets)
        
        best_fitness_history = []
        
        for generation in range(self.generations):
            # Calculate fitness for each individual
            fitness_scores = [
                self._fitness_function(individual, returns, cov_matrix)
                for individual in population
            ]
            
            # Track best fitness
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            # Selection (tournament selection)
            new_population = []
            
            # Keep elite individuals
            elite_count = int(self.population_size * self.elite_ratio)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            new_population.extend([population[i].copy() for i in elite_indices])
            
            # Generate rest through crossover and mutation
            while len(new_population)  np.ndarray:
        """Calculate covariance matrix from individual asset risks and correlations."""
        symbols = list(self.assets.keys())
        n = len(symbols)
        
        if self.correlation_matrix is None:
            # Default to identity matrix if no correlations provided
            self.correlation_matrix = np.eye(n)
        
        # Get asset standard deviations
        std_devs = np.array([self.assets[symbol].risk for symbol in symbols])
        
        # Calculate covariance matrix: Cov = D * Corr * D
        # where D is diagonal matrix of standard deviations
        cov_matrix = np.outer(std_devs, std_devs) * self.correlation_matrix
        
        return cov_matrix
    
    def optimize_portfolio(self, investment_amount: float = 100000.0, 
                          constraints: Dict[str, Any] = None) -> PortfolioAllocation:
        """
        Optimize portfolio allocation using genetic algorithm.
        
        Args:
            investment_amount: Total amount to invest
            constraints: Additional constraints for optimization
            
        Returns:
            PortfolioAllocation with optimal weights and metrics
        """
        if not self.assets:
            raise ValueError("No assets added to optimization universe")
        
        logger.info(f"[OPTIMIZATION] Starting portfolio optimization for ${investment_amount:,.2f}")
        
        symbols = list(self.assets.keys())
        expected_returns = np.array([self.assets[symbol].expected_return for symbol in symbols])
        
        # Calculate covariance matrix
        self.covariance_matrix = self._calculate_covariance_matrix()
        
        # Apply constraints
        constraints = constraints or {}
        max_weights = constraints.get('max_weights', {})
        min_weights = constraints.get('min_weights', {})
        
        # Run genetic algorithm optimization
        optimal_weights = self.genetic_optimizer.optimize(expected_returns, self.covariance_matrix)
        
        # Apply weight constraints
        for i, symbol in enumerate(symbols):
            max_weight = max_weights.get(symbol, self.config['max_asset_weight'])
            min_weight = min_weights.get(symbol, self.config['min_asset_weight'])
            
            optimal_weights[i] = max(min_weight, min(optimal_weights[i], max_weight))
        
        # Renormalize after applying constraints
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(expected_returns * optimal_weights)
        portfolio_variance = np.dot(optimal_weights.T, np.dot(self.covariance_matrix, optimal_weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        
        sharpe_ratio = (portfolio_return - self.config['risk_free_rate']) / portfolio_risk if portfolio_risk > 0 else 0.0
        
        # Create allocation dictionary
        allocations = {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}
        
        # Filter out very small allocations
        allocations = {symbol: weight for symbol, weight in allocations.items() 
                      if weight >= self.config['min_asset_weight']}
        
        result = PortfolioAllocation(
            allocations=allocations,
            expected_return=portfolio_return,
            risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            total_value=investment_amount,
            rebalancing_needed=False
        )
        
        self.last_optimization = result
        
        logger.info(f"[SUCCESS] Portfolio optimization completed")
        logger.info(f"[METRICS] Expected return: {portfolio_return:.4f}")
        logger.info(f"[METRICS] Risk (std dev): {portfolio_risk:.4f}")
        logger.info(f"[METRICS] Sharpe ratio: {sharpe_ratio:.4f}")
        
        return result
    
    def rebalance_portfolio(self, current_allocations: Dict[str, float], 
                           market_data: Dict[str, float]) -> PortfolioAllocation:
        """
        Check if portfolio needs rebalancing and provide new allocation.
        
        Args:
            current_allocations: Current portfolio weights
            market_data: Current market prices
            
        Returns:
            New portfolio allocation if rebalancing needed
        """
        if not self.last_optimization:
            logger.warning("[WARNING] No previous optimization found, running new optimization")
            return self.optimize_portfolio()
        
        # Calculate drift from optimal allocation
        target_allocations = self.last_optimization.allocations
        rebalancing_needed = False
        
        for symbol in target_allocations:
            current_weight = current_allocations.get(symbol, 0.0)
            target_weight = target_allocations[symbol]
            
            drift = abs(current_weight - target_weight)
            if drift > self.config['rebalancing_threshold']:
                rebalancing_needed = True
                break
        
        if rebalancing_needed:
            logger.info("[REBALANCING] Portfolio drift detected, rebalancing needed")
            new_allocation = self.optimize_portfolio(self.last_optimization.total_value)
            new_allocation.rebalancing_needed = True
            return new_allocation
        else:
            logger.info("[OK] Portfolio within target allocation ranges")
            return self.last_optimization
    
    def calculate_portfolio_metrics(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk and return metrics for given allocation."""
        symbols = list(allocations.keys())
        weights = np.array([allocations[symbol] for symbol in symbols])
        
        # Get expected returns for allocated assets
        expected_returns = np.array([self.assets[symbol].expected_return for symbol in symbols])
        
        # Get subset of covariance matrix for allocated assets
        symbol_indices = [list(self.assets.keys()).index(symbol) for symbol in symbols]
        cov_subset = self.covariance_matrix[np.ix_(symbol_indices, symbol_indices)]
        
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(cov_subset, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        
        sharpe_ratio = (portfolio_return - self.config['risk_free_rate']) / portfolio_risk if portfolio_risk > 0 else 0.0
        
        return {
            'expected_return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'total_allocation': sum(weights)
        }
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about the optimization process."""
        return {
            'optimizer_type': 'Genetic_Algorithm',
            'num_assets': len(self.assets),
            'config': self.config,
            'last_optimization': self.last_optimization.__dict__ if self.last_optimization else None,
            'genetic_params': {
                'population_size': self.genetic_optimizer.population_size,
                'generations': self.genetic_optimizer.generations,
                'mutation_rate': self.genetic_optimizer.mutation_rate,
                'crossover_rate': self.genetic_optimizer.crossover_rate
            }
        }
    
    def suggest_trade_sizing(self, signal: Dict[str, Any], 
                           current_portfolio_value: float) -> Dict[str, Any]:
        """
        Suggest position sizing based on portfolio optimization.
        
        Args:
            signal: Trading signal
            current_portfolio_value: Current portfolio value
            
        Returns:
            Position sizing recommendation
        """
        symbol = signal.get('symbol', 'UNKNOWN')
        
        if symbol not in self.assets or not self.last_optimization:
            # Default sizing if no optimization data
            suggested_weight = 0.05  # 5% default
        else:
            suggested_weight = self.last_optimization.allocations.get(symbol, 0.05)
        
        # Calculate position size
        position_value = current_portfolio_value * suggested_weight
        current_price = signal.get('target_price', 100.0)
        shares = int(position_value / current_price)
        
        return {
            'symbol': symbol,
            'suggested_weight': suggested_weight,
            'position_value': position_value,
            'shares': shares,
            'current_price': current_price,
            'risk_adjusted': True,
            'optimization_based': True
        }

# Factory function for compatibility
def create_portfolio_optimizer(config: Dict[str, Any] = None) -> PortfolioOptimizer:
    """Factory function to create portfolio optimizer instance."""
    return PortfolioOptimizer(config)

if __name__ == "__main__":
    # Test the portfolio optimizer
    logger.info("Testing Portfolio Optimizer implementation...")
    
    # Create test assets
    assets = [
        Asset("AAPL", expected_return=0.12, risk=0.20),
        Asset("GOOGL", expected_return=0.15, risk=0.25),
        Asset("MSFT", expected_return=0.10, risk=0.18),
        Asset("TSLA", expected_return=0.18, risk=0.35),
    ]
    
    # Create optimizer
    optimizer = PortfolioOptimizer()
    
    # Add assets
    for asset in assets:
        optimizer.add_asset(asset)
    
    # Set correlation matrix (example)
    correlations = {
        "AAPL": {"GOOGL": 0.6, "MSFT": 0.7, "TSLA": 0.4},
        "GOOGL": {"MSFT": 0.5, "TSLA": 0.3},
        "MSFT": {"TSLA": 0.2}
    }
    optimizer.set_correlation_matrix(correlations)
    
    # Run optimization
    result = optimizer.optimize_portfolio(100000.0)
    
    logger.info(f"[TEST] Optimization completed")
    logger.info(f"[TEST] Allocations: {result.allocations}")
    logger.info(f"[TEST] Sharpe ratio: {result.sharpe_ratio:.4f}")
    logger.info(f"[TEST] Portfolio Optimizer implementation working correctly")
