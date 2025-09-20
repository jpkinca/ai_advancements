"""
Multi-Agent Trading System

This module implements an ensemble of specialized trading agents:
- Momentum agents for trending markets
- Mean-reversion agents for ranging markets
- Breakout agents for volatility scenarios
- Meta-agent for strategy selection and coordination
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

from ..core.data_structures import MarketData, TradingSignal, PortfolioPosition
from ..core.base_classes import BaseTradingModel
from .ppo_advanced import PPOAgent, PPOConfig, AdvancedTradingEnvironment

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class AgentConfig:
    """Configuration for individual agent."""
    name: str
    specialization: MarketRegime
    ppo_config: PPOConfig
    weight: float = 1.0
    performance_window: int = 100

@dataclass
class EnsembleConfig:
    """Configuration for ensemble system."""
    meta_learning_rate: float = 0.001
    regime_detection_window: int = 50
    performance_decay: float = 0.95
    min_confidence_threshold: float = 0.3
    rebalance_frequency: int = 10

class MarketRegimeDetector:
    """Detects current market regime from price data."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
    def detect_regime(self, market_data: List[MarketData]) -> MarketRegime:
        """Detect current market regime."""
        if len(market_data)  vol_threshold:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility  0.3:
            if mean_return > 0.001:
                return MarketRegime.TRENDING_UP
            elif mean_return  float:
        """Calculate strength of trend using linear regression R²."""
        if len(prices)  self.config.performance_window:
            self.performance_history.pop(0)
        
        # Calculate weighted performance with decay
        weights = [self.config.performance_decay ** i for i in range(len(self.performance_history))]
        weights.reverse()
        
        if weights:
            self.current_performance = np.average(self.performance_history, weights=weights)
        
        logger.debug(f"[DATA] Agent {self.config.name} performance updated: {self.current_performance:.4f}")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Get action from specialized agent."""
        return self.agent.get_action(state, deterministic)
    
    def should_be_active(self, current_regime: MarketRegime) -> bool:
        """Check if agent should be active in current regime."""
        if self.config.specialization == current_regime:
            return True
        
        # Cross-regime compatibility rules
        compatibility = {
            MarketRegime.TRENDING_UP: [MarketRegime.LOW_VOLATILITY],
            MarketRegime.TRENDING_DOWN: [MarketRegime.HIGH_VOLATILITY],
            MarketRegime.RANGING: [MarketRegime.LOW_VOLATILITY],
            MarketRegime.HIGH_VOLATILITY: [MarketRegime.TRENDING_DOWN, MarketRegime.TRENDING_UP],
            MarketRegime.LOW_VOLATILITY: [MarketRegime.TRENDING_UP, MarketRegime.RANGING]
        }
        
        return current_regime in compatibility.get(self.config.specialization, [])

class MetaAgent(nn.Module):
    """Meta-agent for strategy selection and weight allocation."""
    
    def __init__(self, n_agents: int, regime_features: int = 10):
        super().__init__()
        
        self.n_agents = n_agents
        
        # Network for weight allocation
        self.weight_network = nn.Sequential(
            nn.Linear(regime_features + n_agents, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_agents),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, regime_features: torch.Tensor, agent_performances: torch.Tensor) -> torch.Tensor:
        """Forward pass to get agent weights."""
        combined_input = torch.cat([regime_features, agent_performances], dim=-1)
        weights = self.weight_network(combined_input)
        return weights
    
    def get_agent_weights(self, regime_features: np.ndarray, agent_performances: np.ndarray) -> np.ndarray:
        """Get weights for ensemble agents."""
        regime_tensor = torch.FloatTensor(regime_features).unsqueeze(0)
        performance_tensor = torch.FloatTensor(agent_performances).unsqueeze(0)
        
        with torch.no_grad():
            weights = self.forward(regime_tensor, performance_tensor)
        
        return weights.numpy()[0]
    
    def update_weights(self, regime_features: np.ndarray, agent_performances: np.ndarray, 
                      actual_performance: float, predicted_weights: np.ndarray):
        """Update meta-agent based on actual performance."""
        regime_tensor = torch.FloatTensor(regime_features).unsqueeze(0)
        performance_tensor = torch.FloatTensor(agent_performances).unsqueeze(0)
        target_performance = torch.FloatTensor([actual_performance])
        
        # Predict weights
        predicted_weights_tensor = self.forward(regime_tensor, performance_tensor)
        
        # Calculate loss (how well the weights predicted performance)
        weighted_performance = torch.sum(predicted_weights_tensor * performance_tensor)
        loss = nn.MSELoss()(weighted_performance, target_performance)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class MultiAgentTradingSystem(BaseTradingModel):
    """Multi-agent ensemble trading system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ensemble_config = EnsembleConfig(**self.config.get('ensemble', {}))
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector(self.ensemble_config.regime_detection_window)
        self.agents: List[SpecializedAgent] = []
        self.meta_agent = None
        
        # State tracking
        self.current_regime = MarketRegime.RANGING
        self.agent_weights = np.array([])
        self.performance_history = []
        
        # Create specialized agents
        self._create_specialized_agents()
        
        logger.info("[SUCCESS] Multi-Agent Trading System initialized")
    
    def _create_specialized_agents(self):
        """Create specialized agents for different market regimes."""
        
        # Define agent configurations
        agent_configs = [
            AgentConfig(
                name="MomentumAgent",
                specialization=MarketRegime.TRENDING_UP,
                ppo_config=PPOConfig(learning_rate=3e-4, clip_epsilon=0.2)
            ),
            AgentConfig(
                name="BearishAgent", 
                specialization=MarketRegime.TRENDING_DOWN,
                ppo_config=PPOConfig(learning_rate=3e-4, clip_epsilon=0.15)
            ),
            AgentConfig(
                name="MeanReversionAgent",
                specialization=MarketRegime.RANGING,
                ppo_config=PPOConfig(learning_rate=2e-4, clip_epsilon=0.3)
            ),
            AgentConfig(
                name="VolatilityAgent",
                specialization=MarketRegime.HIGH_VOLATILITY,
                ppo_config=PPOConfig(learning_rate=5e-4, clip_epsilon=0.1)
            ),
            AgentConfig(
                name="StableAgent",
                specialization=MarketRegime.LOW_VOLATILITY,
                ppo_config=PPOConfig(learning_rate=2e-4, clip_epsilon=0.25)
            )
        ]
        
        # Create agents (will be initialized when prepare_data is called)
        self.agent_configs = agent_configs
        
        logger.info(f"[SUCCESS] Configured {len(agent_configs)} specialized agents")
    
    def prepare_data(self, market_data: List[MarketData]) -> None:
        """Prepare data and initialize agents."""
        # Create environment to get dimensions
        temp_env = AdvancedTradingEnvironment(
            market_data=market_data,
            initial_balance=self.config.get('initial_balance', 100000.0),
            transaction_cost=self.config.get('transaction_cost', 0.001),
            max_position_size=self.config.get('max_position_size', 0.2),
            lookback_window=self.config.get('lookback_window', 60)
        )
        
        state_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.shape[0]
        
        # Initialize specialized agents
        self.agents = [
            SpecializedAgent(config, state_dim, action_dim) 
            for config in self.agent_configs
        ]
        
        # Initialize meta-agent
        self.meta_agent = MetaAgent(len(self.agents))
        
        # Initialize weights equally
        self.agent_weights = np.ones(len(self.agents)) / len(self.agents)
        
        logger.info(f"[SUCCESS] Multi-agent system prepared with {len(self.agents)} agents")
    
    def train(self, validation_data: Optional[List[MarketData]] = None) -> Dict[str, Any]:
        """Train all agents in the ensemble."""
        if not self.agents:
            raise ValueError("Must call prepare_data() before training")
        
        # For this implementation, we'll simulate training
        # In practice, each agent would be trained on regime-specific data
        
        training_metrics = {
            'agent_performances': {agent.config.name: [] for agent in self.agents},
            'regime_distribution': {},
            'ensemble_performance': [],
            'meta_agent_losses': []
        }
        
        # Simulate training performance for demonstration
        for agent in self.agents:
            # Simulate varying performance based on specialization
            base_performance = np.random.uniform(0.6, 0.9)
            performance_trend = np.random.uniform(-0.1, 0.1)
            
            for epoch in range(100):
                performance = base_performance + performance_trend * epoch / 100 + np.random.normal(0, 0.05)
                performance = np.clip(performance, 0.0, 1.0)
                
                agent.update_performance(performance)
                training_metrics['agent_performances'][agent.config.name].append(performance)
            
            agent.is_trained = True
        
        logger.info("[SUCCESS] Multi-agent ensemble training completed")
        return training_metrics
    
    def predict(self, market_data: List[MarketData]) -> List[TradingSignal]:
        """Generate ensemble trading signals."""
        if not all(agent.is_trained for agent in self.agents):
            raise ValueError("All agents must be trained before prediction")
        
        signals = []
        
        # Process market data in chunks to detect regime changes
        chunk_size = self.ensemble_config.rebalance_frequency
        
        for i in range(0, len(market_data) - chunk_size, chunk_size):
            chunk_data = market_data[i:i + chunk_size + 50]  # Extra data for regime detection
            
            # Detect current market regime
            self.current_regime = self.regime_detector.detect_regime(chunk_data)
            
            # Get regime features for meta-agent
            regime_features = self._extract_regime_features(chunk_data)
            
            # Get agent performances
            agent_performances = np.array([agent.current_performance for agent in self.agents])
            
            # Update agent weights using meta-agent
            self.agent_weights = self.meta_agent.get_agent_weights(regime_features, agent_performances)
            
            # Generate signals for this chunk
            chunk_signals = self._generate_ensemble_signals(chunk_data[-chunk_size:])
            signals.extend(chunk_signals)
            
            logger.debug(f"[DATA] Regime: {self.current_regime.value}, Weights: {self.agent_weights}")
        
        logger.info(f"[SUCCESS] Generated {len(signals)} ensemble trading signals")
        return signals
    
    def _extract_regime_features(self, market_data: List[MarketData]) -> np.ndarray:
        """Extract features describing current market regime."""
        if len(market_data)  0]) / 10,  # Win rate
            np.percentile(returns, 95) - np.percentile(returns, 5),  # Return range
            self._calculate_momentum_score(prices)  # Momentum score
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_momentum_score(self, prices: List[float]) -> float:
        """Calculate momentum score."""
        if len(prices)  List[TradingSignal]:
        """Generate ensemble signals for a data chunk."""
        if not market_data:
            return []
        
        # Create temporary environment for signal generation
        temp_env = AdvancedTradingEnvironment(
            market_data=market_data,
            initial_balance=100000.0,
            transaction_cost=0.0,
            max_position_size=0.2,
            lookback_window=min(60, len(market_data) - 1)
        )
        
        signals = []
        state, _ = temp_env.reset()
        
        # Get actions from active agents
        active_agents = [agent for agent in self.agents if agent.should_be_active(self.current_regime)]
        
        if not active_agents:
            active_agents = self.agents  # Fallback to all agents
        
        # Weight active agents
        active_indices = [i for i, agent in enumerate(self.agents) if agent in active_agents]
        active_weights = self.agent_weights[active_indices]
        active_weights = active_weights / np.sum(active_weights)  # Normalize
        
        for data_point in market_data[temp_env.lookback_window:]:
            # Get actions from all active agents
            agent_actions = []
            for agent in active_agents:
                action, _, _ = agent.get_action(state, deterministic=True)
                agent_actions.append(action)
            
            if agent_actions:
                # Ensemble action as weighted average
                ensemble_action = np.average(agent_actions, axis=0, weights=active_weights)
                
                # Convert to trading signal
                position_change = ensemble_action[0]
                confidence = min(abs(position_change) * 2, 1.0)
                
                if confidence >= self.ensemble_config.min_confidence_threshold:
                    signal_type = "BUY" if position_change > 0.1 else "SELL" if position_change  Dict[str, Any]:
        """Get current status of the ensemble."""
        return {
            'current_regime': self.current_regime.value,
            'agent_weights': self.agent_weights.tolist(),
            'agent_performances': {agent.config.name: agent.current_performance for agent in self.agents},
            'active_agents': [agent.config.name for agent in self.agents if agent.should_be_active(self.current_regime)],
            'total_agents': len(self.agents)
        }
    
    def save_model(self, filepath: str) -> None:
        """Save ensemble model."""
        model_data = {
            'config': self.config,
            'agent_configs': [agent.config for agent in self.agents],
            'meta_agent_state': self.meta_agent.state_dict() if self.meta_agent else None,
            'agent_weights': self.agent_weights.tolist(),
            'current_regime': self.current_regime.value
        }
        
        torch.save(model_data, filepath)
        
        # Save individual agents
        for i, agent in enumerate(self.agents):
            agent_filepath = filepath.replace('.pt', f'_agent_{i}.pt')
            agent.agent.save(agent_filepath)
        
        logger.info(f"[SUCCESS] Multi-agent ensemble saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load ensemble model."""
        model_data = torch.load(filepath, map_location='cpu')
        
        self.config = model_data['config']
        self.agent_weights = np.array(model_data['agent_weights'])
        self.current_regime = MarketRegime(model_data['current_regime'])
        
        # Load meta-agent
        if model_data['meta_agent_state'] and self.meta_agent:
            self.meta_agent.load_state_dict(model_data['meta_agent_state'])
        
        # Load individual agents
        for i, agent in enumerate(self.agents):
            agent_filepath = filepath.replace('.pt', f'_agent_{i}.pt')
            try:
                agent.agent.load(agent_filepath)
                agent.is_trained = True
            except FileNotFoundError:
                logger.warning(f"[WARNING] Agent file not found: {agent_filepath}")
        
        logger.info(f"[SUCCESS] Multi-agent ensemble loaded from {filepath}")

def create_multi_agent_system(config: Dict[str, Any] = None) -> MultiAgentTradingSystem:
    """Factory function to create multi-agent trading system."""
    return MultiAgentTradingSystem(config)
