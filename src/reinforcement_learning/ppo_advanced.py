"""
Advanced Reinforcement Learning Trading Module

This module implements sophisticated RL algorithms for trading:
- PPO (Proximal Policy Optimization) for continuous action spaces
- Multi-agent trading systems with ensemble strategies
- Advanced training pipelines with custom environments
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime
import logging

# Import our core components
from ..core.data_structures import MarketData, TradingSignal, PortfolioPosition
from ..core.base_classes import BaseTradingModel
from ..core.timezone_utils import now_eastern

logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    
@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    total_timesteps: int = 100000
    eval_freq: int = 1000
    save_freq: int = 5000
    log_freq: int = 100
    n_eval_episodes: int = 10
    early_stopping_patience: int = 10

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extraction
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions normalized to [-1, 1]
        )
        
        self.actor_std = nn.Parameter(torch.ones(action_dim) * 0.1)
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution and value."""
        features = self.shared_net(state)
        
        # Actor outputs
        action_mean = self.actor_mean(features)
        action_std = self.actor_std.expand_as(action_mean)
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action_and_value(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        """Get action distribution and value for given state."""
        action_mean, action_std, value = self.forward(state)
        action_dist = Normal(action_mean, action_std)
        
        if action is None:
            action = action_dist.sample()
        
        action_logprob = action_dist.log_prob(action).sum(axis=-1)
        entropy = action_dist.entropy().sum(axis=-1)
        
        return action, action_logprob, entropy, value

class AdvancedTradingEnvironment(gym.Env):
    """Advanced trading environment with multiple timeframes and risk management."""
    
    def __init__(self, 
                 market_data: List[MarketData],
                 initial_balance: float = 100000.0,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 0.2,
                 lookback_window: int = 60):
        
        super().__init__()
        
        self.market_data = market_data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        
        # State: OHLCV + technical indicators + portfolio state
        # Action: [position_change, stop_loss_pct, take_profit_pct]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window * 5 + 10,), 
            dtype=np.float32
        )
        
        # Continuous action space: position change [-1, 1], stop loss [0, 0.1], take profit [0, 0.3]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]), 
            high=np.array([1.0, 0.1, 0.3]), 
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one trading step."""
        if self.current_step >= len(self.market_data) - 1:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        current_price = float(self.market_data[self.current_step].close)
        position_change = action[0] * self.max_position_size
        stop_loss_pct = action[1]
        take_profit_pct = action[2]
        
        # Calculate reward before action
        prev_portfolio_value = self._get_portfolio_value()
        
        # Execute trade
        reward = self._execute_trade(position_change, stop_loss_pct, take_profit_pct, current_price)
        
        # Update stop loss and take profit if in position
        if abs(self.position) > 0.001:
            self.stop_loss = current_price * (1 - stop_loss_pct) if self.position > 0 else current_price * (1 + stop_loss_pct)
            self.take_profit = current_price * (1 + take_profit_pct) if self.position > 0 else current_price * (1 - take_profit_pct)
        
        # Check stop loss and take profit
        self._check_exit_conditions(current_price)
        
        self.current_step += 1
        
        # Calculate portfolio change for reward
        new_portfolio_value = self._get_portfolio_value()
        portfolio_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Enhanced reward function
        reward = self._calculate_reward(portfolio_return, action)
        
        done = self.current_step >= len(self.market_data) - 1
        
        return self._get_observation(), reward, done, False, self._get_info()
    
    def _execute_trade(self, position_change: float, stop_loss_pct: float, take_profit_pct: float, price: float) -> float:
        """Execute trade with transaction costs."""
        if abs(position_change)  0.001:
            self.entry_price = price
            self.total_trades += 1
        
        # Deduct transaction cost
        self.balance -= cost
        
        return -cost / self.initial_balance  # Negative reward for costs
    
    def _check_exit_conditions(self, current_price: float):
        """Check and execute stop loss or take profit."""
        if abs(self.position)  0:
            if current_price = self.take_profit:
                exit_triggered = True
        
        # Short position checks
        elif self.position = self.stop_loss or current_price  0 and current_price > self.entry_price:
                self.winning_trades += 1
            elif self.position  float:
        """Calculate sophisticated reward function."""
        # Base return reward
        reward = portfolio_return * 100  # Scale up
        
        # Risk-adjusted reward (penalize volatility)
        if hasattr(self, 'return_history'):
            if len(self.return_history) > 10:
                volatility = np.std(self.return_history[-10:])
                reward -= volatility * 10  # Penalize high volatility
        else:
            self.return_history = []
        
        self.return_history.append(portfolio_return)
        
        # Penalty for excessive trading
        position_change = abs(action[0])
        if position_change > 0.1:  # Large position changes
            reward -= position_change * 0.5
        
        # Bonus for profitable trades
        if self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades
            if win_rate > 0.6:  # Good win rate
                reward += 0.1
        
        # Penalty for large drawdowns
        current_value = self._get_portfolio_value()
        max_value = getattr(self, 'max_portfolio_value', self.initial_balance)
        self.max_portfolio_value = max(max_value, current_value)
        
        drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        if drawdown > 0.1:  # More than 10% drawdown
            reward -= drawdown * 2
        
        return reward
    
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        current_price = float(self.market_data[self.current_step].close)
        position_value = self.position * current_price * self.balance
        return self.balance + position_value
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        # Market data features (OHLCV for lookback window)
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        market_features = []
        for i in range(start_idx, end_idx):
            if i  0 else 0.0,  # Unrealized P&L
            self.stop_loss / current_price if self.stop_loss > 0 else 0.0,  # Normalized stop loss
            self.take_profit / current_price if self.take_profit > 0 else 0.0,  # Normalized take profit
            self.total_trades / 100.0,  # Normalized trade count
            self.winning_trades / max(1, self.total_trades),  # Win rate
            len(getattr(self, 'return_history', [])) / 1000.0,  # Time factor
            np.std(getattr(self, 'return_history', [0.0])[-10:]) if len(getattr(self, 'return_history', [])) > 1 else 0.0  # Recent volatility
        ]
        
        observation = np.array(market_features + portfolio_features, dtype=np.float32)
        return observation
    
    def _get_info(self) -> Dict:
        """Get environment info."""
        return {
            'portfolio_value': self._get_portfolio_value(),
            'position': self.position,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'step': self.current_step
        }

class PPOAgent:
    """Proximal Policy Optimization agent for trading."""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 config: PPOConfig = None,
                 device: str = None):
        
        self.config = config or PPOConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize network
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # Training buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        logger.info(f"[SUCCESS] PPO Agent initialized on {self.device}")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Get action from current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(state_tensor)
        
        if deterministic:
            # Use mean action for deterministic behavior
            action_mean, _, _ = self.network(state_tensor)
            action = action_mean
        
        return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        value: float, log_prob: float, done: bool):
        """Store transition in buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO."""
        if len(self.states) == 0:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.BoolTensor(self.dones).to(self.device)
        
        # Calculate advantages using GAE
        advantages = self._calculate_gae(rewards, values, dones)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(self.config.n_epochs):
            # Get current policy outputs
            _, new_log_probs, entropy, new_values = self.network.get_action_and_value(states, actions)
            
            # Calculate ratios
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Policy loss with clipping
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(new_values.squeeze(), returns)
            
            # Entropy loss for exploration
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = (policy_loss + 
                         self.config.value_coef * value_loss + 
                         self.config.entropy_coef * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
        
        # Clear buffers
        self.clear_buffer()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item()
        }
    
    def _calculate_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def clear_buffer(self):
        """Clear experience buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
        logger.info(f"[SUCCESS] PPO model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        logger.info(f"[SUCCESS] PPO model loaded from {filepath}")

class AdvancedRLTradingModel(BaseTradingModel):
    """Advanced reinforcement learning trading model using PPO."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ppo_config = PPOConfig(**self.config.get('ppo', {}))
        self.training_config = TrainingConfig(**self.config.get('training', {}))
        
        self.agent = None
        self.env = None
        self.is_trained = False
        
        logger.info("[SUCCESS] Advanced RL Trading Model initialized")
    
    def prepare_data(self, market_data: List[MarketData]) -> None:
        """Prepare trading environment with market data."""
        self.env = AdvancedTradingEnvironment(
            market_data=market_data,
            initial_balance=self.config.get('initial_balance', 100000.0),
            transaction_cost=self.config.get('transaction_cost', 0.001),
            max_position_size=self.config.get('max_position_size', 0.2),
            lookback_window=self.config.get('lookback_window', 60)
        )
        
        # Initialize agent
        self.agent = PPOAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            config=self.ppo_config
        )
        
        logger.info(f"[SUCCESS] Environment prepared with {len(market_data)} data points")
    
    def train(self, validation_data: Optional[List[MarketData]] = None) -> Dict[str, Any]:
        """Train the PPO agent."""
        if self.env is None or self.agent is None:
            raise ValueError("Must call prepare_data() before training")
        
        training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rates': [],
            'portfolio_values': [],
            'policy_losses': [],
            'value_losses': []
        }
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.training_config.total_timesteps):
            # Get action
            action, log_prob, value = self.agent.get_action(state)
            
            # Take step
            next_state, reward, done, _, info = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                # Log episode metrics
                training_metrics['episode_rewards'].append(episode_reward)
                training_metrics['episode_lengths'].append(episode_length)
                training_metrics['win_rates'].append(info['win_rate'])
                training_metrics['portfolio_values'].append(info['portfolio_value'])
                
                # Reset environment
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                
                # Update agent
                update_metrics = self.agent.update()
                if update_metrics:
                    training_metrics['policy_losses'].append(update_metrics.get('policy_loss', 0))
                    training_metrics['value_losses'].append(update_metrics.get('value_loss', 0))
            
            # Logging
            if step % self.training_config.log_freq == 0 and len(training_metrics['episode_rewards']) > 0:
                avg_reward = np.mean(training_metrics['episode_rewards'][-10:])
                avg_win_rate = np.mean(training_metrics['win_rates'][-10:])
                logger.info(f"[TRAINING] Step {step}: Avg Reward={avg_reward:.3f}, Win Rate={avg_win_rate:.3f}")
            
            # Save model
            if step % self.training_config.save_freq == 0 and step > 0:
                self.save_model(f"ppo_model_step_{step}.pt")
        
        self.is_trained = True
        logger.info("[SUCCESS] PPO training completed")
        
        return training_metrics
    
    def predict(self, market_data: List[MarketData]) -> List[TradingSignal]:
        """Generate trading signals using trained agent."""
        if not self.is_trained or self.agent is None:
            raise ValueError("Model must be trained before prediction")
        
        # Create temporary environment for prediction
        pred_env = AdvancedTradingEnvironment(
            market_data=market_data,
            initial_balance=self.config.get('initial_balance', 100000.0),
            transaction_cost=0.0,  # No costs for prediction
            max_position_size=self.config.get('max_position_size', 0.2),
            lookback_window=self.config.get('lookback_window', 60)
        )
        
        signals = []
        state, _ = pred_env.reset()
        
        for i in range(len(market_data) - pred_env.lookback_window):
            action, _, _ = self.agent.get_action(state, deterministic=True)
            
            # Convert action to trading signal
            position_change = action[0]
            confidence = min(abs(position_change) * 2, 1.0)  # Scale confidence
            
            signal_type = "BUY" if position_change > 0.1 else "SELL" if position_change  None:
        """Save trained model."""
        if self.agent is None:
            raise ValueError("No model to save")
        
        self.agent.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        if self.agent is None:
            raise ValueError("Must initialize agent before loading")
        
        self.agent.load(filepath)
        self.is_trained = True

def create_advanced_rl_model(config: Dict[str, Any] = None) -> AdvancedRLTradingModel:
    """Factory function to create advanced RL model."""
    return AdvancedRLTradingModel(config)
