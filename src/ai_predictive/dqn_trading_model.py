"""
AI Trading Advancements - Predictive Analytics with Reinforcement Learning

This module implements DQN (Deep Q-Network) and other RL algorithms for trading.
Designed as a modular library component that can be integrated with existing systems.

Author: AI Trading Development Team
Date: August 31, 2025
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone
import logging
import pickle
from pathlib import Path

# Import core components
from ..core import (
    MarketData, TradingSignal, ModelMetrics, SignalType,
    TimeFrame, BaseAIModel, get_config
)

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for reinforcement learning.
    Implements OpenAI Gym interface for compatibility with stable-baselines3.
    """
    
    def __init__(
        self,
        market_data: List[MarketData],
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        lookback_window: int = 20
    ):
        """
        Initialize trading environment.
        
        Args:
            market_data: Historical market data for training
            initial_balance: Starting portfolio value
            transaction_cost: Transaction cost as percentage
            lookback_window: Number of historical periods to include in state
        """
        super().__init__()
        
        self.market_data = market_data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # Convert market data to numpy arrays for faster processing
        self.prices = np.array([float(data.close_price) for data in market_data])
        self.volumes = np.array([data.volume for data in market_data])
        self.highs = np.array([float(data.high_price) for data in market_data])
        self.lows = np.array([float(data.low_price) for data in market_data])
        self.opens = np.array([float(data.open_price) for data in market_data])
        
        # Calculate technical indicators
        self._calculate_features()
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Number of shares held
        self.total_trades = 0
        self.winning_trades = 0
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price features + position info + account info
        feature_count = len(self.features[0]) if self.features.size > 0 else 10
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_count + 3,),  # features + position + balance + pnl
            dtype=np.float32
        )
        
        logger.info(f"[SUCCESS] Trading environment initialized")
        logger.info(f"[DATA] Market data points: {len(market_data)}")
        logger.info(f"[DATA] Observation space: {self.observation_space.shape}")
    
    def _calculate_features(self) -> None:
        """Calculate technical indicators and features."""
        if len(self.prices)  np.ndarray:
        """Calculate Simple Moving Average."""
        sma = np.convolve(data, np.ones(window)/window, mode='valid')
        # Pad the beginning with the first calculated value
        padding = np.full(window-1, sma[0])
        return np.concatenate([padding, sma])
    
    def _rsi(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas  Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.current_step >= len(self.market_data) - 1:
            return self._get_observation(), 0.0, True, False, {}
        
        # Get current price
        current_price = self.prices[self.current_step]
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.market_data) - 1
        
        # Calculate portfolio value
        portfolio_value = self.balance + self.position * self.prices[self.current_step]
        
        info = {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1)
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute trading action and calculate reward."""
        reward = 0.0
        
        if action == 1:  # Buy
            if self.balance > current_price * (1 + self.transaction_cost):
                shares_to_buy = self.balance / (current_price * (1 + self.transaction_cost))
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                
                self.position += shares_to_buy
                self.balance -= cost
                self.total_trades += 1
                
                # Small negative reward for transaction cost
                reward = -self.transaction_cost
        
        elif action == 2:  # Sell
            if self.position > 0:
                proceeds = self.position * current_price * (1 - self.transaction_cost)
                
                # Calculate profit/loss
                cost_basis = (self.initial_balance - self.balance) / max(self.position, 1e-8)
                profit_per_share = current_price - cost_basis
                
                if profit_per_share > 0:
                    self.winning_trades += 1
                    reward = profit_per_share / cost_basis  # Return as reward
                else:
                    reward = profit_per_share / cost_basis  # Loss as negative reward
                
                self.balance += proceeds
                self.position = 0.0
                self.total_trades += 1
        
        # Add small reward for holding profitable positions
        if self.position > 0:
            unrealized_pnl = self.position * current_price - (self.initial_balance - self.balance)
            reward += unrealized_pnl / self.initial_balance * 0.01  # Small holding reward
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        if self.current_step  bool:
        if self.n_calls % self.check_freq == 0:
            # Log training progress
            logger.info(f"[PROCESSING] Training step: {self.n_calls}")
            
            # Get recent rewards
            if hasattr(self.locals.get('infos', [{}])[0], 'portfolio_value'):
                portfolio_value = self.locals['infos'][0].get('portfolio_value', 0)
                logger.info(f"[DATA] Portfolio value: ${portfolio_value:.2f}")
        
        return True


class DQNTradingModel(BaseAIModel):
    """
    Deep Q-Network trading model implementation.
    Extends BaseAIModel for standardized interface.
    """
    
    def __init__(
        self,
        name: str = "DQN_Trader",
        version: str = "1.0.0",
        learning_rate: float = 0.0001,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_final_eps: float = 0.02
    ):
        """Initialize DQN trading model."""
        super().__init__(name, version)
        
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        
        self.model = None
        self.env = None
        
        logger.info(f"[SUCCESS] DQN Trading Model initialized: {name}")
    
    async def train(
        self,
        training_data: List[MarketData],
        validation_data: Optional[List[MarketData]] = None,
        **kwargs
    ) -> ModelMetrics:
        """Train the DQN model with market data."""
        self.logger.info("[STARTING] DQN training process")
        
        # Create training environment
        self.env = TradingEnvironment(
            market_data=training_data,
            initial_balance=kwargs.get('initial_balance', 100000.0),
            transaction_cost=kwargs.get('transaction_cost', 0.001),
            lookback_window=kwargs.get('lookback_window', 20)
        )
        
        # Initialize DQN model
        self.model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            target_update_interval=self.target_update_interval,
            exploration_fraction=self.exploration_fraction,
            exploration_final_eps=self.exploration_final_eps,
            verbose=1,
            device="auto"
        )
        
        # Training parameters
        total_timesteps = kwargs.get('total_timesteps', 50000)
        
        # Create callback for monitoring
        callback = TradingCallback(check_freq=1000)
        
        # Train the model
        self.logger.info(f"[PROCESSING] Training for {total_timesteps} timesteps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        self.is_trained = True
        
        # Evaluate on training data
        metrics = await self._evaluate_model(training_data)
        self.metrics = metrics
        
        self.logger.info("[SUCCESS] DQN training completed")
        return metrics
    
    async def predict(self, data: List[MarketData]) -> List[TradingSignal]:
        """Generate trading signals from market data."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        signals = []
        
        # Create environment for prediction
        pred_env = TradingEnvironment(
            market_data=data,
            initial_balance=100000.0,
            lookback_window=20
        )
        
        obs, _ = pred_env.reset()
        
        for i, market_point in enumerate(data[20:], 20):  # Skip lookback window
            # Get model prediction
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Convert action to trading signal
            signal_type = SignalType.HOLD
            confidence = Decimal('0.5')
            
            if action == 1:  # Buy
                signal_type = SignalType.BUY
                confidence = Decimal('0.8')
            elif action == 2:  # Sell
                signal_type = SignalType.SELL
                confidence = Decimal('0.8')
            
            signal = TradingSignal(
                symbol=market_point.symbol,
                timestamp=market_point.timestamp,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=market_point.close_price,
                model_version=self.version,
                reasoning=f"DQN action: {action}"
            )
            
            signals.append(signal)
            
            # Step environment
            obs, _, done, _, _ = pred_env.step(action)
            if done:
                break
        
        self.logger.info(f"[SUCCESS] Generated {len(signals)} trading signals")
        return signals
    
    async def evaluate(self, test_data: List[MarketData]) -> ModelMetrics:
        """Evaluate model performance on test data."""
        return await self._evaluate_model(test_data)
    
    async def _evaluate_model(self, data: List[MarketData]) -> ModelMetrics:
        """Internal method to evaluate model performance."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Create evaluation environment
        eval_env = TradingEnvironment(
            market_data=data,
            initial_balance=100000.0,
            lookback_window=20
        )
        
        # Run evaluation episode
        obs, _ = eval_env.reset()
        total_reward = 0.0
        steps = 0
        
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Calculate metrics
        final_portfolio = info.get('portfolio_value', 100000.0)
        total_return = (final_portfolio - 100000.0) / 100000.0
        win_rate = info.get('win_rate', 0.0)
        
        metrics = ModelMetrics(
            model_name=self.name,
            model_version=self.version,
            training_date=datetime.now(timezone.utc),
            accuracy=Decimal(str(win_rate)),
            precision=Decimal(str(win_rate)),  # Simplified for now
            recall=Decimal(str(win_rate)),     # Simplified for now
            f1_score=Decimal(str(win_rate)),   # Simplified for now
            validation_loss=Decimal(str(-total_reward)),
            training_samples=len(data) - 20,
            validation_samples=0,
            test_samples=len(data) - 20,
            hyperparameters={
                'learning_rate': self.learning_rate,
                'buffer_size': self.buffer_size,
                'batch_size': self.batch_size,
                'total_return': total_return,
                'final_portfolio_value': final_portfolio
            }
        )
        
        return metrics
    
    async def save_model(self, path: Path) -> bool:
        """Save trained model to disk."""
        try:
            if self.model is None:
                self.logger.error("[ERROR] No model to save")
                return False
            
            # Save the stable-baselines3 model
            model_path = path / f"{self.name}_{self.version}.zip"
            self.model.save(str(model_path))
            
            # Save additional metadata
            metadata = {
                'name': self.name,
                'version': self.version,
                'is_trained': self.is_trained,
                'metrics': self.metrics.to_dict() if self.metrics else None,
                'hyperparameters': {
                    'learning_rate': self.learning_rate,
                    'buffer_size': self.buffer_size,
                    'batch_size': self.batch_size
                }
            }
            
            metadata_path = path / f"{self.name}_{self.version}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            self.model_path = model_path
            self.logger.info(f"[SUCCESS] Model saved to {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to save model: {e}")
            return False
    
    async def load_model(self, path: Path) -> bool:
        """Load trained model from disk."""
        try:
            model_path = path / f"{self.name}_{self.version}.zip"
            metadata_path = path / f"{self.name}_{self.version}_metadata.pkl"
            
            if not model_path.exists():
                self.logger.error(f"[ERROR] Model file not found: {model_path}")
                return False
            
            # Load the model
            self.model = DQN.load(str(model_path))
            
            # Load metadata if available
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    
                if metadata.get('metrics'):
                    # Reconstruct metrics object
                    metrics_data = metadata['metrics']
                    self.metrics = ModelMetrics(
                        model_name=metrics_data['model_name'],
                        model_version=metrics_data['model_version'],
                        training_date=datetime.fromisoformat(metrics_data['training_date']),
                        accuracy=Decimal(metrics_data['accuracy']),
                        precision=Decimal(metrics_data['precision']),
                        recall=Decimal(metrics_data['recall']),
                        f1_score=Decimal(metrics_data['f1_score']),
                        validation_loss=Decimal(metrics_data['validation_loss']),
                        training_samples=metrics_data['training_samples'],
                        validation_samples=metrics_data['validation_samples'],
                        test_samples=metrics_data['test_samples'],
                        hyperparameters=metrics_data['hyperparameters']
                    )
            
            self.is_trained = True
            self.model_path = model_path
            self.logger.info(f"[SUCCESS] Model loaded from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load model: {e}")
            return False


# Export public interfaces
__all__ = [
    'TradingEnvironment',
    'DQNTradingModel',
    'TradingCallback'
]


if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    from datetime import timedelta
    
    async def test_dqn_model():
        print("\n=== DQN Trading Model Test ===")
        
        # Create synthetic market data for testing
        base_price = 100.0
        data_points = 1000
        
        market_data = []
        for i in range(data_points):
            # Simple random walk with trend
            price_change = np.random.normal(0.0, 1.0) + 0.01  # Slight upward trend
            base_price *= (1 + price_change / 100)
            
            market_data.append(MarketData(
                symbol="TEST",
                timestamp=datetime.now(timezone.utc) + timedelta(days=i),
                open_price=Decimal(str(base_price * 0.999)),
                high_price=Decimal(str(base_price * 1.005)),
                low_price=Decimal(str(base_price * 0.995)),
                close_price=Decimal(str(base_price)),
                volume=int(np.random.uniform(100000, 1000000)),
                timeframe=TimeFrame.DAY_1
            ))
        
        print(f"[SUCCESS] Created {len(market_data)} synthetic data points")
        
        # Initialize and train DQN model
        model = DQNTradingModel(
            name="TestDQN",
            learning_rate=0.001,
            buffer_size=10000
        )
        
        # Split data
        train_data = market_data[:800]
        test_data = market_data[800:]
        
        # Train model
        print("[PROCESSING] Training DQN model...")
        metrics = await model.train(
            training_data=train_data,
            total_timesteps=5000  # Reduced for testing
        )
        
        print(f"[SUCCESS] Training completed")
        print(f"[DATA] Final portfolio value: ${metrics.hyperparameters['final_portfolio_value']:.2f}")
        print(f"[DATA] Total return: {metrics.hyperparameters['total_return']:.2%}")
        
        # Generate predictions
        print("[PROCESSING] Generating predictions...")
        signals = await model.predict(test_data)
        
        buy_signals = sum(1 for s in signals if s.signal_type == SignalType.BUY)
        sell_signals = sum(1 for s in signals if s.signal_type == SignalType.SELL)
        hold_signals = len(signals) - buy_signals - sell_signals
        
        print(f"[SUCCESS] Generated {len(signals)} signals")
        print(f"[DATA] Buy: {buy_signals}, Sell: {sell_signals}, Hold: {hold_signals}")
        
        # Test save/load
        config = get_config()
        model_path = Path(config.ai_models.model_save_path)
        
        print("[PROCESSING] Testing model save/load...")
        save_success = await model.save_model(model_path)
        print(f"[DATA] Save successful: {save_success}")
        
        # Create new model instance and load
        new_model = DQNTradingModel(name="TestDQN")
        load_success = await new_model.load_model(model_path)
        print(f"[DATA] Load successful: {load_success}")
        
        print("\n[SUCCESS] DQN model test completed")
    
    # Run the test
    asyncio.run(test_dqn_model())
