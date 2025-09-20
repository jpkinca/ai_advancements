"""
AI Trading Integration Module

This module demonstrates how to integrate AI trading models with the PostgreSQL database.
It provides examples of storing models, training sessions, signals, and performance data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal

from src.core.data_structures import MarketData, TradingSignal
from src.database import (
    AITradingDatabase, AIModelManager, TrainingSessionManager,
    SignalManager, FeatureManager, PerformanceManager
)

logger = logging.getLogger(__name__)

class AITradingIntegrator:
    """Integration layer between AI models and database."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize the integrator with database connection."""
        self.db = AITradingDatabase(database_url)
        self.model_manager = AIModelManager(self.db)
        self.training_manager = TrainingSessionManager(self.db)
        self.signal_manager = SignalManager(self.db)
        self.feature_manager = FeatureManager(self.db)
        self.performance_manager = PerformanceManager(self.db)
        
        logger.info("[SUCCESS] AI Trading Integrator initialized")
    
    async def connect(self):
        """Connect to database."""
        await self.db.connect()
    
    async def disconnect(self):
        """Disconnect from database."""
        await self.db.disconnect()
    
    async def register_rl_model(self, model_name: str, config: Dict[str, Any]) -> str:
        """Register a reinforcement learning model."""
        return await self.model_manager.register_model(
            model_name=model_name,
            model_type="reinforcement_learning",
            model_subtype=config.get('algorithm', 'ppo'),
            configuration=config,
            description=f"Reinforcement Learning model using {config.get('algorithm', 'PPO')}"
        )
    
    async def register_genetic_model(self, model_name: str, config: Dict[str, Any]) -> str:
        """Register a genetic optimization model."""
        return await self.model_manager.register_model(
            model_name=model_name,
            model_type="genetic_optimization",
            model_subtype=config.get('optimization_type', 'parameter'),
            configuration=config,
            description=f"Genetic optimization for {config.get('optimization_type', 'parameter')} optimization"
        )
    
    async def register_spectrum_model(self, model_name: str, config: Dict[str, Any]) -> str:
        """Register a sparse spectrum analysis model."""
        analysis_type = "fourier"
        if 'wavelet' in config:
            analysis_type = "wavelet"
        elif 'compressed_sensing' in config:
            analysis_type = "compressed_sensing"
        
        return await self.model_manager.register_model(
            model_name=model_name,
            model_type="sparse_spectrum",
            model_subtype=analysis_type,
            configuration=config,
            description=f"Sparse spectrum analysis using {analysis_type}"
        )
    
    async def start_model_training(self, model_id: str, training_name: str,
                                 training_data: List[MarketData],
                                 validation_data: Optional[List[MarketData]] = None) -> str:
        """Start a training session for a model."""
        # Determine training period from data
        training_start = min(data.timestamp for data in training_data)
        training_end = max(data.timestamp for data in training_data)
        training_period = (training_start, training_end)
        
        validation_period = None
        if validation_data:
            validation_start = min(data.timestamp for data in validation_data)
            validation_end = max(data.timestamp for data in validation_data)
            validation_period = (validation_start, validation_end)
        
        return await self.training_manager.start_training_session(
            model_id=model_id,
            session_name=training_name,
            training_period=training_period,
            validation_period=validation_period
        )
    
    async def record_training_progress(self, session_id: str, epoch: int, 
                                     metrics: Dict[str, Any]) -> None:
        """Record training progress during model training."""
        training_metrics = {
            'current_epoch': epoch,
            'latest_metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.training_manager.update_training_metrics(session_id, training_metrics)
    
    async def complete_training(self, session_id: str, final_metrics: Dict[str, Any]) -> None:
        """Complete a training session with final results."""
        await self.training_manager.complete_training_session(
            session_id=session_id,
            final_performance=final_metrics
        )
    
    async def store_ai_signals(self, model_id: str, signals: List[TradingSignal]) -> List[str]:
        """Store AI-generated trading signals."""
        if not signals:
            return []
        
        if len(signals) == 1:
            signal_id = await self.signal_manager.store_signal(signals[0], model_id)
            return [signal_id]
        else:
            return await self.signal_manager.store_signals_batch(signals, model_id)
    
    async def get_recent_signals(self, model_id: Optional[str] = None,
                               symbol: Optional[str] = None,
                               hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent AI signals with optional filtering."""
        return await self.signal_manager.get_recent_signals(model_id, symbol, hours)
    
    async def store_extracted_features(self, symbol: str, timestamp: datetime,
                                     features: Dict[str, Any], feature_type: str) -> None:
        """Store extracted features for ML models."""
        await self.feature_manager.store_features(
            symbol=symbol,
            timestamp=timestamp,
            feature_set_name=feature_type,
            features=features,
            extraction_method="ai_model"
        )
    
    async def get_features_for_training(self, symbol: str, feature_type: str,
                                      start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get features for model training."""
        return await self.feature_manager.get_features(symbol, feature_type, start_time, end_time)
    
    async def record_model_performance(self, model_id: str, 
                                     evaluation_period: tuple,
                                     performance_metrics: Dict[str, Any]) -> None:
        """Record model performance metrics."""
        await self.performance_manager.record_performance(
            model_id=model_id,
            evaluation_period=evaluation_period,
            metrics=performance_metrics
        )
    
    async def get_model_performance_history(self, model_id: str,
                                          days_back: int = 30) -> List[Dict[str, Any]]:
        """Get model performance history."""
        return await self.performance_manager.get_model_performance_history(model_id, days_back)
    
    async def update_model_summary_performance(self, model_id: str,
                                             latest_metrics: Dict[str, Any]) -> None:
        """Update the summary performance metrics in the model registry."""
        await self.model_manager.update_model_performance(model_id, latest_metrics)

class ModelPerformanceTracker:
    """Track and analyze AI model performance over time."""
    
    def __init__(self, integrator: AITradingIntegrator):
        self.integrator = integrator
    
    async def evaluate_signal_performance(self, model_id: str, 
                                        evaluation_period_days: int = 7) -> Dict[str, Any]:
        """Evaluate signal performance for a model over a period."""
        # Get signals from the evaluation period
        end_time = datetime.now()
        start_time = end_time - timedelta(days=evaluation_period_days)
        
        # This would typically involve:
        # 1. Getting signals from the period
        # 2. Getting actual market data for those signals
        # 3. Calculating performance metrics
        
        # For now, return simulated metrics
        performance_metrics = {
            'total_signals': 45,
            'successful_signals': 32,
            'signal_accuracy': 0.711,  # 71.1%
            'avg_confidence': 0.762,
            'sharpe_ratio': 1.423,
            'max_drawdown': 0.087,  # 8.7%
            'total_return': 0.134,  # 13.4%
            'volatility': 0.156,
            'win_rate': 0.689,
            'avg_holding_period_hours': 18.5,
            'risk_adjusted_return': 0.198
        }
        
        # Store the performance metrics
        await self.integrator.record_model_performance(
            model_id=model_id,
            evaluation_period=(start_time, end_time),
            performance_metrics=performance_metrics
        )
        
        # Update model summary
        await self.integrator.update_model_summary_performance(model_id, {
            'latest_sharpe_ratio': performance_metrics['sharpe_ratio'],
            'latest_accuracy': performance_metrics['signal_accuracy'],
            'latest_evaluation': end_time.isoformat()
        })
        
        logger.info(f"[SUCCESS] Performance evaluation completed for model: {model_id}")
        return performance_metrics
    
    async def compare_model_performance(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple models."""
        comparison_results = {}
        
        for model_id in model_ids:
            performance_history = await self.integrator.get_model_performance_history(model_id)
            
            if performance_history:
                latest_performance = performance_history[0]
                comparison_results[model_id] = {
                    'latest_sharpe_ratio': latest_performance.get('sharpe_ratio', 0),
                    'latest_accuracy': latest_performance.get('signal_accuracy', 0),
                    'latest_return': latest_performance.get('total_return', 0),
                    'performance_records': len(performance_history)
                }
        
        # Rank models by Sharpe ratio
        ranked_models = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['latest_sharpe_ratio'],
            reverse=True
        )
        
        comparison_summary = {
            'model_count': len(model_ids),
            'ranking': ranked_models,
            'best_performer': ranked_models[0] if ranked_models else None,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"[SUCCESS] Model comparison completed for {len(model_ids)} models")
        return comparison_summary

# Factory function for easy integration
def create_ai_trading_integrator(database_url: Optional[str] = None) -> AITradingIntegrator:
    """Factory function to create AI trading integrator."""
    return AITradingIntegrator(database_url)
