"""
AI Trading Integration Module

This module demonstrates how to integrate the Week 2 AI trading implementations
with the PostgreSQL database, providing end-to-end workflows from model training
to signal generation and storage.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from .ai_database import get_ai_database, AITradingDatabase
from .data_structures import MarketData, TradingSignal

logger = logging.getLogger(__name__)

class AITradingIntegration:
    """
    Integration layer between AI trading modules and PostgreSQL database.
    
    Provides high-level workflows for:
    - Model registration and management
    - Training data storage and retrieval
    - Signal generation and persistence
    - Performance tracking and analytics
    """
    
    def __init__(self, database_url: str = None):
        """Initialize AI trading integration."""
        self.db = get_ai_database(database_url)
        self.registered_models = {}
        
        logger.info("[SUCCESS] AI Trading Integration initialized")
    
    # ===========================
    # MODEL LIFECYCLE MANAGEMENT
    # ===========================
    
    def register_rl_model(self, model_name: str, config: Dict[str, Any]) -> int:
        """Register reinforcement learning model."""
        model_id = self.db.register_ai_model(
            model_type='reinforcement_learning',
            model_name=model_name,
            version='1.0',
            config=config
        )
        
        self.registered_models[model_name] = {
            'id': model_id,
            'type': 'reinforcement_learning',
            'config': config
        }
        
        logger.info(f"[SUCCESS] Registered RL model: {model_name} (ID: {model_id})")
        return model_id
    
    def register_genetic_model(self, model_name: str, config: Dict[str, Any]) -> int:
        """Register genetic optimization model."""
        model_id = self.db.register_ai_model(
            model_type='genetic_optimization',
            model_name=model_name,
            version='1.0',
            config=config
        )
        
        self.registered_models[model_name] = {
            'id': model_id,
            'type': 'genetic_optimization',
            'config': config
        }
        
        logger.info(f"[SUCCESS] Registered genetic model: {model_name} (ID: {model_id})")
        return model_id
    
    def register_spectrum_model(self, model_name: str, analysis_type: str, config: Dict[str, Any]) -> int:
        """Register spectrum analysis model."""
        model_id = self.db.register_ai_model(
            model_type='sparse_spectrum',
            model_name=f"{model_name}_{analysis_type}",
            version='1.0',
            config={**config, 'analysis_type': analysis_type}
        )
        
        self.registered_models[f"{model_name}_{analysis_type}"] = {
            'id': model_id,
            'type': 'sparse_spectrum',
            'analysis_type': analysis_type,
            'config': config
        }
        
        logger.info(f"[SUCCESS] Registered spectrum model: {model_name}_{analysis_type} (ID: {model_id})")
        return model_id
    
    # ===========================
    # REINFORCEMENT LEARNING WORKFLOWS
    # ===========================
    
    def train_rl_model_with_storage(self, model_name: str, market_data: List[MarketData], 
                                   episodes: int = 100) -> Dict[str, Any]:
        """Train RL model and store episode data in database."""
        if model_name not in self.registered_models:
            raise ValueError(f"Model {model_name} not registered")
        
        model_info = self.registered_models[model_name]
        model_id = model_info['id']
        
        # Update training status
        self.db.update_model_training_status(model_id, 'TRAINING')
        
        # Simulate RL training process with database storage
        logger.info(f"[PROCESSING] Training RL model {model_name} for {episodes} episodes...")
        
        training_results = {
            'model_id': model_id,
            'total_episodes': episodes,
            'episode_results': []
        }
        
        try:
            # Simulate training episodes
            for episode in range(1, episodes + 1):
                # Simulate episode data
                episode_data = {
                    'episode_number': episode,
                    'total_reward': 0.1 + (episode * 0.01) + (0.05 * (episode % 10 - 5) / 5),
                    'episode_length': 100 + (episode % 20),
                    'average_loss': 0.1 - (episode * 0.0008),
                    'portfolio_value': 10000 + (episode * 50) + (100 * (episode % 15 - 7)),
                    'sharpe_ratio': max(0.5 + (episode * 0.005), 0.1),
                    'actions_taken': 80 + (episode % 30),
                    'exploration_rate': max(0.1, 1.0 - (episode / episodes)),
                    'training_data': {
                        'market_symbols': [data.symbol for data in market_data[:5]],
                        'data_points': len(market_data),
                        'episode_summary': f"Episode {episode} training results"
                    }
                }
                
                # Store episode in database
                episode_id = self.db.store_rl_episode(model_id, episode_data)
                
                training_results['episode_results'].append({
                    'episode_id': episode_id,
                    'episode_number': episode,
                    'total_reward': episode_data['total_reward']
                })
                
                # Log progress every 10 episodes
                if episode % 10 == 0:
                    logger.info(f"    Episode {episode}: Reward = {episode_data['total_reward']:.4f}")
            
            # Calculate final performance metrics
            final_episode = training_results['episode_results'][-1]
            performance_metrics = {
                'final_reward': final_episode['total_reward'],
                'total_episodes': episodes,
                'training_completed': True,
                'convergence_episode': episodes - 20  # Simulated convergence
            }
            
            # Update model status to trained
            self.db.update_model_training_status(model_id, 'TRAINED', performance_metrics)
            
            training_results['performance_metrics'] = performance_metrics
            logger.info(f"[SUCCESS] RL model {model_name} training completed")
            
            return training_results
            
        except Exception as e:
            # Update model status to failed
            self.db.update_model_training_status(model_id, 'FAILED', {'error': str(e)})
            logger.error(f"[ERROR] RL model training failed: {e}")
            raise
    
    def generate_rl_signals_with_storage(self, model_name: str, market_data: List[MarketData]) -> List[int]:
        """Generate RL trading signals and store in database."""
        if model_name not in self.registered_models:
            raise ValueError(f"Model {model_name} not registered")
        
        model_info = self.registered_models[model_name]
        model_id = model_info['id']
        
        logger.info(f"[PROCESSING] Generating RL signals with {model_name}...")
        
        stored_signal_ids = []
        
        # Simulate RL signal generation
        for i in range(min(3, len(market_data))):  # Generate up to 3 signals
            data_point = market_data[-(i+1)]
            
            # Simulate RL decision making
            confidence = 0.7 + (i * 0.1)
            signal_type = ["BUY", "SELL", "BUY"][i]
            
            signal = TradingSignal(
                symbol=data_point.symbol,
                signal_type=signal_type,
                confidence=Decimal(str(confidence)),
                price_target=data_point.close,
                stop_loss=data_point.close * Decimal('0.98') if signal_type == "BUY" else data_point.close * Decimal('1.02'),
                take_profit=data_point.close * Decimal('1.05') if signal_type == "BUY" else data_point.close * Decimal('0.95'),
                timestamp=data_point.timestamp,
                metadata={
                    'model_type': 'Reinforcement_Learning',
                    'agent_type': 'PPO_Advanced',
                    'market_regime': 'trending',
                    'rl_confidence': confidence
                }
            )
            
            # Store signal in database
            signal_id = self.db.store_trading_signal(signal, model_id)
            stored_signal_ids.append(signal_id)
        
        logger.info(f"[SUCCESS] Generated and stored {len(stored_signal_ids)} RL signals")
        return stored_signal_ids
    
    # ===========================
    # GENETIC OPTIMIZATION WORKFLOWS
    # ===========================
    
    def optimize_parameters_with_storage(self, model_name: str, market_data: List[MarketData],
                                       parameter_ranges: Dict[str, tuple], generations: int = 50) -> Dict[str, Any]:
        """Optimize parameters using genetic algorithm and store generation data."""
        if model_name not in self.registered_models:
            raise ValueError(f"Model {model_name} not registered")
        
        model_info = self.registered_models[model_name]
        model_id = model_info['id']
        
        # Update training status
        self.db.update_model_training_status(model_id, 'OPTIMIZING')
        
        logger.info(f"[PROCESSING] Optimizing parameters with {model_name} for {generations} generations...")
        
        optimization_results = {
            'model_id': model_id,
            'total_generations': generations,
            'generation_results': [],
            'parameter_ranges': parameter_ranges
        }
        
        try:
            # Simulate genetic algorithm optimization
            for generation in range(1, generations + 1):
                # Simulate generation data
                generation_data = {
                    'generation_number': generation,
                    'best_fitness': 0.4 + (generation * 0.01) + (0.02 * (generation % 8 - 4) / 4),
                    'average_fitness': 0.3 + (generation * 0.008),
                    'worst_fitness': 0.1 + (generation * 0.005),
                    'population_diversity': max(0.1, 0.8 - (generation / generations * 0.6)),
                    'mutation_rate': 0.1 + (0.05 * (generation % 10 - 5) / 5),
                    'crossover_rate': 0.8,
                    'best_individual': {
                        'genes': {param: ranges[0] + (ranges[1] - ranges[0]) * (generation / generations) 
                                for param, ranges in parameter_ranges.items()},
                        'fitness': 0.4 + (generation * 0.01)
                    },
                    'population_stats': {
                        'population_size': 50,
                        'elite_count': 10,
                        'convergence_metric': min(0.95, generation / generations)
                    }
                }
                
                # Store generation in database
                generation_id = self.db.store_genetic_generation(model_id, generation_data)
                
                optimization_results['generation_results'].append({
                    'generation_id': generation_id,
                    'generation_number': generation,
                    'best_fitness': generation_data['best_fitness']
                })
                
                # Log progress every 10 generations
                if generation % 10 == 0:
                    logger.info(f"    Generation {generation}: Best Fitness = {generation_data['best_fitness']:.4f}")
            
            # Get final best parameters
            final_generation = optimization_results['generation_results'][-1]
            best_parameters = {
                'sma_short': 8,
                'sma_long': 35,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            }
            
            performance_metrics = {
                'best_fitness': final_generation['best_fitness'],
                'total_generations': generations,
                'optimization_completed': True,
                'best_parameters': best_parameters
            }
            
            # Update model status
            self.db.update_model_training_status(model_id, 'OPTIMIZED', performance_metrics)
            
            optimization_results['best_parameters'] = best_parameters
            optimization_results['performance_metrics'] = performance_metrics
            
            logger.info(f"[SUCCESS] Parameter optimization completed for {model_name}")
            return optimization_results
            
        except Exception as e:
            self.db.update_model_training_status(model_id, 'FAILED', {'error': str(e)})
            logger.error(f"[ERROR] Parameter optimization failed: {e}")
            raise
    
    # ===========================
    # SPECTRUM ANALYSIS WORKFLOWS
    # ===========================
    
    def analyze_spectrum_with_storage(self, model_name: str, analysis_type: str, 
                                    market_data: List[MarketData]) -> List[int]:
        """Perform spectrum analysis and store results."""
        full_model_name = f"{model_name}_{analysis_type}"
        
        if full_model_name not in self.registered_models:
            raise ValueError(f"Model {full_model_name} not registered")
        
        model_info = self.registered_models[full_model_name]
        model_id = model_info['id']
        
        logger.info(f"[PROCESSING] Performing {analysis_type} analysis with {model_name}...")
        
        stored_analysis_ids = []
        
        # Group data by symbol
        symbol_data = {}
        for data in market_data:
            if data.symbol not in symbol_data:
                symbol_data[data.symbol] = []
            symbol_data[data.symbol].append(data)
        
        # Analyze each symbol
        for symbol, data_points in symbol_data.items():
            if len(data_points) < self.min_data_points:
        """Generate trading signals from spectrum analysis and store them."""
        full_model_name = f"{model_name}_{analysis_type}"
        
        if full_model_name not in self.registered_models:
            raise ValueError(f"Model {full_model_name} not registered")
        
        model_info = self.registered_models[full_model_name]
        model_id = model_info['id']
        
        # First perform analysis
        analysis_ids = self.analyze_spectrum_with_storage(model_name, analysis_type, market_data)
        
        logger.info(f"[PROCESSING] Generating {analysis_type} signals...")
        
        stored_signal_ids = []
        
        # Generate signals based on analysis
        unique_symbols = list(set(data.symbol for data in market_data))
        
        for i, symbol in enumerate(unique_symbols[:3]):  # Limit to 3 signals
            # Get latest data for symbol
            symbol_data = [data for data in market_data if data.symbol == symbol]
            if not symbol_data:
                continue
            
            latest_data = symbol_data[-1]
            
            # Generate signal based on analysis type
            if analysis_type == 'fourier':
                signal_type = "BUY" if i % 2 == 0 else "SELL"
                confidence = 0.78
                metadata = {
                    'model_type': 'Fourier_Analysis',
                    'dominant_frequency': 0.05,
                    'harmonic_pattern': 'ascending_triangle',
                    'spectral_confidence': confidence
                }
            elif analysis_type == 'wavelet':
                signal_type = "SELL" if i == 1 else "BUY"
                confidence = 0.83
                metadata = {
                    'model_type': 'Wavelet_Analysis',
                    'wavelet_type': 'db4',
                    'trend_component': 'medium_term_up',
                    'denoising_applied': True
                }
            else:  # compressed_sensing
                signal_type = "BUY"
                confidence = 0.91
                metadata = {
                    'model_type': 'Compressed_Sensing',
                    'anomaly_detected': True,
                    'sparsity_level': 0.875,
                    'reconstruction_error': 0.0234
                }
            
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=Decimal(str(confidence)),
                price_target=latest_data.close,
                stop_loss=latest_data.close * Decimal('0.975') if signal_type == "BUY" else latest_data.close * Decimal('1.025'),
                take_profit=latest_data.close * Decimal('1.05') if signal_type == "BUY" else latest_data.close * Decimal('0.95'),
                timestamp=latest_data.timestamp,
                metadata=metadata
            )
            
            # Store signal in database
            signal_id = self.db.store_trading_signal(signal, model_id)
            stored_signal_ids.append(signal_id)
        
        logger.info(f"[SUCCESS] Generated and stored {len(stored_signal_ids)} {analysis_type} signals")
        return stored_signal_ids
    
    # ===========================
    # ANALYTICS AND REPORTING
    # ===========================
    
    def get_model_performance_summary(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance summary for a model."""
        if model_name not in self.registered_models:
            raise ValueError(f"Model {model_name} not registered")
        
        model_info = self.registered_models[model_name]
        model_id = model_info['id']
        
        # Calculate recent performance
        today = datetime.now().date()
        self.db.calculate_model_performance(model_id, today)
        
        # Get performance history
        performance_history = self.db.get_model_performance_history(model_id, days)
        
        # Get recent signals
        recent_signals = self.db.get_active_signals(model_id=model_id, limit=50)
        
        summary = {
            'model_name': model_name,
            'model_id': model_id,
            'model_type': model_info['type'],
            'performance_history': performance_history,
            'recent_signals': recent_signals,
            'summary_stats': self._calculate_summary_stats(performance_history, recent_signals)
        }
        
        return summary
    
    def _calculate_summary_stats(self, performance_history: List[Dict], recent_signals: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from performance data."""
        if not performance_history:
            return {'error': 'No performance data available'}
        
        # Calculate aggregated metrics
        total_signals = sum(p.get('total_signals', 0) for p in performance_history)
        successful_signals = sum(p.get('successful_signals', 0) for p in performance_history)
        
        avg_win_rate = sum(p.get('win_rate', 0) for p in performance_history) / len(performance_history)
        avg_return = sum(p.get('average_return', 0) for p in performance_history) / len(performance_history)
        avg_sharpe = sum(p.get('sharpe_ratio', 0) for p in performance_history) / len(performance_history)
        
        return {
            'total_signals_period': total_signals,
            'successful_signals_period': successful_signals,
            'overall_win_rate': (successful_signals / total_signals * 100) if total_signals > 0 else 0,
            'average_win_rate': avg_win_rate,
            'average_return': avg_return,
            'average_sharpe_ratio': avg_sharpe,
            'active_signals': len([s for s in recent_signals if s.get('status') == 'ACTIVE']),
            'recent_signal_count': len(recent_signals)
        }
    
    def get_comprehensive_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for AI trading dashboard."""
        dashboard_data = {
            'registered_models': {},
            'active_signals': [],
            'recent_performance': {},
            'model_summaries': {}
        }
        
        # Get data for all registered models
        for model_name, model_info in self.registered_models.items():
            model_id = model_info['id']
            
            # Get model summary
            try:
                summary = self.get_model_performance_summary(model_name, days=7)
                dashboard_data['model_summaries'][model_name] = summary
            except Exception as e:
                logger.warning(f"Failed to get summary for {model_name}: {e}")
            
            # Get active signals for this model
            active_signals = self.db.get_active_signals(model_id=model_id, limit=10)
            dashboard_data['active_signals'].extend(active_signals)
        
        dashboard_data['registered_models'] = self.registered_models
        
        logger.info(f"[SUCCESS] Generated dashboard data for {len(self.registered_models)} models")
        return dashboard_data
