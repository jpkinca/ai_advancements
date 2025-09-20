#!/usr/bin/env python3
"""
Week 2 AI Models Enhanced with Level II Data

Enhanced versions of Week 2 AI models using Level II market data:
- PPO Trader with order book features
- Genetic Algorithm with execution optimization
- Spectrum Analysis with microstructure patterns

Author: GitHub Copilot
Date: August 31, 2025
"""

import sys
import os
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import json

# Add the TradeAppComponents_fresh path for imports
sys.path.append(r'c:\Users\nzcon\VSPython\TradeAppComponents_fresh')

# Import Week 2 AI models
from ai_predictive.predictive_rl import PPOTrader
from adaptive_genetic.genetic_optimizer import GeneticTradingOptimizer
from sparse_spectrum.fourier_spectrum_trader import FourierSpectrumTrader

# Import data integration
from level_ii_data_integration import LevelIIDataCollector
from modules.database.railway_db_manager import RailwayPostgreSQLManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LevelIIEnhancedPPOTrader(PPOTrader):
    """
    PPO Trader enhanced with Level II order book features
    
    Adds market microstructure features to the state space:
    - Order imbalance signals
    - Liquidity conditions
    - Institutional flow detection
    - Execution quality metrics
    """
    
    def __init__(self, config: Dict, level_ii_collector: LevelIIDataCollector):
        super().__init__(config)
        self.level_ii_collector = level_ii_collector
        self.db_manager = RailwayPostgreSQLManager()
        
        # Enhanced state space size
        self.original_state_size = self.state_size
        self.level_ii_features = 8  # Number of Level II features
        self.state_size = self.original_state_size + self.level_ii_features
        
        logger.info(f"[ENHANCED] PPO state space expanded: {self.original_state_size} -> {self.state_size}")
    
    def get_enhanced_market_state(self, symbol: str) -> np.ndarray:
        """Get enhanced market state with Level II features"""
        # Get base market state
        base_state = super().get_market_state(symbol)
        
        # Get Level II features
        level_ii_features = self._get_level_ii_features(symbol)
        
        # Combine states
        enhanced_state = np.concatenate([base_state, level_ii_features])
        
        return enhanced_state
    
    def _get_level_ii_features(self, symbol: str) -> np.ndarray:
        """Extract Level II features for AI model"""
        try:
            # Get recent Level II data
            features = self.level_ii_collector.get_ai_model_features(symbol, lookback_minutes=2)
            
            if not features:
                # Return zeros if no data available
                return np.zeros(self.level_ii_features)
            
            # Normalize features for neural network
            level_ii_state = np.array([
                self._normalize_imbalance(features.get('avg_order_imbalance', 0)),
                self._normalize_volatility(features.get('imbalance_volatility', 0)),
                self._normalize_liquidity(features.get('avg_liquidity', 0)),
                self._normalize_spread(features.get('avg_spread_bps', 0)),
                self._normalize_momentum(features.get('avg_momentum', 0)),
                self._normalize_institutional(features.get('avg_institutional_flow', 0)),
                self._normalize_sample_quality(features.get('sample_count', 0)),
                self._calculate_execution_quality(features)
            ])
            
            return level_ii_state
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get Level II features for {symbol}: {e}")
            return np.zeros(self.level_ii_features)
    
    def _normalize_imbalance(self, imbalance: float) -> float:
        """Normalize order imbalance to [-1, 1]"""
        return np.clip(imbalance, -1.0, 1.0)
    
    def _normalize_volatility(self, volatility: float) -> float:
        """Normalize imbalance volatility to [0, 1]"""
        return np.clip(volatility / 0.5, 0.0, 1.0)  # Cap at 0.5 imbalance volatility
    
    def _normalize_liquidity(self, liquidity: float) -> float:
        """Normalize liquidity score to [0, 1]"""
        return np.clip(liquidity / 100000, 0.0, 1.0)  # Cap at 100k liquidity score
    
    def _normalize_spread(self, spread_bps: float) -> float:
        """Normalize spread in basis points to [0, 1]"""
        return np.clip(spread_bps / 50, 0.0, 1.0)  # Cap at 50 bps
    
    def _normalize_momentum(self, momentum: float) -> float:
        """Normalize momentum signal to [-1, 1]"""
        return np.clip(momentum * 10, -1.0, 1.0)
    
    def _normalize_institutional(self, institutional_flow: float) -> float:
        """Normalize institutional flow ratio to [0, 1]"""
        return np.clip(institutional_flow, 0.0, 1.0)
    
    def _normalize_sample_quality(self, sample_count: int) -> float:
        """Normalize sample count quality to [0, 1]"""
        return np.clip(sample_count / 120, 0.0, 1.0)  # 120 samples = 2 minutes
    
    def _calculate_execution_quality(self, features: Dict) -> float:
        """Calculate execution quality score from Level II data"""
        try:
            liquidity = features.get('avg_liquidity', 0)
            spread = features.get('avg_spread_bps', 50)
            institutional_flow = features.get('avg_institutional_flow', 0)
            
            # Higher liquidity + lower spread + institutional presence = better execution
            quality_score = (
                (liquidity / 50000) * 0.4 +  # Liquidity component
                (1 - spread / 50) * 0.4 +     # Spread component (inverted)
                institutional_flow * 0.2       # Institutional component
            )
            
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception:
            return 0.5  # Neutral quality
    
    def make_enhanced_trading_decision(self, symbol: str) -> Dict[str, Any]:
        """Make trading decision with Level II intelligence"""
        # Get enhanced state
        enhanced_state = self.get_enhanced_market_state(symbol)
        
        # Get base decision
        base_decision = super().predict_action(enhanced_state)
        
        # Get Level II context
        level_ii_features = self.level_ii_collector.get_ai_model_features(symbol)
        
        # Enhance decision with Level II insights
        enhanced_decision = self._apply_level_ii_filter(base_decision, level_ii_features, symbol)
        
        return enhanced_decision
    
    def _apply_level_ii_filter(self, base_decision: Dict, level_ii_features: Dict, symbol: str) -> Dict:
        """Apply Level II filters to trading decision"""
        enhanced_decision = base_decision.copy()
        
        if not level_ii_features:
            return enhanced_decision
        
        # Extract Level II metrics
        order_imbalance = level_ii_features.get('avg_order_imbalance', 0)
        liquidity = level_ii_features.get('avg_liquidity', 0)
        spread_bps = level_ii_features.get('avg_spread_bps', 0)
        momentum = level_ii_features.get('avg_momentum', 0)
        institutional_flow = level_ii_features.get('avg_institutional_flow', 0)
        
        # Liquidity filter - reduce position size in low liquidity
        if liquidity  20:  # Wide spread threshold
            enhanced_decision['position_size'] *= 0.7
            enhanced_decision['confidence'] *= 0.9
            enhanced_decision['risk_adjustment'] = 'reduced_size_wide_spread'
        
        # Imbalance confirmation - boost confidence when aligned
        if enhanced_decision['action'] == 'BUY' and order_imbalance > 0.2:
            enhanced_decision['confidence'] *= 1.2
            enhanced_decision['signal_confirmation'] = 'bullish_imbalance_confirmation'
        elif enhanced_decision['action'] == 'SELL' and order_imbalance  0.1:
            enhanced_decision['position_size'] *= 0.8
            enhanced_decision['momentum_warning'] = 'selling_against_momentum'
        
        # Institutional flow consideration
        if institutional_flow > 0.3:  # High institutional activity
            enhanced_decision['execution_strategy'] = 'iceberg_orders'
            enhanced_decision['institutional_activity'] = 'high'
        
        # Optimal execution timing
        enhanced_decision['optimal_execution'] = self._calculate_optimal_execution(level_ii_features)
        
        return enhanced_decision
    
    def _calculate_optimal_execution(self, level_ii_features: Dict) -> Dict:
        """Calculate optimal execution parameters from Level II data"""
        try:
            liquidity = level_ii_features.get('avg_liquidity', 0)
            spread_bps = level_ii_features.get('avg_spread_bps', 0)
            institutional_flow = level_ii_features.get('avg_institutional_flow', 0)
            
            # Determine execution strategy
            if liquidity > 50000 and spread_bps  20000 and spread_bps  0.2:
                strategy = 'iceberg_vwap'
                urgency = 'low'
            else:
                strategy = 'limit_order_passive'
                urgency = 'low'
            
            return {
                'strategy': strategy,
                'urgency': urgency,
                'estimated_slippage_bps': min(spread_bps * 0.3, 10),
                'recommended_order_size_pct': min(liquidity / 100000 * 0.1, 0.05),  # Max 5% of liquidity
                'execution_confidence': self._calculate_execution_confidence(level_ii_features)
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to calculate optimal execution: {e}")
            return {
                'strategy': 'limit_order_passive',
                'urgency': 'low',
                'estimated_slippage_bps': 15,
                'recommended_order_size_pct': 0.01,
                'execution_confidence': 0.5
            }
    
    def _calculate_execution_confidence(self, level_ii_features: Dict) -> float:
        """Calculate confidence in execution quality"""
        try:
            liquidity = level_ii_features.get('avg_liquidity', 0)
            spread_bps = level_ii_features.get('avg_spread_bps', 50)
            sample_count = level_ii_features.get('sample_count', 0)
            
            # Higher liquidity, lower spread, more samples = higher confidence
            confidence = (
                np.clip(liquidity / 50000, 0, 1) * 0.4 +      # Liquidity factor
                np.clip(1 - spread_bps / 50, 0, 1) * 0.4 +    # Spread factor (inverted)
                np.clip(sample_count / 120, 0, 1) * 0.2       # Sample quality factor
            )
            
            return float(confidence)
            
        except Exception:
            return 0.5


class LevelIIEnhancedGeneticOptimizer(GeneticTradingOptimizer):
    """
    Genetic Algorithm enhanced with Level II execution optimization
    
    Evolves trading parameters considering:
    - Order execution costs
    - Market impact optimization
    - Liquidity-aware position sizing
    - Timing optimization
    """
    
    def __init__(self, config: Dict, level_ii_collector: LevelIIDataCollector):
        super().__init__(config)
        self.level_ii_collector = level_ii_collector
        
        # Add Level II-specific genes
        self.level_ii_genes = {
            'liquidity_threshold': (1000, 100000),      # Min liquidity for trading
            'max_spread_bps': (1, 50),                  # Max acceptable spread
            'imbalance_threshold': (0.1, 0.8),          # Min imbalance for signal
            'execution_patience': (1, 300),             # Seconds to wait for better execution
            'order_size_liquidity_pct': (0.001, 0.1),   # % of liquidity to trade
            'institutional_flow_weight': (0.0, 2.0),    # Weight for institutional signals
            'microstructure_momentum_weight': (0.0, 3.0) # Weight for microstructure momentum
        }
        
        # Add to gene space
        self.gene_space.update(self.level_ii_genes)
        
        logger.info(f"[ENHANCED] Genetic optimizer expanded with {len(self.level_ii_genes)} Level II genes")
    
    def evaluate_individual_with_level_ii(self, individual: Dict, symbol: str, 
                                         historical_data: List[Dict]) -> float:
        """Evaluate individual with Level II execution costs"""
        # Get base fitness
        base_fitness = super().evaluate_individual(individual, symbol, historical_data)
        
        # Add Level II execution quality assessment
        execution_cost_penalty = self._calculate_execution_cost_penalty(individual, symbol)
        liquidity_bonus = self._calculate_liquidity_bonus(individual, symbol)
        timing_bonus = self._calculate_timing_bonus(individual, symbol)
        
        # Combined fitness with Level II factors
        level_ii_fitness = (
            base_fitness * 0.7 +                    # Base strategy performance
            execution_cost_penalty * 0.15 +         # Execution cost optimization
            liquidity_bonus * 0.1 +                 # Liquidity utilization
            timing_bonus * 0.05                     # Execution timing
        )
        
        return level_ii_fitness
    
    def _calculate_execution_cost_penalty(self, individual: Dict, symbol: str) -> float:
        """Calculate penalty for poor execution parameters"""
        try:
            # Get recent Level II data
            features = self.level_ii_collector.get_ai_model_features(symbol)
            
            if not features:
                return 0.0
            
            # Extract individual's execution parameters
            max_spread_bps = individual.get('max_spread_bps', 20)
            order_size_pct = individual.get('order_size_liquidity_pct', 0.01)
            liquidity_threshold = individual.get('liquidity_threshold', 10000)
            
            # Current market conditions
            current_spread = features.get('avg_spread_bps', 20)
            current_liquidity = features.get('avg_liquidity', 10000)
            
            penalty = 0.0
            
            # Penalty for accepting wide spreads
            if max_spread_bps > current_spread * 1.5:
                penalty += 0.2  # Accepting unnecessarily wide spreads
            
            # Penalty for oversized orders relative to liquidity
            if order_size_pct * current_liquidity > current_liquidity * 0.05:
                penalty += 0.3  # Order too large for liquidity
            
            # Penalty for trading in low liquidity when threshold allows it
            if liquidity_threshold  float:
        """Calculate bonus for efficient liquidity utilization"""
        try:
            features = self.level_ii_collector.get_ai_model_features(symbol)
            
            if not features:
                return 0.5
            
            liquidity_threshold = individual.get('liquidity_threshold', 10000)
            order_size_pct = individual.get('order_size_liquidity_pct', 0.01)
            current_liquidity = features.get('avg_liquidity', 10000)
            
            # Bonus for appropriate liquidity requirements
            if current_liquidity > liquidity_threshold * 2:
                liquidity_bonus = 0.3  # Good liquidity buffer
            elif current_liquidity > liquidity_threshold:
                liquidity_bonus = 0.2  # Adequate liquidity
            else:
                liquidity_bonus = 0.0  # Insufficient liquidity
            
            # Bonus for reasonable order sizing
            if 0.005  float:
        """Calculate bonus for good execution timing parameters"""
        try:
            features = self.level_ii_collector.get_ai_model_features(symbol)
            
            if not features:
                return 0.5
            
            execution_patience = individual.get('execution_patience', 60)
            imbalance_threshold = individual.get('imbalance_threshold', 0.2)
            
            current_imbalance_volatility = features.get('imbalance_volatility', 0.1)
            
            # Bonus for patience in volatile imbalance conditions
            if current_imbalance_volatility > 0.2 and execution_patience > 120:
                patience_bonus = 0.3  # Patient in volatile conditions
            elif current_imbalance_volatility  Dict:
        """Analyze frequency patterns in Level II microstructure data"""
        try:
            # Get Level II time series data
            microstructure_data = self._get_microstructure_time_series(symbol, lookback_minutes)
            
            if not microstructure_data:
                return {}
            
            # Extract different signals for frequency analysis
            signals = {
                'order_imbalance': [d['order_imbalance'] for d in microstructure_data],
                'spread_bps': [d['spread_bps'] for d in microstructure_data],
                'liquidity_score': [d['liquidity_score'] for d in microstructure_data],
                'institutional_flow': [d['institutional_flow'] for d in microstructure_data],
                'momentum_signal': [d['momentum_signal'] for d in microstructure_data]
            }
            
            # Perform FFT analysis on each signal
            spectrum_analysis = {}
            for signal_name, signal_data in signals.items():
                if len(signal_data) >= 10:  # Minimum data for FFT
                    spectrum_analysis[signal_name] = self._analyze_signal_spectrum(signal_data)
            
            # Detect dominant patterns
            dominant_patterns = self._detect_dominant_microstructure_patterns(spectrum_analysis)
            
            return {
                'spectrum_analysis': spectrum_analysis,
                'dominant_patterns': dominant_patterns,
                'signal_quality': self._assess_signal_quality(signals),
                'trading_signals': self._generate_microstructure_trading_signals(dominant_patterns)
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to analyze microstructure spectrum for {symbol}: {e}")
            return {}
    
    def _get_microstructure_time_series(self, symbol: str, lookback_minutes: int) -> List[Dict]:
        """Get microstructure time series from database"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        
        query_sql = """
        SELECT timestamp, order_imbalance, spread_bps, liquidity_score, 
               institutional_flow, momentum_signal
        FROM level_ii_data.order_book_snapshots 
        WHERE symbol = %(symbol)s AND timestamp >= %(cutoff_time)s
        ORDER BY timestamp ASC
        """
        
        try:
            with RailwayPostgreSQLManager().get_session() as session:
                result = session.execute(query_sql, {
                    'symbol': symbol,
                    'cutoff_time': cutoff_time
                }).fetchall()
                
                return [
                    {
                        'timestamp': row[0],
                        'order_imbalance': float(row[1] or 0),
                        'spread_bps': float(row[2] or 0),
                        'liquidity_score': float(row[3] or 0),
                        'institutional_flow': float(row[4] or 0),
                        'momentum_signal': float(row[5] or 0)
                    }
                    for row in result
                ]
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to get microstructure time series: {e}")
            return []
    
    def _analyze_signal_spectrum(self, signal_data: List[float]) -> Dict:
        """Perform FFT analysis on signal data"""
        try:
            signal_array = np.array(signal_data)
            
            # Remove DC component and trend
            signal_array = signal_array - np.mean(signal_array)
            
            # Apply windowing to reduce spectral leakage
            window = np.hanning(len(signal_array))
            windowed_signal = signal_array * window
            
            # Compute FFT
            fft_result = np.fft.fft(windowed_signal)
            frequencies = np.fft.fftfreq(len(signal_array), d=1.0)  # 1 second sampling
            
            # Get magnitude spectrum
            magnitude_spectrum = np.abs(fft_result)
            
            # Find dominant frequencies
            dominant_indices = np.argsort(magnitude_spectrum)[-5:]  # Top 5 frequencies
            dominant_frequencies = frequencies[dominant_indices]
            dominant_magnitudes = magnitude_spectrum[dominant_indices]
            
            return {
                'dominant_frequencies': dominant_frequencies.tolist(),
                'dominant_magnitudes': dominant_magnitudes.tolist(),
                'total_power': float(np.sum(magnitude_spectrum**2)),
                'frequency_bands': self._classify_frequency_bands(frequencies, magnitude_spectrum)
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to analyze signal spectrum: {e}")
            return {}
    
    def _classify_frequency_bands(self, frequencies: np.ndarray, 
                                magnitude_spectrum: np.ndarray) -> Dict:
        """Classify power in different frequency bands"""
        band_power = {}
        
        for band_name, (low_freq, high_freq) in self.microstructure_bands.items():
            band_mask = (np.abs(frequencies) >= low_freq) & (np.abs(frequencies)  0:
            band_power = {k: v/total_power for k, v in band_power.items()}
        
        return band_power
    
    def _detect_dominant_microstructure_patterns(self, spectrum_analysis: Dict) -> Dict:
        """Detect dominant patterns across all microstructure signals"""
        patterns = {}
        
        try:
            # Analyze each signal's dominant patterns
            for signal_name, analysis in spectrum_analysis.items():
                if 'frequency_bands' in analysis:
                    bands = analysis['frequency_bands']
                    
                    # Find dominant frequency band
                    dominant_band = max(bands.keys(), key=lambda k: bands[k])
                    patterns[f'{signal_name}_dominant_band'] = dominant_band
                    patterns[f'{signal_name}_power'] = bands[dominant_band]
            
            # Cross-signal pattern detection
            patterns['overall_pattern'] = self._classify_overall_pattern(patterns)
            patterns['pattern_strength'] = self._calculate_pattern_strength(spectrum_analysis)
            patterns['synchronization'] = self._detect_signal_synchronization(spectrum_analysis)
            
            return patterns
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to detect dominant patterns: {e}")
            return {}
    
    def _classify_overall_pattern(self, patterns: Dict) -> str:
        """Classify the overall microstructure pattern"""
        try:
            # Count dominant bands across signals
            band_counts = {}
            for key, value in patterns.items():
                if key.endswith('_dominant_band'):
                    band_counts[value] = band_counts.get(value, 0) + 1
            
            if not band_counts:
                return 'no_pattern'
            
            dominant_overall_band = max(band_counts.keys(), key=lambda k: band_counts[k])
            
            # Map to trading patterns
            pattern_mapping = {
                'ultra_high_freq': 'high_frequency_trading',
                'high_freq': 'algorithmic_activity',
                'medium_freq': 'institutional_rebalancing',
                'low_freq': 'trend_development'
            }
            
            return pattern_mapping.get(dominant_overall_band, 'mixed_pattern')
            
        except Exception:
            return 'unknown_pattern'
    
    def _calculate_pattern_strength(self, spectrum_analysis: Dict) -> float:
        """Calculate overall pattern strength"""
        try:
            total_powers = []
            for analysis in spectrum_analysis.values():
                if 'total_power' in analysis:
                    total_powers.append(analysis['total_power'])
            
            if not total_powers:
                return 0.0
            
            # Higher power = stronger patterns
            avg_power = np.mean(total_powers)
            power_std = np.std(total_powers)
            
            # Normalize to [0, 1] range
            strength = min(avg_power / (avg_power + power_std + 1e-6), 1.0)
            
            return float(strength)
            
        except Exception:
            return 0.0
    
    def _detect_signal_synchronization(self, spectrum_analysis: Dict) -> float:
        """Detect synchronization between different microstructure signals"""
        try:
            if len(spectrum_analysis)  0:
                synchronization = 1.0 - min(freq_std / freq_mean, 1.0)
            else:
                synchronization = 0.0
            
            return float(synchronization)
            
        except Exception:
            return 0.0
    
    def _assess_signal_quality(self, signals: Dict) -> Dict:
        """Assess quality of microstructure signals"""
        quality_metrics = {}
        
        for signal_name, signal_data in signals.items():
            try:
                if len(signal_data)  Dict:
        """Generate trading signals from microstructure patterns"""
        signals = {
            'signal_strength': 0.0,
            'signal_direction': 'NEUTRAL',
            'confidence': 0.0,
            'pattern_based_action': 'HOLD',
            'execution_timing': 'NORMAL'
        }
        
        try:
            overall_pattern = patterns.get('overall_pattern', 'no_pattern')
            pattern_strength = patterns.get('pattern_strength', 0.0)
            synchronization = patterns.get('synchronization', 0.0)
            
            # Generate signals based on pattern type and strength
            if overall_pattern == 'high_frequency_trading' and pattern_strength > 0.7:
                signals['signal_direction'] = 'BULLISH' if synchronization > 0.6 else 'BEARISH'
                signals['signal_strength'] = pattern_strength * synchronization
                signals['pattern_based_action'] = 'BUY' if signals['signal_direction'] == 'BULLISH' else 'SELL'
                signals['execution_timing'] = 'FAST'
                
            elif overall_pattern == 'institutional_rebalancing' and pattern_strength > 0.5:
                signals['signal_direction'] = 'BULLISH'  # Institutional buying typically bullish
                signals['signal_strength'] = pattern_strength * 0.8
                signals['pattern_based_action'] = 'BUY'
                signals['execution_timing'] = 'PATIENT'
                
            elif overall_pattern == 'trend_development' and synchronization > 0.7:
                signals['signal_direction'] = 'BULLISH'
                signals['signal_strength'] = synchronization
                signals['pattern_based_action'] = 'BUY'
                signals['execution_timing'] = 'NORMAL'
            
            # Calculate overall confidence
            signals['confidence'] = (pattern_strength + synchronization) / 2
            
            return signals
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to generate microstructure trading signals: {e}")
            return signals


def main():
    """Main demonstration of Level II enhanced AI models"""
    print("=== Week 2 AI Models Enhanced with Level II Data ===")
    print("[STARTING] Level II Enhanced AI Model Testing")
    
    try:
        # Initialize Level II data collector
        print("[PROCESSING] Initializing Level II data collector...")
        symbols = ['SPY', 'QQQ', 'TSLA']
        level_ii_collector = LevelIIDataCollector(symbols)
        
        # Start brief data collection for demonstration
        print("[PROCESSING] Starting brief Level II data collection...")
        level_ii_collector.start_level_ii_streaming()
        
        # Let data collect for 2 minutes
        import time
        for i in range(12):  # 2 minutes in 10-second intervals
            time.sleep(10)
            print(f"[PROCESSING] Data collection progress: {(i+1)*10} seconds...")
        
        # Test Enhanced PPO Trader
        print("[PROCESSING] Testing Enhanced PPO Trader...")
        ppo_config = {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10
        }
        
        enhanced_ppo = LevelIIEnhancedPPOTrader(ppo_config, level_ii_collector)
        
        for symbol in symbols:
            decision = enhanced_ppo.make_enhanced_trading_decision(symbol)
            print(f"[DATA] {symbol} Enhanced PPO Decision: {decision}")
        
        # Test Enhanced Genetic Optimizer
        print("[PROCESSING] Testing Enhanced Genetic Optimizer...")
        genetic_config = {
            'population_size': 50,
            'generations': 10,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8
        }
        
        enhanced_genetic = LevelIIEnhancedGeneticOptimizer(genetic_config, level_ii_collector)
        print(f"[DATA] Enhanced Genetic Optimizer gene space size: {len(enhanced_genetic.gene_space)}")
        
        # Test Enhanced Spectrum Trader
        print("[PROCESSING] Testing Enhanced Spectrum Trader...")
        spectrum_config = {
            'analysis_window': 100,
            'frequency_bands': 5,
            'signal_threshold': 0.6
        }
        
        enhanced_spectrum = LevelIIEnhancedSpectrumTrader(spectrum_config, level_ii_collector)
        
        for symbol in symbols:
            microstructure_analysis = enhanced_spectrum.analyze_microstructure_spectrum(symbol)
            if microstructure_analysis:
                print(f"[DATA] {symbol} Microstructure Analysis: {microstructure_analysis.get('trading_signals', {})}")
        
        print("[SUCCESS] Level II enhanced AI models tested successfully!")
        
    except Exception as e:
        logger.error(f"[ERROR] Level II enhanced AI testing failed: {e}")
        print(f"[ERROR] Failed to test Level II enhanced AI models: {e}")
    
    finally:
        # Cleanup
        try:
            level_ii_collector.stop_streaming()
        except:
            pass


if __name__ == "__main__":
    main()
