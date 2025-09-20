"""
Signal Bridge Module for TensorTrade ↔ TradeApp Integration

This module provides bidirectional communication between TensorTrade RL system
and TradeAppComponents, enabling hybrid trading strategies that combine
reinforcement learning with traditional technical analysis.

Author: TensorTrade Development Team
Created: August 17, 2025
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
import json
import asyncio
from pathlib import Path

from .db_utils import DatabaseManager


class SignalType(Enum):
    """Signal types"""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"
    REDUCE = "reduce"
    INCREASE = "increase"


class SignalSource(Enum):
    """Signal sources"""
    TENSORTRADE_RL = "tensortrade_rl"
    PATTERN_RECOGNITION = "pattern_recognition"
    TECHNICAL_ANALYSIS = "technical_analysis"
    TRADE_INITIATOR = "trade_initiator"
    RISK_MANAGER = "risk_manager"
    EXTERNAL = "external"


class ConfidenceLevel(Enum):
    """Confidence levels"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class Signal:
    """Universal signal format for TensorTrade ↔ TradeApp communication"""
    signal_id: str
    symbol: str
    signal_type: SignalType
    source: SignalSource
    confidence: float  # 0.0 to 1.0
    strength: float    # 0.0 to 1.0
    timestamp: datetime
    
    # Signal details
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    
    # Technical indicators
    technical_data: Dict[str, Any] = field(default_factory=dict)
    
    # RL-specific data
    rl_action: Optional[np.ndarray] = None
    rl_features: Optional[np.ndarray] = None
    rl_q_values: Optional[np.ndarray] = None
    
    # Pattern recognition data
    pattern_type: Optional[str] = None
    pattern_confidence: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level enum"""
        if self.confidence  bool:
        """Check if signal is an entry signal"""
        return self.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]
    
    @property
    def is_exit(self) -> bool:
        """Check if signal is an exit signal"""
        return self.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]
    
    @property
    def is_long(self) -> bool:
        """Check if signal is for long position"""
        return self.signal_type in [SignalType.ENTRY_LONG, SignalType.EXIT_LONG]
    
    @property
    def is_short(self) -> bool:
        """Check if signal is for short position"""
        return self.signal_type in [SignalType.ENTRY_SHORT, SignalType.EXIT_SHORT]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for serialization"""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'source': self.source.value,
            'confidence': self.confidence,
            'strength': self.strength,
            'timestamp': self.timestamp.isoformat(),
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'position_size': self.position_size,
            'technical_data': self.technical_data,
            'rl_action': self.rl_action.tolist() if self.rl_action is not None else None,
            'rl_features': self.rl_features.tolist() if self.rl_features is not None else None,
            'rl_q_values': self.rl_q_values.tolist() if self.rl_q_values is not None else None,
            'pattern_type': self.pattern_type,
            'pattern_confidence': self.pattern_confidence,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create signal from dictionary"""
        return cls(
            signal_id=data['signal_id'],
            symbol=data['symbol'],
            signal_type=SignalType(data['signal_type']),
            source=SignalSource(data['source']),
            confidence=data['confidence'],
            strength=data['strength'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            entry_price=data.get('entry_price'),
            target_price=data.get('target_price'),
            stop_loss=data.get('stop_loss'),
            position_size=data.get('position_size'),
            technical_data=data.get('technical_data', {}),
            rl_action=np.array(data['rl_action']) if data.get('rl_action') else None,
            rl_features=np.array(data['rl_features']) if data.get('rl_features') else None,
            rl_q_values=np.array(data['rl_q_values']) if data.get('rl_q_values') else None,
            pattern_type=data.get('pattern_type'),
            pattern_confidence=data.get('pattern_confidence'),
            metadata=data.get('metadata', {})
        )


class RLSignalGenerator:
    """Converts TensorTrade RL actions to standard signals"""
    
    def __init__(self, 
                 action_space_size: int = 3,  # Hold, Buy, Sell
                 confidence_threshold: float = 0.1):
        self.action_space_size = action_space_size
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
    
    def convert_rl_action_to_signal(self,
                                  symbol: str,
                                  action: np.ndarray,
                                  features: Optional[np.ndarray] = None,
                                  q_values: Optional[np.ndarray] = None,
                                  current_price: Optional[float] = None) -> Optional[Signal]:
        """
        Convert RL action to standard signal format
        
        Args:
            symbol: Trading symbol
            action: RL action array
            features: Feature vector used for decision
            q_values: Q-values from RL model
            current_price: Current market price
            
        Returns:
            Signal object or None if no significant action
        """
        # Determine action type and confidence
        if self.action_space_size == 3:
            # [Hold, Buy, Sell]
            hold_prob, buy_prob, sell_prob = action
            
            if buy_prob > max(hold_prob, sell_prob) and buy_prob > self.confidence_threshold:
                signal_type = SignalType.ENTRY_LONG
                confidence = buy_prob
                strength = buy_prob - max(hold_prob, sell_prob)
            elif sell_prob > max(hold_prob, buy_prob) and sell_prob > self.confidence_threshold:
                signal_type = SignalType.ENTRY_SHORT
                confidence = sell_prob
                strength = sell_prob - max(hold_prob, buy_prob)
            else:
                signal_type = SignalType.HOLD
                confidence = hold_prob
                strength = 0.0
        
        elif self.action_space_size == 5:
            # [Strong Sell, Sell, Hold, Buy, Strong Buy]
            strong_sell, sell, hold, buy, strong_buy = action
            
            if strong_buy > 0.5:
                signal_type = SignalType.ENTRY_LONG
                confidence = strong_buy
                strength = strong_buy
            elif buy > max(hold, sell, strong_sell):
                signal_type = SignalType.ENTRY_LONG
                confidence = buy
                strength = buy - max(hold, sell, strong_sell)
            elif strong_sell > 0.5:
                signal_type = SignalType.ENTRY_SHORT
                confidence = strong_sell
                strength = strong_sell
            elif sell > max(hold, buy, strong_buy):
                signal_type = SignalType.ENTRY_SHORT
                confidence = sell
                strength = sell - max(hold, buy, strong_buy)
            else:
                signal_type = SignalType.HOLD
                confidence = hold
                strength = 0.0
        
        else:
            # Continuous action space
            action_value = action[0] if len(action) > 0 else 0.0
            
            if action_value > self.confidence_threshold:
                signal_type = SignalType.ENTRY_LONG
                confidence = min(abs(action_value), 1.0)
                strength = confidence
            elif action_value  List[Signal]:
        """Convert batch of RL actions to signals"""
        signals = []
        
        for i, symbol in enumerate(symbols):
            if i  List[Signal]:
        """
        Fetch signals from TradeApp components
        
        This would integrate with actual TradeApp database/API
        For now, we'll simulate the interface
        """
        signals = []
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        # This would be replaced with actual TradeApp integration
        # For example:
        # - Query pattern_recognition results
        # - Query trade_initiator signals
        # - Query technical_analysis outputs
        
        # Simulated pattern recognition signals
        for symbol in symbols:
            # Simulate pattern detection
            if np.random.random() > 0.7:  # 30% chance of pattern
                pattern_types = list(self.pattern_feature_map.keys())
                pattern = np.random.choice(pattern_types)
                
                signal = Signal(
                    signal_id=f"pattern_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_LONG if np.random.random() > 0.5 else SignalType.ENTRY_SHORT,
                    source=SignalSource.PATTERN_RECOGNITION,
                    confidence=np.random.uniform(0.6, 0.9),
                    strength=np.random.uniform(0.5, 0.8),
                    timestamp=datetime.now(),
                    pattern_type=pattern,
                    pattern_confidence=np.random.uniform(0.7, 0.95)
                )
                
                signals.append(signal)
        
        return signals
    
    def convert_signals_to_features(self, 
                                  signals: List[Signal],
                                  symbols: List[str]) -> np.ndarray:
        """
        Convert TradeApp signals to feature vectors for RL training
        
        Args:
            signals: List of signals from TradeApp
            symbols: List of symbols to generate features for
            
        Returns:
            Feature matrix [symbols x features]
        """
        n_symbols = len(symbols)
        n_pattern_features = len(self.pattern_feature_map)
        n_technical_features = len(self.technical_feature_map)
        n_total_features = n_pattern_features + n_technical_features + 5  # +5 for general features
        
        features = np.zeros((n_symbols, n_total_features))
        
        # Group signals by symbol
        symbol_signals = {symbol: [] for symbol in symbols}
        for signal in signals:
            if signal.symbol in symbol_signals:
                symbol_signals[signal.symbol].append(signal)
        
        for i, symbol in enumerate(symbols):
            symbol_signal_list = symbol_signals[symbol]
            
            if not symbol_signal_list:
                continue
            
            # Pattern recognition features
            pattern_features = np.zeros(n_pattern_features)
            for signal in symbol_signal_list:
                if signal.source == SignalSource.PATTERN_RECOGNITION and signal.pattern_type:
                    if signal.pattern_type in self.pattern_feature_map:
                        idx = self.pattern_feature_map[signal.pattern_type]
                        pattern_features[idx] = signal.confidence
            
            # Technical analysis features
            technical_features = np.zeros(n_technical_features)
            for signal in symbol_signal_list:
                if signal.source == SignalSource.TECHNICAL_ANALYSIS:
                    # Extract technical indicator signals from metadata
                    for indicator, value in signal.technical_data.items():
                        if indicator in self.technical_feature_map:
                            idx = self.technical_feature_map[indicator]
                            technical_features[idx] = value
            
            # General signal features
            general_features = np.zeros(5)
            if symbol_signal_list:
                # Signal count
                general_features[0] = len(symbol_signal_list) / 10.0  # Normalize
                
                # Average confidence
                general_features[1] = np.mean([s.confidence for s in symbol_signal_list])
                
                # Average strength
                general_features[2] = np.mean([s.strength for s in symbol_signal_list])
                
                # Long signal ratio
                long_signals = sum(1 for s in symbol_signal_list if s.is_long)
                general_features[3] = long_signals / len(symbol_signal_list)
                
                # Entry signal ratio
                entry_signals = sum(1 for s in symbol_signal_list if s.is_entry)
                general_features[4] = entry_signals / len(symbol_signal_list)
            
            # Combine all features
            features[i] = np.concatenate([
                pattern_features,
                technical_features,
                general_features
            ])
        
        return features


class HybridStrategyCoordinator:
    """Coordinates hybrid RL + traditional trading strategies"""
    
    def __init__(self, 
                 rl_weight: float = 0.6,
                 pattern_weight: float = 0.3,
                 technical_weight: float = 0.1):
        self.rl_weight = rl_weight
        self.pattern_weight = pattern_weight
        self.technical_weight = technical_weight
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure weights sum to 1
        total_weight = rl_weight + pattern_weight + technical_weight
        self.rl_weight /= total_weight
        self.pattern_weight /= total_weight
        self.technical_weight /= total_weight
    
    def combine_signals(self, 
                       rl_signals: List[Signal],
                       pattern_signals: List[Signal],
                       technical_signals: List[Signal]) -> List[Signal]:
        """
        Combine signals from different sources into unified decisions
        
        Args:
            rl_signals: Signals from RL model
            pattern_signals: Signals from pattern recognition
            technical_signals: Signals from technical analysis
            
        Returns:
            Combined signal list with weighted decisions
        """
        # Group signals by symbol
        symbol_signals = {}
        
        for signal in rl_signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = {'rl': [], 'pattern': [], 'technical': []}
            symbol_signals[signal.symbol]['rl'].append(signal)
        
        for signal in pattern_signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = {'rl': [], 'pattern': [], 'technical': []}
            symbol_signals[signal.symbol]['pattern'].append(signal)
        
        for signal in technical_signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = {'rl': [], 'pattern': [], 'technical': []}
            symbol_signals[signal.symbol]['technical'].append(signal)
        
        combined_signals = []
        
        for symbol, signals in symbol_signals.items():
            combined_signal = self._combine_symbol_signals(symbol, signals)
            if combined_signal:
                combined_signals.append(combined_signal)
        
        return combined_signals
    
    def _combine_symbol_signals(self, 
                               symbol: str, 
                               signals: Dict[str, List[Signal]]) -> Optional[Signal]:
        """Combine signals for a single symbol"""
        rl_signals = signals['rl']
        pattern_signals = signals['pattern']
        technical_signals = signals['technical']
        
        # Calculate weighted scores
        long_score = 0.0
        short_score = 0.0
        hold_score = 0.0
        
        total_confidence = 0.0
        total_strength = 0.0
        
        # RL signals
        for signal in rl_signals:
            weight = self.rl_weight * signal.confidence
            if signal.signal_type == SignalType.ENTRY_LONG:
                long_score += weight
            elif signal.signal_type == SignalType.ENTRY_SHORT:
                short_score += weight
            else:
                hold_score += weight
            
            total_confidence += signal.confidence * self.rl_weight
            total_strength += signal.strength * self.rl_weight
        
        # Pattern signals
        for signal in pattern_signals:
            weight = self.pattern_weight * signal.confidence
            if signal.signal_type == SignalType.ENTRY_LONG:
                long_score += weight
            elif signal.signal_type == SignalType.ENTRY_SHORT:
                short_score += weight
            else:
                hold_score += weight
            
            total_confidence += signal.confidence * self.pattern_weight
            total_strength += signal.strength * self.pattern_weight
        
        # Technical signals
        for signal in technical_signals:
            weight = self.technical_weight * signal.confidence
            if signal.signal_type == SignalType.ENTRY_LONG:
                long_score += weight
            elif signal.signal_type == SignalType.ENTRY_SHORT:
                short_score += weight
            else:
                hold_score += weight
            
            total_confidence += signal.confidence * self.technical_weight
            total_strength += signal.strength * self.technical_weight
        
        # Determine final signal
        max_score = max(long_score, short_score, hold_score)
        
        if max_score  List[Signal]:
        """Export RL model predictions as TradeApp-compatible signals"""
        signals = self.rl_generator.batch_convert_actions(
            symbols=symbols,
            actions=actions,
            features=features,
            current_prices=current_prices
        )
        
        # Store signals
        for signal in signals:
            self._store_signal(signal)
            self._notify_signal_handlers(signal)
        
        self.signal_count += len(signals)
        self.logger.info(f"Exported {len(signals)} RL signals")
        
        return signals
    
    def import_tradeapp_signals(self, 
                              symbols: List[str],
                              lookback_hours: int = 24) -> np.ndarray:
        """Import TradeApp signals as RL training features"""
        signals = self.tradeapp_importer.fetch_tradeapp_signals(
            symbols=symbols,
            lookback_hours=lookback_hours
        )
        
        features = self.tradeapp_importer.convert_signals_to_features(
            signals=signals,
            symbols=symbols
        )
        
        self.logger.info(f"Imported {len(signals)} TradeApp signals as {features.shape} feature matrix")
        
        return features
    
    def coordinate_hybrid_strategy(self,
                                 symbols: List[str],
                                 rl_actions: np.ndarray,
                                 rl_features: Optional[np.ndarray] = None,
                                 current_prices: Optional[Dict[str, float]] = None) -> List[Signal]:
        """Coordinate hybrid RL + traditional strategy"""
        # Generate RL signals
        rl_signals = self.rl_generator.batch_convert_actions(
            symbols=symbols,
            actions=rl_actions,
            features=rl_features,
            current_prices=current_prices
        )
        
        # Fetch TradeApp signals
        pattern_signals = self.tradeapp_importer.fetch_tradeapp_signals(symbols)
        technical_signals = []  # Would fetch from technical analysis component
        
        # Combine signals
        combined_signals = self.hybrid_coordinator.combine_signals(
            rl_signals=rl_signals,
            pattern_signals=pattern_signals,
            technical_signals=technical_signals
        )
        
        # Store and notify
        for signal in combined_signals:
            self._store_signal(signal)
            self._notify_signal_handlers(signal)
        
        self.logger.info(f"Generated {len(combined_signals)} hybrid signals from "
                        f"{len(rl_signals)} RL + {len(pattern_signals)} pattern signals")
        
        return combined_signals
    
    def _store_signal(self, signal: Signal):
        """Store signal for persistence and analysis"""
        # Add to recent signals (keep last 1000)
        self.recent_signals.append(signal)
        if len(self.recent_signals) > 1000:
            self.recent_signals = self.recent_signals[-1000:]
        
        # Save to file
        signal_file = self.signal_storage_path / f"signals_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(signal_file, 'a') as f:
            f.write(json.dumps(signal.to_dict()) + '\n')
        
        # Store in database (optional)
        if self.db_manager:
            self._store_signal_in_db(signal)
    
    def _store_signal_in_db(self, signal: Signal):
        """Store signal in database"""
        try:
            # This would insert into a tt_signals table
            # For now, we'll skip the actual database insertion
            pass
        except Exception as e:
            self.logger.error(f"Error storing signal in database: {e}")
    
    def _notify_signal_handlers(self, signal: Signal):
        """Notify all signal handlers"""
        for handler in self.signal_handlers:
            try:
                handler(signal)
            except Exception as e:
                self.logger.error(f"Error in signal handler: {e}")
    
    def get_recent_signals(self, 
                          symbols: Optional[List[str]] = None,
                          signal_types: Optional[List[SignalType]] = None,
                          hours_back: int = 24) -> List[Signal]:
        """Get recent signals with optional filtering"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_signals = []
        for signal in self.recent_signals:
            if signal.timestamp  Dict[str, Any]:
        """Generate signal performance report"""
        return {
            'total_signals_generated': self.signal_count,
            'successful_signals': self.successful_signals,
            'success_rate': self.successful_signals / max(self.signal_count, 1),
            'recent_signals_count': len(self.recent_signals),
            'signal_sources': self._get_signal_source_breakdown(),
            'signal_types': self._get_signal_type_breakdown()
        }
    
    def _get_signal_source_breakdown(self) -> Dict[str, int]:
        """Get breakdown of signals by source"""
        breakdown = {}
        for signal in self.recent_signals:
            source = signal.source.value
            breakdown[source] = breakdown.get(source, 0) + 1
        return breakdown
    
    def _get_signal_type_breakdown(self) -> Dict[str, int]:
        """Get breakdown of signals by type"""
        breakdown = {}
        for signal in self.recent_signals:
            signal_type = signal.signal_type.value
            breakdown[signal_type] = breakdown.get(signal_type, 0) + 1
        return breakdown


# Example usage and testing
if __name__ == "__main__":
    # Test signal bridge
    db_manager = DatabaseManager()
    bridge = TensorTradeSignalBridge(db_manager)
    
    # Add signal handler
    def on_signal(signal: Signal):
        print(f"Signal: {signal.signal_type.value} {signal.symbol} "
              f"(confidence: {signal.confidence:.2f}, source: {signal.source.value})")
    
    bridge.add_signal_handler(on_signal)
    
    # Test RL signal export
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    actions = np.array([
        [0.1, 0.7, 0.2],  # AAPL - Buy signal
        [0.8, 0.1, 0.1],  # MSFT - Hold signal
        [0.2, 0.2, 0.6]   # GOOGL - Sell signal
    ])
    
    prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2800.0}
    
    # Export RL signals
    rl_signals = bridge.export_rl_signals(
        symbols=symbols,
        actions=actions,
        current_prices=prices
    )
    
    print(f"\nGenerated {len(rl_signals)} RL signals")
    
    # Test hybrid strategy
    hybrid_signals = bridge.coordinate_hybrid_strategy(
        symbols=symbols,
        rl_actions=actions,
        current_prices=prices
    )
    
    print(f"Generated {len(hybrid_signals)} hybrid signals")
    
    # Get performance report
    report = bridge.get_signal_performance_report()
    print("\nSignal Performance Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
