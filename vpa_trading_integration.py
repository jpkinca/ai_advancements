#!/usr/bin/env python3
"""
VPA Trading Integration

This module integrates Volume Price Action analysis into the trading pipeline,
providing VPA-enhanced signals for decision making.

Features:
- VPA signal generation for trading decisions
- Integration with existing multimodal fusion
- Real-time VPA analysis for live trading
- Risk management with VPA filters
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VPATradingIntegration:
    """
    Integrate VPA analysis into trading decisions
    """

    def __init__(self, vpa_analyzer=None, multimodal_fusion=None, faiss_matcher=None):
        self.vpa_analyzer = vpa_analyzer
        self.multimodal_fusion = multimodal_fusion
        self.faiss_matcher = faiss_matcher

        # VPA trading parameters
        self.min_vpa_confidence = 0.6
        self.volume_threshold_multiplier = 1.2
        self.max_position_size = 1000  # shares
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04

        logger.info("[INIT] VPA Trading Integration initialized")

    async def analyze_symbol_vpa(self, symbol: str, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze VPA for a symbol and generate trading signals

        Args:
            symbol: Stock symbol
            recent_data: Recent OHLCV data (last 50-100 bars)

        Returns:
            VPA analysis results
        """
        try:
            if self.vpa_analyzer is None:
                # Fallback to basic VPA computation
                from volume.volume_price_action import VPAFeatures, VPAAnalyzer
                self.vpa_analyzer = VPAAnalyzer()

            # Ensure we have enough data
            if len(recent_data) < 20:
                return {'signal': 'insufficient_data', 'confidence': 0.0}

            # Get VPA prediction
            vpa_result = self.vpa_analyzer.predict_vpa_signal(recent_data)

            # Enhance with additional checks
            enhanced_result = self._enhance_vpa_signal(symbol, recent_data, vpa_result)

            logger.info(f"[VPA] {symbol}: {enhanced_result['signal']} (conf: {enhanced_result['confidence']:.3f})")
            return enhanced_result

        except Exception as e:
            logger.error(f"[ERROR] VPA analysis failed for {symbol}: {e}")
            return {'signal': 'error', 'confidence': 0.0, 'error': str(e)}

    def _enhance_vpa_signal(self, symbol: str, data: pd.DataFrame, vpa_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance VPA signal with additional trading logic

        Args:
            symbol: Stock symbol
            data: Market data
            vpa_result: Basic VPA result

        Returns:
            Enhanced VPA result
        """
        enhanced = vpa_result.copy()

        # Add position sizing based on VPA strength
        confidence = vpa_result.get('confidence', 0.0)
        volume_confirmation = vpa_result.get('volume_confirmation', 0)

        # Calculate position size based on confidence and volume
        base_position = self.max_position_size
        confidence_multiplier = min(confidence * 2, 1.0)  # Max 2x position for high confidence
        volume_multiplier = min(volume_confirmation / 5, 1.0)  # Scale by volume confirmations

        position_size = int(base_position * confidence_multiplier * volume_multiplier)
        enhanced['position_size'] = max(position_size, 100)  # Minimum 100 shares

        # Add risk management levels
        current_price = data['close'].iloc[-1] if not data.empty else 0

        if vpa_result['signal'] in ['bullish', 'bearish']:
            # Set stop loss and take profit
            if vpa_result['signal'] == 'bullish':
                enhanced['stop_loss'] = current_price * (1 - self.stop_loss_pct)
                enhanced['take_profit'] = current_price * (1 + self.take_profit_pct)
                enhanced['entry_price'] = current_price
            else:  # bearish
                enhanced['stop_loss'] = current_price * (1 + self.stop_loss_pct)
                enhanced['take_profit'] = current_price * (1 - self.take_profit_pct)
                enhanced['entry_price'] = current_price

            # Calculate risk-reward ratio
            risk = abs(current_price - enhanced['stop_loss'])
            reward = abs(enhanced['take_profit'] - current_price)
            enhanced['risk_reward_ratio'] = reward / risk if risk > 0 else 0

        # Add market regime assessment
        enhanced['market_regime'] = self._assess_market_regime(data)

        return enhanced

    def _assess_market_regime(self, data: pd.DataFrame) -> str:
        """
        Assess current market regime based on VPA and price action

        Args:
            data: Market data

        Returns:
            Market regime string
        """
        if len(data) < 20:
            return 'unknown'

        # Calculate recent volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized

        # Volume trends
        volume_trend = data['volume'].tail(10).mean() / data['volume'].tail(20).mean()

        # Trend strength
        sma_20 = data['close'].rolling(20).mean()
        trend_strength = abs(data['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]

        # Determine regime
        if volatility > 0.3 and volume_trend > 1.2:
            return 'high_volatility'
        elif trend_strength > 0.05 and volume_trend > 1.1:
            return 'trending_bullish' if data['close'].iloc[-1] > sma_20.iloc[-1] else 'trending_bearish'
        elif volatility < 0.2 and volume_trend < 0.9:
            return 'low_volatility'
        else:
            return 'sideways'

    async def get_multimodal_vpa_signal(self, symbol: str, data: pd.DataFrame,
                                       chart_image: Any = None, technical_features: Dict = None) -> Dict[str, Any]:
        """
        Get combined multimodal + VPA signal

        Args:
            symbol: Stock symbol
            data: Market data
            chart_image: Chart image for CLIP
            technical_features: Technical indicators

        Returns:
            Combined signal
        """
        try:
            # Get VPA signal
            vpa_signal = await self.analyze_symbol_vpa(symbol, data)

            if self.multimodal_fusion is None or chart_image is None or technical_features is None:
                # Return VPA-only signal
                return vpa_signal

            # Get multimodal signal
            multimodal_result = self.multimodal_fusion.predict_multimodal(
                chart_image, ['bullish pattern', 'bearish pattern', 'neutral'], technical_features, data
            )

            # Combine signals
            combined_signal = self._combine_signals(vpa_signal, multimodal_result)

            logger.info(f"[COMBINED] {symbol}: VPA={vpa_signal['signal']}, Multi={multimodal_result.get('prediction', 'unknown')} -> {combined_signal['signal']}")

            return combined_signal

        except Exception as e:
            logger.error(f"[ERROR] Combined signal generation failed for {symbol}: {e}")
            return vpa_signal if 'vpa_signal' in locals() else {'signal': 'error'}

    def _combine_signals(self, vpa_result: Dict[str, Any], multimodal_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine VPA and multimodal signals

        Args:
            vpa_result: VPA analysis result
            multimodal_result: Multimodal fusion result

        Returns:
            Combined result
        """
        combined = vpa_result.copy()

        # Get signal strengths
        vpa_conf = vpa_result.get('confidence', 0.0)
        multi_conf = multimodal_result.get('confidence', 0.0)

        # Map multimodal prediction to signal
        multi_pred = multimodal_result.get('prediction', 2)  # Default neutral
        multi_signal = {0: 'bullish', 1: 'bearish', 2: 'neutral'}.get(multi_pred, 'neutral')

        # Agreement bonus
        if vpa_result['signal'] == multi_signal and vpa_result['signal'] != 'neutral':
            combined['confidence'] = min(vpa_conf * 1.2, 1.0)  # 20% boost for agreement
            combined['agreement'] = True
        else:
            # Weighted average for disagreement
            combined['confidence'] = (vpa_conf * 0.6 + multi_conf * 0.4)
            combined['agreement'] = False

        # Keep higher position size if signals agree
        if combined.get('agreement', False):
            combined['position_size'] = min(combined.get('position_size', 1000) * 1.5, 2000)

        combined['multimodal_signal'] = multi_signal
        combined['multimodal_confidence'] = multi_conf

        return combined

    async def execute_vpa_trade(self, symbol: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade based on VPA signal

        Args:
            symbol: Stock symbol
            signal: VPA signal with position details

        Returns:
            Trade execution result
        """
        try:
            if signal['signal'] not in ['bullish', 'bearish'] or signal['confidence'] < self.min_vpa_confidence:
                return {'status': 'no_trade', 'reason': 'signal_too_weak'}

            # In a real implementation, this would connect to IBKR or other broker
            # For now, simulate trade execution

            trade_details = {
                'symbol': symbol,
                'action': 'BUY' if signal['signal'] == 'bullish' else 'SELL',
                'quantity': signal.get('position_size', 100),
                'entry_price': signal.get('entry_price', 0),
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit': signal.get('take_profit', 0),
                'timestamp': datetime.now(),
                'signal_confidence': signal['confidence'],
                'vpa_features': signal.get('vpa_features', {})
            }

            logger.info(f"[TRADE] Executing {trade_details['action']} {trade_details['quantity']} {symbol} at ${trade_details['entry_price']:.2f}")

            # Simulate execution
            trade_details['status'] = 'executed'
            trade_details['execution_price'] = trade_details['entry_price'] * (1 + np.random.normal(0, 0.001))  # Small slippage
            trade_details['order_id'] = f"vpa_{symbol}_{int(datetime.now().timestamp())}"

            return trade_details

        except Exception as e:
            logger.error(f"[ERROR] Trade execution failed for {symbol}: {e}")
            return {'status': 'error', 'error': str(e)}

    async def monitor_vpa_positions(self, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Monitor open positions using VPA analysis

        Args:
            positions: List of open positions

        Returns:
            Updated positions with exit signals
        """
        updated_positions = []

        for position in positions:
            try:
                symbol = position['symbol']

                # Get current market data (would need real data source)
                # For simulation, assume we have current_data
                current_data = await self._get_current_market_data(symbol)

                if current_data is not None:
                    # Re-analyze VPA for exit signal
                    current_vpa = await self.analyze_symbol_vpa(symbol, current_data)

                    # Check exit conditions
                    exit_signal = self._check_exit_conditions(position, current_vpa, current_data)

                    position_copy = position.copy()
                    position_copy['current_vpa'] = current_vpa
                    position_copy['exit_signal'] = exit_signal

                    updated_positions.append(position_copy)
                else:
                    updated_positions.append(position)

            except Exception as e:
                logger.error(f"[ERROR] Position monitoring failed for {position['symbol']}: {e}")
                updated_positions.append(position)

        return updated_positions

    def _check_exit_conditions(self, position: Dict[str, Any], current_vpa: Dict[str, Any], current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if position should be exited based on VPA and risk management

        Args:
            position: Position details
            current_vpa: Current VPA analysis
            current_data: Current market data

        Returns:
            Exit signal details
        """
        current_price = current_data['close'].iloc[-1] if not current_data.empty else position['execution_price']

        # Stop loss check
        if position['action'] == 'BUY':
            if current_price <= position['stop_loss']:
                return {'exit': True, 'reason': 'stop_loss', 'price': current_price}
            elif current_price >= position['take_profit']:
                return {'exit': True, 'reason': 'take_profit', 'price': current_price}
        else:  # SELL
            if current_price >= position['stop_loss']:
                return {'exit': True, 'reason': 'stop_loss', 'price': current_price}
            elif current_price <= position['take_profit']:
                return {'exit': True, 'reason': 'take_profit', 'price': current_price}

        # VPA-based exit signals
        if current_vpa['signal'] == 'neutral' and current_vpa['confidence'] > 0.7:
            return {'exit': True, 'reason': 'vpa_neutral', 'price': current_price}

        # Opposite VPA signal with high confidence
        opposite_signals = {
            'BUY': 'bearish',
            'SELL': 'bullish'
        }

        if current_vpa['signal'] == opposite_signals[position['action']] and current_vpa['confidence'] > 0.8:
            return {'exit': True, 'reason': 'vpa_reversal', 'price': current_price}

        return {'exit': False, 'reason': 'hold'}

    async def _get_current_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get current market data for a symbol
        In real implementation, this would fetch from IBKR or other source
        """
        # Placeholder - in real implementation, fetch current bars
        return None

# Integration functions

async def process_vpa_trading_signal(symbol: str, bar_data: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Process VPA trading signal for a symbol (equivalent to Celery task)

    Args:
        symbol: Stock symbol
        bar_data: Recent bar data

    Returns:
        Trading decision
    """
    try:
        # Convert bar data to DataFrame
        df = pd.DataFrame(bar_data)
        df['date'] = pd.to_datetime(df.get('timestamp', df.get('date')))
        df.set_index('date', inplace=True)

        # Initialize VPA trading integration
        vpa_trading = VPATradingIntegration()

        # Analyze VPA
        vpa_signal = await vpa_trading.analyze_symbol_vpa(symbol, df)

        # Execute trade if signal is strong enough
        if vpa_signal['confidence'] >= 0.7:
            trade_result = await vpa_trading.execute_vpa_trade(symbol, vpa_signal)
            return {
                'symbol': symbol,
                'vpa_signal': vpa_signal,
                'trade_result': trade_result,
                'timestamp': datetime.now()
            }
        else:
            return {
                'symbol': symbol,
                'vpa_signal': vpa_signal,
                'trade_result': {'status': 'no_trade', 'reason': 'low_confidence'},
                'timestamp': datetime.now()
            }

    except Exception as e:
        logger.error(f"[ERROR] VPA trading signal processing failed for {symbol}: {e}")
        return {
            'symbol': symbol,
            'error': str(e),
            'timestamp': datetime.now()
        }

# Example usage
async def demo_vpa_trading():
    """Demonstrate VPA trading integration"""
    logger.info("[DEMO] VPA Trading Integration Demo")

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.random.randn(50).cumsum() * 0.5,
        'high': 105 + np.random.randn(50).cumsum() * 0.5,
        'low': 95 + np.random.randn(50).cumsum() * 0.5,
        'close': 100 + np.random.randn(50).cumsum() * 0.5,
        'volume': np.random.lognormal(15, 1, 50)
    })
    sample_data.set_index('date', inplace=True)

    # Initialize VPA trading
    vpa_trading = VPATradingIntegration()

    # Analyze VPA for sample symbol
    symbol = 'AAPL'
    vpa_result = await vpa_trading.analyze_symbol_vpa(symbol, sample_data)

    print(f"VPA Analysis for {symbol}:")
    print(f"  Signal: {vpa_result['signal']}")
    print(f"  Confidence: {vpa_result['confidence']:.3f}")
    print(f"  Position Size: {vpa_result.get('position_size', 'N/A')}")

    if vpa_result['signal'] in ['bullish', 'bearish']:
        print(f"  Entry Price: ${vpa_result.get('entry_price', 0):.2f}")
        print(f"  Stop Loss: ${vpa_result.get('stop_loss', 0):.2f}")
        print(f"  Take Profit: ${vpa_result.get('take_profit', 0):.2f}")
        print(f"  Risk-Reward: {vpa_result.get('risk_reward_ratio', 0):.2f}")

if __name__ == "__main__":
    asyncio.run(demo_vpa_trading())