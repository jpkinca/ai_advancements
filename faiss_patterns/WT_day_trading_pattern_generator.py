import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class WarriorTradingPatternGenerator:
    """
    Cameron Ross / Warrior Trading day trading pattern recognition system
    Focus on intraday momentum, volume, and quick scalping opportunities
    """
    
    def __init__(self):
        self.pattern_weights = {
            'bull_flag': 0.20,
            'flat_top_breakout': 0.18,
            'abcd_pattern': 0.15,
            'gap_and_go': 0.15,
            'orb_opening_range_breakout': 0.12,
            'doji_reversal': 0.10,
            'red_to_green_move': 0.10
        }
        
        # Entry and exit rules
        self.entry_rules = {
            'min_volume_ratio': 2.0,        # Volume vs 20-day avg
            'min_price': 2.00,               # Above $2 for momentum
            'max_spread_pct': 0.05,          # 5% max bid-ask spread
            'min_relative_volume': 1.5       # 1.5x normal volume
        }
        
        self.risk_management = {
            'max_risk_per_trade': 0.02,      # 2% account risk
            'profit_target_1': 0.10,         # 10% profit target 1
            'profit_target_2': 0.20,         # 20% profit target 2
            'stop_loss_pct': 0.05,           # 5% stop loss
            'trailing_stop_pct': 0.03        # 3% trailing stop
        }
    
    def generate_warrior_patterns(self, 
                                 minute_data: pd.DataFrame,
                                 volume_data: pd.DataFrame,
                                 level2_data: pd.DataFrame = None,
                                 premarket_data: pd.DataFrame = None) -> Dict:
        """
        Generate comprehensive Warrior Trading pattern vectors
        """
        patterns = {}
        
        # Core momentum patterns
        patterns['bull_flag'] = self._detect_bull_flag(minute_data, volume_data)
        patterns['flat_top_breakout'] = self._detect_flat_top_breakout(minute_data, volume_data)
        patterns['abcd_pattern'] = self._detect_abcd_pattern(minute_data, volume_data)
        
        # Opening patterns
        patterns['gap_and_go'] = self._detect_gap_and_go(minute_data, volume_data, premarket_data)
        patterns['orb'] = self._detect_opening_range_breakout(minute_data, volume_data)
        patterns['red_to_green'] = self._detect_red_to_green_move(minute_data, volume_data)
        
        # Reversal patterns
        patterns['doji_reversal'] = self._detect_doji_reversal(minute_data, volume_data)
        patterns['hammer_reversal'] = self._detect_hammer_reversal(minute_data, volume_data)
        
        # Volume-based patterns
        patterns['volume_spike'] = self._detect_volume_spike_breakout(minute_data, volume_data)
        patterns['consolidation_breakout'] = self._detect_consolidation_breakout(minute_data, volume_data)
        
        # Level 2 patterns (if data available)
        if level2_data is not None:
            patterns['level2_support_break'] = self._detect_level2_support_break(minute_data, level2_data)
            patterns['iceberg_orders'] = self._detect_iceberg_orders(level2_data)
        
        # Entry/Exit signals
        patterns['buy_signals'] = self._generate_buy_signals(patterns, minute_data, volume_data)
        patterns['sell_signals'] = self._generate_sell_signals(patterns, minute_data, volume_data)
        patterns['risk_management'] = self._calculate_risk_management_levels(minute_data, patterns)
        
        return patterns
    
    def _detect_bull_flag(self, minute_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Detect bull flag pattern - sharp move up followed by tight consolidation
        Key Warrior Trading pattern for momentum continuation
        """
        if len(minute_data) = len(window):
            pole_volume = volume_data.iloc[-len(window):].iloc[pole_start_idx:pole_end_idx+1]['Volume'].mean()
            avg_volume = volume_data.iloc[-len(window):]['Volume'].mean()
            volume_confirmation = pole_volume / avg_volume if avg_volume > 0 else 1
        else:
            volume_confirmation = 1
        
        # Flag formation (consolidation after pole)
        flag_start_idx = pole_end_idx + 1
        if flag_start_idx >= len(window):
            return {'detected': False, 'confidence': 0.0, 'vector': np.zeros(10)}
        
        flag_data = window.iloc[flag_start_idx:]
        if len(flag_data) = len(flag_data) else pole_volume
        volume_dryup = max(0, (pole_volume - flag_volume) / pole_volume) if pole_volume > 0 else 0
        
        # Breakout setup (current price near flag high)
        current_price = minute_data['Close'].iloc[-1]
        breakout_proximity = max(0, 1 - (flag_high - current_price) / (flag_high * 0.02))  # Within 2%
        
        # Pattern vector
        vector = np.array([
            min(pole_gain * 10, 2.0),        # Pole strength
            tightness_score,                  # Flag tightness
            angle_score,                      # Flag angle
            min(volume_confirmation, 3.0),    # Volume confirmation
            min(volume_dryup * 2, 1.0),      # Volume dry-up
            breakout_proximity,               # Breakout setup
            len(flag_data) / 10.0,           # Flag duration
            flag_range,                       # Raw flag range
            flag_slope,                       # Raw flag slope
            1.0 if pole_gain > 0.05 else 0.0  # Strong pole (>5%)
        ], dtype=np.float32)
        
        # Confidence calculation
        confidence = (
            min(pole_gain * 5, 1.0) * 0.3 +      # Pole strength
            tightness_score * 0.25 +              # Flag quality
            min(volume_confirmation / 2, 1.0) * 0.25 +  # Volume
            breakout_proximity * 0.2              # Entry timing
        )
        
        # Buy point calculation
        buy_point = flag_high * 1.02  # 2% above flag high
        stop_loss = flag_low * 0.98   # 2% below flag low
        
        return {
            'detected': confidence > 0.6 and pole_gain > 0.03,
            'confidence': confidence,
            'vector': vector,
            'buy_point': buy_point,
            'stop_loss': stop_loss,
            'profit_target_1': buy_point * 1.10,
            'profit_target_2': buy_point * 1.20,
            'pole_gain': pole_gain,
            'flag_range': flag_range
        }
    
    def _detect_flat_top_breakout(self, minute_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Detect flat top breakout - multiple tests of resistance followed by volume breakout
        """
        if len(minute_data) = resistance_level * 0.995  # Within 0.5%
        
        if len(volume_data) >= len(window) and near_resistance:
            recent_volume = volume_data.iloc[-5:]['Volume'].mean()
            avg_volume = volume_data.iloc[-20:]['Volume'].mean()
            volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1
        else:
            volume_surge = 1
        
        # Breakout strength
        breakout_strength = max(0, (current_price - resistance_level) / resistance_level)
        
        # Support level (recent lows)
        lows = window['Low'].rolling(5).min()
        support_level = lows.iloc[-10:].mean()  # Average support
        risk_reward = (resistance_level - support_level) / support_level if support_level > 0 else 0
        
        vector = np.array([
            flatness_score,
            test_score,
            min(volume_surge, 3.0),
            min(breakout_strength * 20, 2.0),
            min(risk_reward * 5, 1.0),
            1.0 if near_resistance else 0.0,
            len(highs) / 5.0,  # Number of tests normalized
            resistance_std / resistance_level if resistance_level > 0 else 0
        ], dtype=np.float32)
        
        confidence = (flatness_score + test_score + min(volume_surge / 2, 1.0)) / 3.0
        
        return {
            'detected': confidence > 0.5 and tests_in_range >= 2,
            'confidence': confidence,
            'vector': vector,
            'resistance_level': resistance_level,
            'support_level': support_level,
            'buy_point': resistance_level * 1.005,  # Just above resistance
            'stop_loss': support_level,
            'volume_surge': volume_surge
        }
    
    def _detect_abcd_pattern(self, minute_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Detect ABCD pattern - measured move pattern with predictable target
        """
        if len(minute_data)  0 else 0
        
        # 3. Fibonacci relationships
        bc_retracement = abs(c_price - b_price) / abs(b_price - a_price)
        fib_score = 1.0 if 0.618 = len(window):
            vol_window = volume_data.iloc[-len(window):]
            cd_volume = vol_window.iloc[abcd_points[2][0]:abcd_points[3][0]+1]['Volume'].mean()
            ab_volume = vol_window.iloc[abcd_points[0][0]:abcd_points[1][0]+1]['Volume'].mean()
            volume_ratio = cd_volume / ab_volume if ab_volume > 0 else 1
        else:
            volume_ratio = 1
        
        # Current position and projection
        current_price = minute_data['Close'].iloc[-1]
        
        # Calculate D projection if we're at C
        if abcd_points[3][0] == len(window) - 1:  # D is current point
            pattern_completion = 1.0
        else:
            # Project where D should be
            projected_d = c_price + (ab_move * c_price * (1 if d_price > c_price else -1))
            completion_accuracy = 1.0 - abs(d_price - projected_d) / projected_d if projected_d > 0 else 0
            pattern_completion = completion_accuracy
        
        vector = np.array([
            move_similarity,
            time_similarity,
            fib_score,
            min(volume_ratio, 2.0),
            pattern_completion,
            ab_move,
            cd_move,
            bc_retracement,
            1.0 if move_similarity > 0.8 and fib_score > 0.5 else 0.0
        ], dtype=np.float32)
        
        confidence = (move_similarity + time_similarity + fib_score + min(volume_ratio / 2, 1.0)) / 4.0
        
        # Target calculation (D + AB move)
        target_price = d_price + (ab_move * d_price * (1 if d_price > c_price else -1))
        
        return {
            'detected': confidence > 0.6 and move_similarity > 0.7,
            'confidence': confidence,
            'vector': vector,
            'abcd_points': {'A': a_price, 'B': b_price, 'C': c_price, 'D': d_price},
            'target_price': target_price,
            'buy_point': d_price if d_price  Dict:
        """
        Detect gap and go pattern - gap up on news with volume follow-through
        """
        if len(minute_data)  0:
            previous_close = premarket_data['Close'].iloc[-1]
        else:
            # Assume gap from first bar of the day
            previous_close = minute_data['Open'].iloc[0]
        
        market_open = minute_data['Open'].iloc[0]
        gap_size = (market_open - previous_close) / previous_close if previous_close > 0 else 0
        
        if gap_size  0 else 0
        
        # Volume confirmation (first hour should be high volume)
        if len(volume_data) >= 10:
            opening_volume = volume_data.iloc[:10]['Volume'].sum()
            # Compare to historical first hour volume (assume average daily volume / 8)
            avg_hourly_volume = volume_data.iloc[:min(len(volume_data), 100)]['Volume'].mean()
            volume_ratio = opening_volume / (avg_hourly_volume * 10) if avg_hourly_volume > 0 else 1
        else:
            volume_ratio = 1
        
        # Gap holding pattern (price staying above gap level)
        holding_gap = (low_after_open - previous_close) / previous_close if previous_close > 0 else 0
        gap_hold_score = max(0, holding_gap / gap_size) if gap_size > 0 else 0
        
        # Momentum score
        momentum = (current_price - market_open) / market_open if market_open > 0 else 0
        momentum_score = max(0, momentum * 10)  # Scale momentum
        
        # Bull flag formation after gap
        if len(first_10_min) >= 5:
            consolidation_range = (first_10_min.iloc[2:]['High'].max() - first_10_min.iloc[2:]['Low'].min()) / market_open
            consolidation_score = max(0, (0.05 - consolidation_range) / 0.05)  # Prefer tight consolidation
        else:
            consolidation_score = 0
        
        vector = np.array([
            min(gap_size * 5, 2.0),          # Gap magnitude
            min(followthrough * 10, 2.0),     # Follow-through strength
            min(volume_ratio, 3.0),          # Volume confirmation
            gap_hold_score,                   # Gap holding
            momentum_score,                   # Current momentum
            consolidation_score,              # Tight consolidation
            1.0 if gap_size > 0.05 else 0.0, # Strong gap (>5%)
            holding_gap                       # Raw gap hold level
        ], dtype=np.float32)
        
        confidence = (
            min(gap_size * 5, 1.0) * 0.3 +
            min(volume_ratio / 2, 1.0) * 0.3 +
            gap_hold_score * 0.2 +
            min(followthrough * 5, 1.0) * 0.2
        )
        
        return {
            'detected': confidence > 0.6 and gap_size > 0.02 and gap_hold_score > 0.5,
            'confidence': confidence,
            'vector': vector,
            'gap_size': gap_size,
            'previous_close': previous_close,
            'buy_point': high_after_open * 1.02,
            'stop_loss': previous_close,
            'volume_ratio': volume_ratio
        }
    
    def _detect_opening_range_breakout(self, minute_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Detect Opening Range Breakout (ORB) - breakout of first 5/15/30 minute range
        """
        if len(minute_data) = 30 else minute_data.iloc[:15]
        }
        
        breakouts = {}
        
        for period, range_data in orb_ranges.items():
            if len(range_data) == 0:
                continue
                
            orb_high = range_data['High'].max()
            orb_low = range_data['Low'].min()
            orb_range = orb_high - orb_low
            
            current_price = minute_data['Close'].iloc[-1]
            
            # Breakout detection
            upside_breakout = current_price > orb_high
            downside_breakout = current_price  breakout_idx and (upside_breakout or downside_breakout):
                breakout_volume = volume_data.iloc[breakout_idx:breakout_idx+5]['Volume'].mean()
                orb_volume = volume_data.iloc[:len(range_data)]['Volume'].mean()
                volume_confirmation = breakout_volume / orb_volume if orb_volume > 0 else 1
            else:
                volume_confirmation = 1
            
            # Breakout strength
            if upside_breakout:
                breakout_distance = (current_price - orb_high) / orb_high
                direction = 1
            elif downside_breakout:
                breakout_distance = (orb_low - current_price) / current_price
                direction = -1
            else:
                breakout_distance = 0
                direction = 0
            
            # Range quality (larger ranges are better for ORB)
            range_quality = min(orb_range / minute_data['Close'].iloc[0] * 20, 1.0)  # Normalize by price
            
            breakouts[period] = {
                'detected': breakout_distance > 0.005,  # >0.5% breakout
                'orb_high': orb_high,
                'orb_low': orb_low,
                'orb_range': orb_range,
                'breakout_distance': breakout_distance,
                'volume_confirmation': volume_confirmation,
                'range_quality': range_quality,
                'direction': direction
            }
        
        # Choose best breakout
        best_breakout = None
        best_score = 0
        
        for period, data in breakouts.items():
            if data['detected']:
                score = data['breakout_distance'] * data['volume_confirmation'] * data['range_quality']
                if score > best_score:
                    best_score = score
                    best_breakout = (period, data)
        
        if best_breakout is None:
            return {'detected': False, 'confidence': 0.0, 'vector': np.zeros(7)}
        
        period, orb_data = best_breakout
        
        vector = np.array([
            orb_data['breakout_distance'] * 20,  # Scaled breakout distance
            min(orb_data['volume_confirmation'], 3.0),
            orb_data['range_quality'],
            orb_data['direction'],  # 1 for up, -1 for down
            orb_data['orb_range'] / minute_data['Close'].iloc[0],  # Range relative to price
            1.0 if orb_data['breakout_distance'] > 0.01 else 0.0,  # Strong breakout
            {'5min': 1, '15min': 2, '30min': 3}.get(period, 1)  # Time period identifier
        ], dtype=np.float32)
        
        confidence = min(orb_data['breakout_distance'] * 10 + 
                        min(orb_data['volume_confirmation'] / 2, 1.0) + 
                        orb_data['range_quality'], 3.0) / 3.0
        
        # Calculate targets
        if orb_data['direction'] == 1:  # Upside breakout
            buy_point = orb_data['orb_high'] * 1.005
            target_1 = orb_data['orb_high'] + orb_data['orb_range']
            target_2 = orb_data['orb_high'] + orb_data['orb_range'] * 2
            stop_loss = orb_data['orb_low']
        else:  # Downside breakout
            buy_point = orb_data['orb_low'] * 0.995  # Short entry
            target_1 = orb_data['orb_low'] - orb_data['orb_range']
            target_2 = orb_data['orb_low'] - orb_data['orb_range'] * 2
            stop_loss = orb_data['orb_high']
        
        return {
            'detected': True,
            'confidence': confidence,
            'vector': vector,
            'period': period,
            'orb_high': orb_data['orb_high'],
            'orb_low': orb_data['orb_low'],
            'buy_point': buy_point,
            'target_1': target_1,
            'target_2': target_2,
            'stop_loss': stop_loss,
            'direction': orb_data['direction']
        }
    
    def _detect_red_to_green_move(self, minute_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Detect red to green move - stock moving from red (negative) to green (positive) on the day
        """
        if len(minute_data) = 0:  # Crossed from red to green
                transition_found = True
                transition_idx = i
                break
        
        if not transition_found or daily_change = len(minute_data):
            transition_volume = volume_data.iloc[max(0, transition_idx-2):transition_idx+3]['Volume'].mean()
            avg_volume = volume_data.iloc[:transition_idx]['Volume'].mean()
            volume_surge = transition_volume / avg_volume if avg_volume > 0 else 1
        else:
            volume_surge = 1
        
        # Momentum after crossing green
        post_transition_data = minute_data.iloc[transition_idx:]
        momentum = (current_price - transition_price) / transition_price if transition_price > 0 else 0
        
        # Depth of red move before recovery
        min_price = minute_data['Close'].iloc[:transition_idx].min()
        red_depth = (open_price - min_price) / open_price if open_price > 0 else 0
        recovery_strength = (transition_price - min_price) / min_price if min_price > 0 else 0
        
        # Sustainability (holding green)
        if len(post_transition_data) >= 3:
            pullback_severity = (transition_price - post_transition_data['Low'].min()) / transition_price
            sustainability = max(0, 1 - pullback_severity / 0.02)  # Prefer  0.6 and daily_change > 0,
            'confidence': confidence,
            'vector': vector,
            'transition_price': transition_price,
            'red_depth': red_depth,
            'buy_point': current_price * 1.01,  # Just above current
            'stop_loss': min_price * 1.02,     # Above the low
            'volume_surge': volume_surge
        }
    
    def _detect_doji_reversal(self, minute_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Detect doji reversal patterns - indecision candles that can signal reversal
        """
        if len(minute_data)  body_size * 2 and lower_shadow  5 else 0.5  # Stronger at resistance
                elif lower_shadow > body_size * 2 and upper_shadow  5 else 0.5  # Stronger at support
                elif abs(upper_shadow - lower_shadow) / total_range  i:
                    doji_volume = volume_data.iloc[i]['Volume']
                    avg_volume = volume_data.iloc[max(0, i-10):i]['Volume'].mean()
                    volume_confirmation = min(doji_volume / avg_volume, 2.0) if avg_volume > 0 else 1
                else:
                    volume_confirmation = 1
                
                # Context analysis (trend before doji)
                if i >= 5:
                    pre_trend = (minute_data['Close'].iloc[i-1] - minute_data['Close'].iloc[i-5]) / minute_data['Close'].iloc[i-5]
                    trend_strength = abs(pre_trend) * 10
                else:
                    trend_strength = 0
                
                doji_signals.append({
                    'index': i,
                    'type': doji_type,
                    'body_ratio': body_ratio,
                    'reversal_strength': reversal_strength,
                    'volume_confirmation': volume_confirmation,
                    'trend_strength': trend_strength,
                    'price': close_price
                })
        
        if not doji_signals:
            return {'detected': False, 'confidence': 0.0, 'vector': np.zeros(7)}
        
        # Take the most recent significant doji
        best_doji = max(doji_signals, key=lambda x: x['reversal_strength'] * x['volume_confirmation'])
        
        # Confirmation from subsequent price action
        doji_idx = best_doji['index']
        if doji_idx  0.5,
            'confidence': confidence,
            'vector': vector,
            'doji_type': best_doji['type'],
            'doji_price': best_doji['price'],
            'reversal_direction': 'bullish' if best_doji['type'] == 'hammer' else 'bearish' if best_doji['type'] == 'shooting_star' else 'neutral'
        }
    
    def _detect_hammer_reversal(self, minute_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Detect hammer reversal pattern - long lower shadow with small body at top
        """
        if len(minute_data) = 0.6 and       # Long lower shadow (60%+ of range)
                body_ratio  i:
                    hammer_volume = volume_data.iloc[i]['Volume']
                    avg_volume = volume_data.iloc[max(0, i-5):i]['Volume'].mean()
                    volume_ratio = hammer_volume / avg_volume if avg_volume > 0 else 1
                else:
                    volume_ratio = 1
                
                # Trend context (should be in downtrend for reversal)
                if i >= 3:
                    trend_before = (minute_data['Close'].iloc[i-3:i].iloc[-1] - 
                                   minute_data['Close'].iloc[i-3:i].iloc[0]) / minute_data['Close'].iloc[i-3:i].iloc[0]
                    downtrend_context = max(0, -trend_before * 5)  # Reward downtrend before hammer
                else:
                    downtrend_context = 0.5
                
                hammer_quality = lower_shadow_ratio + (1 - body_ratio) + (1 - upper_shadow_ratio)
                
                hammer_candidates.append({
                    'index': i,
                    'quality': hammer_quality,
                    'volume_ratio': volume_ratio,
                    'downtrend_context': downtrend_context,
                    'lower_shadow_ratio': lower_shadow_ratio,
                    'body_ratio': body_ratio,
                    'price': close_price,
                    'low': low_price
                })
        
        if not hammer_candidates:
            return {'detected': False, 'confidence': 0.0, 'vector': np.zeros(6)}
        
        # Take best hammer (most recent with high quality)
        best_hammer = max(hammer_candidates, key=lambda x: x['quality'] * (1 + x['volume_ratio']))
        
        # Confirmation from next candle(s)
        hammer_idx = best_hammer['index']
        if hammer_idx  0.6,
            'confidence': confidence,
            'vector': vector,
            'hammer_price': best_hammer['price'],
            'hammer_low': best_hammer['low'],
            'buy_point': best_hammer['price'] * 1.005,
            'stop_loss': best_hammer['low'] * 0.98
        }
    
    def _detect_volume_spike_breakout(self, minute_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Detect volume spike breakouts - unusual volume with price acceleration
        """
        if len(minute_data)  0 else 1
            
            if volume_ratio >= 2.0:  # Volume spike (2x normal)
                # Analyze corresponding price action
                price_change = (minute_data.iloc[i]['Close'] - minute_data.iloc[i-1]['Close']) / minute_data.iloc[i-1]['Close']
                
                # Volume-price relationship
                vp_correlation = volume_ratio * abs(price_change) * 100  # Scale for readability
                
                # Breakout context (are we breaking through resistance/support?)
                if i >= 10:
                    recent_high = minute_data.iloc[i-10:i]['High'].max()
                    recent_low = minute_data.iloc[i-10:i]['Low'].min()
                    current_price = minute_data.iloc[i]['Close']
                    
                    if current_price > recent_high:
                        breakout_type = 'resistance_break'
                        breakout_strength = (current_price - recent_high) / recent_high
                    elif current_price  0 else 1
                else:
                    sustainability = volume_ratio / 2
                
                volume_spikes.append({
                    'index': i,
                    'volume_ratio': volume_ratio,
                    'price_change': price_change,
                    'vp_correlation': vp_correlation,
                    'breakout_type': breakout_type,
                    'breakout_strength': breakout_strength,
                    'sustainability': sustainability,
                    'price': minute_data.iloc[i]['Close']
                })
        
        if not volume_spikes:
            return {'detected': False, 'confidence': 0.0, 'vector': np.zeros(8)}
        
        # Take the most significant volume spike
        best_spike = max(volume_spikes, key=lambda x: x['volume_ratio'] * (1 + abs(x['price_change']) * 10))
        
        # Direction bias
        direction = 1 if best_spike['price_change'] > 0 else -1
        
        vector = np.array([
            min(best_spike['volume_ratio'], 5.0),
            abs(best_spike['price_change']) * 50,  # Scale price change
            min(best_spike['vp_correlation'], 3.0),
            min(abs(best_spike['breakout_strength']) * 20, 2.0),
            min(best_spike['sustainability'], 2.0),
            direction,
            1.0 if best_spike['breakout_type'] != 'none' else 0.0,
            (len(volume_data) - best_spike['index']) / 10.0  # Recency
        ], dtype=np.float32)
        
        confidence = (
            min(best_spike['volume_ratio'] / 3, 1.0) * 0.3 +
            min(abs(best_spike['price_change']) * 20, 1.0) * 0.3 +
            min(best_spike['sustainability'] / 2, 1.0) * 0.4
        )
        
        return {
            'detected': confidence > 0.6,
            'confidence': confidence,
            'vector': vector,
            'spike_volume_ratio': best_spike['volume_ratio'],
            'price_change': best_spike['price_change'],
            'breakout_type': best_spike['breakout_type'],
            'direction': direction
        }
    
    def _detect_consolidation_breakout(self, minute_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Detect consolidation breakout - tight range followed by volume breakout
        """
        if len(minute_data)  0 else 0
                
                # Tight consolidation criteria
                if range_size = start + length:
                        consol_volume = volume_data.iloc[start:start+length]['Volume'].mean()
                        pre_volume = volume_data.iloc[max(0, start-length):start]['Volume'].mean()
                        volume_ratio = consol_volume / pre_volume if pre_volume > 0 else 1
                    else:
                        volume_ratio = 1
                    
                    # Check for breakout after consolidation
                    if start + length  high:  # Upside breakout
                            breakout_strength = (breakout_price - high) / high
                            breakout_direction = 1
                        elif breakout_price  0):
                        breakout_volume = volume_data.iloc[start + length]['Volume']
                        breakout_vol_ratio = breakout_volume / consol_volume if consol_volume > 0 else 1
                    else:
                        breakout_vol_ratio = 1
                    
                    consolidations.append({
                        'start': start,
                        'length': length,
                        'range_size': range_size,
                        'high': high,
                        'low': low,
                        'volume_ratio': volume_ratio,
                        'breakout_strength': breakout_strength,
                        'breakout_direction': breakout_direction,
                        'breakout_vol_ratio': breakout_vol_ratio,
                        'quality': (1 - range_size) * (1 + breakout_strength) * min(breakout_vol_ratio, 2.0)
                    })
        
        if not consolidations:
            return {'detected': False, 'confidence': 0.0, 'vector': np.zeros(7)}
        
        # Take the best consolidation breakout
        best_consolidation = max(consolidations, key=lambda x: x['quality'])
        
        vector = np.array([
            max(0, (0.05 - best_consolidation['range_size']) / 0.05),  # Tightness score
            min(best_consolidation['breakout_strength'] * 20, 2.0),
            min(best_consolidation['breakout_vol_ratio'], 3.0),
            best_consolidation['breakout_direction'],
            best_consolidation['length'] / 15.0,  # Duration normalized
            1.0 if best_consolidation['breakout_strength'] > 0.01 else 0.0,
            min(best_consolidation['volume_ratio'], 2.0)
        ], dtype=np.float32)
        
        confidence = (
            max(0, (0.05 - best_consolidation['range_size']) / 0.05) * 0.3 +
            min(best_consolidation['breakout_strength'] * 10, 1.0) * 0.4 +
            min(best_consolidation['breakout_vol_ratio'] / 2, 1.0) * 0.3
        )
        
        return {
            'detected': confidence > 0.6 and best_consolidation['breakout_strength'] > 0.005,
            'confidence': confidence,
            'vector': vector,
            'consolidation_high': best_consolidation['high'],
            'consolidation_low': best_consolidation['low'],
            'breakout_direction': best_consolidation['breakout_direction'],
            'buy_point': best_consolidation['high'] * 1.005 if best_consolidation['breakout_direction'] == 1 else best_consolidation['low'] * 0.995
        }
    
    def _detect_level2_support_break(self, minute_data: pd.DataFrame, level2_data: pd.DataFrame) -> Dict:
        """
        Detect Level 2 support/resistance breaks using order book data
        """
        if level2_data is None or len(level2_data) == 0:
            return {'detected': False, 'confidence': 0.0, 'vector': np.zeros(6)}
        
        # Analyze current Level 2 data
        current_l2 = level2_data.iloc[-1] if len(level2_data) > 0 else None
        if current_l2 is None:
            return {'detected': False, 'confidence': 0.0, 'vector': np.zeros(6)}
        
        current_price = minute_data['Close'].iloc[-1]
        
        # Find significant support/resistance levels
        bid_levels = []
        ask_levels = []
        
        # Parse Level 2 data (assuming columns: bid_price_1, bid_size_1, ask_price_1, ask_size_1, etc.)
        for i in range(1, 6):  # Top 5 levels
            if f'bid_price_{i}' in current_l2 and f'bid_size_{i}' in current_l2:
                bid_price = current_l2[f'bid_price_{i}']
                bid_size = current_l2[f'bid_size_{i}']
                if bid_price > 0 and bid_size > 0:
                    bid_levels.append((bid_price, bid_size))
            
            if f'ask_price_{i}' in current_l2 and f'ask_size_{i}' in current_l2:
                ask_price = current_l2[f'ask_price_{i}']
                ask_size = current_l2[f'ask_size_{i}']
                if ask_price > 0 and ask_size > 0:
                    ask_levels.append((ask_price, ask_size))
        
        if not bid_levels and not ask_levels:
            return {'detected': False, 'confidence': 0.0, 'vector': np.zeros(6)}
        
        # Find heavy support/resistance
        significant_bid = max(bid_levels, key=lambda x: x[1]) if bid_levels else (0, 0)
        significant_ask = max(ask_levels, key=lambda x: x[1]) if ask_levels else (float('inf'), 0)
        
        # Check for breaks
        support_break = False
        resistance_break = False
        
        if significant_bid[0] > 0 and current_price  significant_ask[0] * 1.001:
            resistance_break = True
            break_strength = (current_price - significant_ask[0]) / significant_ask[0]
            break_type = 'resistance'
        else:
            break_strength = 0
            break_type = 'none'
        
        # Size of the broken level
        if support_break:
            level_size = significant_bid[1]
        elif resistance_break:
            level_size = significant_ask[1]
        else:
            level_size = 0
        
        # Normalized level size (relative to average)
        all_sizes = [level[1] for level in bid_levels + ask_levels]
        avg_size = np.mean(all_sizes) if all_sizes else 1
        size_significance = level_size / avg_size if avg_size > 0 else 1
        
        vector = np.array([
            1.0 if support_break or resistance_break else 0.0,
            min(break_strength * 100, 2.0),  # Scale break strength
            min(size_significance, 5.0),     # Size significance
            1.0 if break_type == 'resistance' else -1.0 if break_type == 'support' else 0.0,
            len(bid_levels) / 5.0,           # Depth of book
            len(ask_levels) / 5.0            # Ask depth
        ], dtype=np.float32)
        
        confidence = (
            (1.0 if support_break or resistance_break else 0.0) * 0.4 +
            min(break_strength * 20, 1.0) * 0.3 +
            min(size_significance / 3, 1.0) * 0.3
        )
        
        return {
            'detected': confidence > 0.5,
            'confidence': confidence,
            'vector': vector,
            'break_type': break_type,
            'broken_level': significant_bid[0] if support_break else significant_ask[0] if resistance_break else 0,
            'level_size': level_size
        }
    
    def _detect_iceberg_orders(self, level2_data: pd.DataFrame) -> Dict:
        """
        Detect iceberg orders (large hidden orders) from Level 2 patterns
        """
        if level2_data is None or len(level2_data)  prev_bid_size * 0.8 and
                        current_bid_size > 0):
                        
                        iceberg_signals.append({
                            'side': 'bid',
                            'price': current_bid_price,
                            'size': current_bid_size,
                            'level': level,
                            'strength': current_bid_size / (prev_bid_size + 1)
                        })
                
                if all(col in current.index for col in [ask_price_col, ask_size_col]):
                    # Ask side iceberg detection
                    current_ask_price = current[ask_price_col]
                    current_ask_size = current[ask_size_col]
                    prev_ask_price = previous.get(ask_price_col, float('inf'))
                    prev_ask_size = previous.get(ask_size_col, 0)
                    
                    # Iceberg pattern on ask side
                    if (current_ask_price == prev_ask_price and 
                        current_ask_size > prev_ask_size * 0.8 and
                        current_ask_size > 0):
                        
                        iceberg_signals.append({
                            'side': 'ask',
                            'price': current_ask_price,
                            'size': current_ask_size,
                            'level': level,
                            'strength': current_ask_size / (prev_ask_size + 1)
                        })
        
        if not iceberg_signals:
            return {'detected': False, 'confidence': 0.0, 'vector': np.zeros(5)}
        
        # Analyze the strongest iceberg signal
        strongest_iceberg = max(iceberg_signals, key=lambda x: x['strength'] * (4 - x['level']))
        
        # Count icebergs on each side
        bid_icebergs = len([s for s in iceberg_signals if s['side'] == 'bid'])
        ask_icebergs = len([s for s in iceberg_signals if s['side'] == 'ask'])
        
        # Market implication
        if bid_icebergs > ask_icebergs:
            market_bias = 1  # Bullish (strong bid support)
        elif ask_icebergs > bid_icebergs:
            market_bias = -1  # Bearish (strong ask resistance)
        else:
            market_bias = 0
        
        vector = np.array([
            1.0 if len(iceberg_signals) > 0 else 0.0,
            min(strongest_iceberg['strength'], 5.0),
            bid_icebergs / 5.0,
            ask_icebergs / 5.0,
            market_bias
        ], dtype=np.float32)
        
        confidence = min(strongest_iceberg['strength'] / 3.0, 1.0)
        
        return {
            'detected': confidence > 0.3,
            'confidence': confidence,
            'vector': vector,
            'strongest_side': strongest_iceberg['side'],
            'market_bias': market_bias,
            'iceberg_price': strongest_iceberg['price']
        }
    
    def _generate_buy_signals(self, patterns: Dict, minute_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive buy signals from detected patterns
        """
        buy_signals = []
        current_price = minute_data['Close'].iloc[-1]
        
        # Bull Flag Buy Signal
        if patterns.get('bull_flag', {}).get('detected', False):
            bull_flag = patterns['bull_flag']
            if current_price >= bull_flag['buy_point']:
                buy_signals.append({
                    'pattern': 'bull_flag',
                    'signal_strength': bull_flag['confidence'],
                    'entry_price': bull_flag['buy_point'],
                    'stop_loss': bull_flag['stop_loss'],
                    'target_1': bull_flag['profit_target_1'],
                    'target_2': bull_flag['profit_target_2'],
                    'risk_reward': (bull_flag['profit_target_1'] - bull_flag['buy_point']) / (bull_flag['buy_point'] - bull_flag['stop_loss']),
                    'timeframe': 'scalp'  # 5-15 minute hold
                })
        
        # Flat Top Breakout Signal
        if patterns.get('flat_top_breakout', {}).get('detected', False):
            flat_top = patterns['flat_top_breakout']
            if current_price >= flat_top['buy_point'] and flat_top.get('volume_surge', 1) > 1.5:
                buy_signals.append({
                    'pattern': 'flat_top_breakout',
                    'signal_strength': flat_top['confidence'],
                    'entry_price': flat_top['buy_point'],
                    'stop_loss': flat_top['stop_loss'],
                    'target_1': flat_top['buy_point'] * 1.15,
                    'target_2': flat_top['buy_point'] * 1.25,
                    'risk_reward': (flat_top['buy_point'] * 1.15 - flat_top['buy_point']) / (flat_top['buy_point'] - flat_top['stop_loss']),
                    'timeframe': 'momentum'  # 15-30 minute hold
                })
        
        # Gap and Go Signal
        if patterns.get('gap_and_go', {}).get('detected', False):
            gap_go = patterns['gap_and_go']
            if gap_go.get('volume_ratio', 1) > 2.0:
                buy_signals.append({
                    'pattern': 'gap_and_go',
                    'signal_strength': gap_go['confidence'],
                    'entry_price': gap_go['buy_point'],
                    'stop_loss': gap_go['stop_loss'],
                    'target_1': gap_go['buy_point'] * 1.20,
                    'target_2': gap_go['buy_point'] * 1.35,
                    'risk_reward': (gap_go['buy_point'] * 1.20 - gap_go['buy_point']) / (gap_go['buy_point'] - gap_go['stop_loss']),
                    'timeframe': 'swing'  # 30-60 minute hold
                })
        
        # Opening Range Breakout Signal
        if patterns.get('orb', {}).get('detected', False):
            orb = patterns['orb']
            if orb.get('direction', 0) == 1:  # Upside breakout only
                buy_signals.append({
                    'pattern': 'opening_range_breakout',
                    'signal_strength': orb['confidence'],
                    'entry_price': orb['buy_point'],
                    'stop_loss': orb['stop_loss'],
                    'target_1': orb['target_1'],
                    'target_2': orb['target_2'],
                    'risk_reward': (orb['target_1'] - orb['buy_point']) / (orb['buy_point'] - orb['stop_loss']),
                    'timeframe': 'scalp'
                })
        
        # Red to Green Signal
        if patterns.get('red_to_green', {}).get('detected', False):
            rtg = patterns['red_to_green']
            if rtg.get('volume_surge', 1) > 1.5:
                buy_signals.append({
                    'pattern': 'red_to_green',
                    'signal_strength': rtg['confidence'],
                    'entry_price': rtg['buy_point'],
                    'stop_loss': rtg['stop_loss'],
                    'target_1': rtg['buy_point'] * 1.10,
                    'target_2': rtg['buy_point'] * 1.18,
                    'risk_reward': (rtg['buy_point'] * 1.10 - rtg['buy_point']) / (rtg['buy_point'] - rtg['stop_loss']),
                    'timeframe': 'scalp'
                })
        
        # Hammer Reversal Signal
        if patterns.get('hammer_reversal', {}).get('detected', False):
            hammer = patterns['hammer_reversal']
            buy_signals.append({
                'pattern': 'hammer_reversal',
                'signal_strength': hammer['confidence'],
                'entry_price': hammer['buy_point'],
                'stop_loss': hammer['stop_loss'],
                'target_1': hammer['buy_point'] * 1.12,
                'target_2': hammer['buy_point'] * 1.20,
                'risk_reward': (hammer['buy_point'] * 1.12 - hammer['buy_point']) / (hammer['buy_point'] - hammer['stop_loss']),
                'timeframe': 'momentum'
            })
        
        # Volume Spike Breakout Signal
        if patterns.get('volume_spike', {}).get('detected', False):
            vol_spike = patterns['volume_spike']
            if vol_spike.get('direction', 0) == 1:  # Upside volume spike
                entry_price = current_price * 1.002  # Just above current
                stop_loss = current_price * 0.96     # 4% stop
                buy_signals.append({
                    'pattern': 'volume_spike_breakout',
                    'signal_strength': vol_spike['confidence'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target_1': entry_price * 1.08,
                    'target_2': entry_price * 1.15,
                    'risk_reward': (entry_price * 1.08 - entry_price) / (entry_price - stop_loss),
                    'timeframe': 'scalp'
                })
        
        # Sort by signal strength
        buy_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
        
        # Calculate combined signal strength
        if buy_signals:
            combined_strength = sum(signal['signal_strength'] for signal in buy_signals) / len(buy_signals)
            best_risk_reward = max(signal['risk_reward'] for signal in buy_signals if signal['risk_reward'] > 0)
        else:
            combined_strength = 0
            best_risk_reward = 0
        
        return {
            'signals': buy_signals,
            'signal_count': len(buy_signals),
            'combined_strength': combined_strength,
            'best_risk_reward': best_risk_reward,
            'recommended_action': 'BUY' if combined_strength > 0.6 and len(buy_signals) >= 2 else 'WAIT'
        }
    
    def _generate_sell_signals(self, patterns: Dict, minute_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Generate sell signals and profit-taking levels
        """
        sell_signals = []
        current_price = minute_data['Close'].iloc[-1]
        
        # Shooting Star Reversal
        if patterns.get('doji_reversal', {}).get('detected', False):
            doji = patterns['doji_reversal']
            if doji.get('reversal_direction') == 'bearish':
                sell_signals.append({
                    'pattern': 'shooting_star_reversal',
                    'signal_strength': doji['confidence'],
                    'sell_price': current_price * 0.998,  # Just below current
                    'stop_loss': current_price * 1.03,    # 3% above current
                    'target_1': current_price * 0.92,     # 8% drop target
                    'target_2': current_price * 0.85,     # 15% drop target
                    'timeframe': 'scalp'
                })
        
        # Volume Spike Downside
        if patterns.get('volume_spike', {}).get('detected', False):
            vol_spike = patterns['volume_spike']
            if vol_spike.get('direction', 0) == -1:  # Downside volume spike
                sell_signals.append({
                    'pattern': 'volume_spike_breakdown',
                    'signal_strength': vol_spike['confidence'],
                    'sell_price': current_price * 0.998,
                    'stop_loss': current_price * 1.04,
                    'target_1': current_price * 0.90,
                    'target_2': current_price * 0.82,
                    'timeframe': 'scalp'
                })
        
        # Level 2 Support Break
        if patterns.get('level2_support_break', {}).get('detected', False):
            l2_break = patterns['level2_support_break']
            if l2_break.get('break_type') == 'support':
                sell_signals.append({
                    'pattern': 'level2_support_break',
                    'signal_strength': l2_break['confidence'],
                    'sell_price': current_price * 0.999,
                    'stop_loss': l2_break['broken_level'] * 1.02,
                    'target_1': l2_break['broken_level'] * 0.95,
                    'target_2': l2_break['broken_level'] * 0.88,
                    'timeframe': 'scalp'
                })
        
        # Profit Taking Signals (for existing positions)
        profit_taking = self._generate_profit_taking_signals(patterns, minute_data, volume_data)
        
        return {
            'sell_signals': sell_signals,
            'profit_taking': profit_taking,
            'signal_count': len(sell_signals),
            'recommended_action': 'SELL' if len(sell_signals) >= 2 else 'HOLD'
        }
    
    def _generate_profit_taking_signals(self, patterns: Dict, minute_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Generate profit-taking and scaling-out signals
        """
        current_price = minute_data['Close'].iloc[-1]
        
        # Analyze recent price action for profit-taking opportunities
        if len(minute_data) >= 10:
            recent_high = minute_data['High'].iloc[-10:].max()
            recent_volume = volume_data['Volume'].iloc[-5:].mean() if len(volume_data) >= 5 else 0
            avg_volume = volume_data['Volume'].iloc[-20:].mean() if len(volume_data) >= 20 else 1
            
            # Volume divergence (price up, volume down = take profits)
            price_momentum = (current_price - minute_data['Close'].iloc[-10]) / minute_data['Close'].iloc[-10]
            volume_momentum = (recent_volume - avg_volume) / avg_volume if avg_volume > 0 else 0
            
            # Divergence signal
            if price_momentum > 0.05 and volume_momentum = recent_high * 0.98
            
            # Parabolic move detection (too much too fast)
            if len(minute_data) >= 15:
                parabolic_gain = (current_price - minute_data['Close'].iloc[-15]) / minute_data['Close'].iloc[-15]
                parabolic_move = parabolic_gain > 0.15  # 15% in 15 minutes
            else:
                parabolic_move = False
                parabolic_gain = 0
        else:
            divergence_signal = False
            divergence_strength = 0
            resistance_approach = False
            parabolic_move = False
            parabolic_gain = 0
        
        # Scaling recommendations
        scaling_plan = []
        
        if divergence_signal:
            scaling_plan.append({
                'trigger': 'volume_divergence',
                'action': 'scale_out_25%',
                'reason': 'Volume divergence detected',
                'urgency': 'medium'
            })
        
        if resistance_approach:
            scaling_plan.append({
                'trigger': 'resistance_approach',
                'action': 'scale_out_50%',
                'reason': 'Approaching resistance level',
                'urgency': 'high'
            })
        
        if parabolic_move:
            scaling_plan.append({
                'trigger': 'parabolic_move',
                'action': 'scale_out_75%',
                'reason': f'Parabolic gain of {parabolic_gain:.1%}',
                'urgency': 'urgent'
            })
        
        return {
            'divergence_signal': divergence_signal,
            'divergence_strength': divergence_strength,
            'resistance_approach': resistance_approach,
            'parabolic_move': parabolic_move,
            'scaling_plan': scaling_plan,
            'recommended_action': scaling_plan[0]['action'] if scaling_plan else 'hold_position'
        }
    
    def _calculate_risk_management_levels(self, minute_data: pd.DataFrame, patterns: Dict) -> Dict:
        """
        Calculate dynamic risk management levels based on detected patterns
        """
        current_price = minute_data['Close'].iloc[-1]
        
        # ATR-based stops (using minute data)
        if len(minute_data) >= 20:
            # Calculate True Range for each minute
            tr_values = []
            for i in range(1, len(minute_data)):
                high = minute_data['High'].iloc[i]
                low = minute_data['Low'].iloc[i]
                prev_close = minute_data['Close'].iloc[i-1]
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_values.append(tr)
            
            atr_20 = np.mean(tr_values[-20:]) if len(tr_values) >= 20 else np.mean(tr_values)
            atr_10 = np.mean(tr_values[-10:]) if len(tr_values) >= 10 else atr_20
        else:
            atr_20 = current_price * 0.02  # Default 2% ATR
            atr_10 = atr_20
        
        # Pattern-specific risk levels
        pattern_stops = {}
        
        for pattern_name, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and pattern_data.get('detected', False):
                if 'stop_loss' in pattern_data:
                    pattern_stops[pattern_name] = pattern_data['stop_loss']
        
        # Calculate dynamic stop levels
        atr_stop_tight = current_price - (atr_10 * 1.5)  # Tight stop (1.5 ATR)
        atr_stop_normal = current_price - (atr_20 * 2.0)  # Normal stop (2 ATR)
        atr_stop_wide = current_price - (atr_20 * 3.0)    # Wide stop (3 ATR)
        
        # Pattern-based stop (average of pattern stops)
        if pattern_stops:
            pattern_stop = np.mean(list(pattern_stops.values()))
        else:
            pattern_stop = current_price * 0.95  # Default 5% stop
        
        # Trailing stop calculations
        if len(minute_data) >= 10:
            recent_high = minute_data['High'].iloc[-10:].max()
            trailing_stop_3pct = recent_high * 0.97    # 3% trailing
            trailing_stop_5pct = recent_high * 0.95    # 5% trailing
            trailing_stop_atr = recent_high - (atr_10 * 2)  # ATR trailing
        else:
            trailing_stop_3pct = current_price * 0.97
            trailing_stop_5pct = current_price * 0.95
            trailing_stop_atr = current_price - (atr_10 * 2)
        
        # Position sizing based on risk
        account_risk_2pct = self.risk_management['max_risk_per_trade']  # 2% account risk
        
        risk_per_share_tight = current_price - atr_stop_tight
        risk_per_share_normal = current_price - atr_stop_normal
        risk_per_share_pattern = current_price - pattern_stop
        
        # Calculate position sizes (assuming $100k account for example)
        account_size = 100000  # This should be passed as parameter
        
        position_size_tight = (account_size * account_risk_2pct) / risk_per_share_tight if risk_per_share_tight > 0 else 0
        position_size_normal = (account_size * account_risk_2pct) / risk_per_share_normal if risk_per_share_normal > 0 else 0
        position_size_pattern = (account_size * account_risk_2pct) / risk_per_share_pattern if risk_per_share_pattern > 0 else 0
        
        # Profit targets
        profit_target_1 = current_price * 1.10  # 10% profit
        profit_target_2 = current_price * 1.20  # 20% profit
        profit_target_atr = current_price + (atr_20 * 2)  # 2 ATR profit target
        
        return {
            'stop_levels': {
                'atr_tight': atr_stop_tight,
                'atr_normal': atr_stop_normal,
                'atr_wide': atr_stop_wide,
                'pattern_based': pattern_stop,
                'recommended': max(atr_stop_normal, pattern_stop)  # More conservative of the two
            },
            'trailing_stops': {
                '3_percent': trailing_stop_3pct,
                '5_percent': trailing_stop_5pct,
                'atr_based': trailing_stop_atr,
                'recommended': trailing_stop_atr
            },
            'position_sizing': {
                'tight_stop': int(position_size_tight),
                'normal_stop': int(position_size_normal),
                'pattern_stop': int(position_size_pattern),
                'recommended': int(position_size_normal)
            },
            'profit_targets': {
                'target_1': profit_target_1,
                'target_2': profit_target_2,
                'atr_target': profit_target_atr,
                'scale_out_plan': {
                    '25%_at': current_price * 1.05,  # Take 25% at 5%
                    '50%_at': current_price * 1.10,  # Take 50% at 10%
                    '25%_runner': 'trail_with_atr'    # Let 25% run with ATR trail
                }
            },
            'atr_values': {
                'atr_10': atr_10,
                'atr_20': atr_20,
                'atr_percentage': (atr_20 / current_price) * 100
            }
        }


# Warrior Trading Specific Entry/Exit Rules
class WarriorTradingRules:
    """
    Specific entry and exit rules used by Cameron Ross and Warrior Trading
    """
    
    @staticmethod
    def validate_entry_conditions(minute_data: pd.DataFrame, volume_data: pd.DataFrame, patterns: Dict) -> Dict:
        """
        Validate entry conditions before taking a trade
        """
        current_price = minute_data['Close'].iloc[-1]
        
        # Pre-market gap check
        open_price = minute_data['Open'].iloc[0]
        gap_size = (open_price - current_price * 0.95) / (current_price * 0.95)  # Approximation
        
        # Volume validation
        if len(volume_data) >= 20:
            current_volume = volume_data['Volume'].iloc[-1]
            avg_volume = volume_data['Volume'].iloc[-20:].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        else:
            volume_ratio = 1
        
        # Price validation
        price_valid = current_price >= 2.00  # Above $2
        
        # Spread check (approximate)
        if len(minute_data) >= 2:
            spread_approx = (minute_data['High'].iloc[-1] - minute_data['Low'].iloc[-1]) / minute_data['Close'].iloc[-1]
            spread_valid = spread_approx = 0.6
        
        # News validation (would need news feed)
        news_valid = True  # Assume positive/neutral news
        
        validation_results = {
            'price_valid': price_valid,
            'volume_valid': volume_ratio >= 1.5,
            'spread_valid': spread_valid,
            'float_valid': float_valid,
            'market_cap_valid': market_cap_valid,
            'time_valid': time_valid,
            'pattern_valid': pattern_valid,
            'news_valid': news_valid,
            'gap_size': gap_size,
            'volume_ratio': volume_ratio,
            'pattern_strength': pattern_strength
        }
        
        # Overall validation
        validation_results['overall_valid'] = all([
            price_valid, validation_results['volume_valid'], spread_valid,
            time_valid, pattern_valid, news_valid
        ])
        
        return validation_results
    
    @staticmethod
    def calculate_position_size(account_size: float, current_price: float, stop_loss: float, max_risk_pct: float = 0.02) -> Dict:
        """
        Calculate position size based on account risk
        """
        risk_per_share = current_price - stop_loss
        
        if risk_per_share  Dict:
        """
        Generate exit strategy based on pattern and timeframe
        """
        
        # Timeframe-specific targets
        if timeframe == 'scalp':  # 2-10 minute holds
            profit_targets = [1.05, 1.08, 1.12]  # 5%, 8%, 12%
            stop_loss_pct = 0.03  # 3% stop
            scale_out_levels = [0.5, 0.3, 0.2]  # 50%, 30%, 20%
            
        elif timeframe == 'momentum':  # 10-30 minute holds
            profit_targets = [1.10, 1.15, 1.25]  # 10%, 15%, 25%
            stop_loss_pct = 0.05  # 5% stop
            scale_out_levels = [0.4, 0.4, 0.2]
            
        elif timeframe == 'swing':  # 30-60 minute holds
            profit_targets = [1.15, 1.25, 1.40]  # 15%, 25%, 40%
            stop_loss_pct = 0.07  # 7% stop
            scale_out_levels = [0.3, 0.4, 0.3]
            
        else:  # Default scalp
            profit_targets = [1.05, 1.08, 1.12]
            stop_loss_pct = 0.03
            scale_out_levels = [0.5, 0.3, 0.2]
        
        exit_strategy = {
            'stop_loss': entry_price * (1 - stop_loss_pct),
            'profit_targets': [entry_price * target for target in profit_targets],
            'scale_out_plan': [
                {
                    'target_price': entry_price * profit_targets[i],
                    'percentage_to_sell': scale_out_levels[i],
                    'action': f'Sell {int(scale_out_levels[i]*100)}% at {profit_targets[i]*100-100:.0f}% profit'
                }
                for i in range(len(profit_targets))
            ],
            'timeframe': timeframe,
            'max_hold_time_minutes': {
                'scalp': 10,
                'momentum': 30,
                'swing': 60
            }.get(timeframe, 10),
            'trailing_stop_trigger': entry_price * profit_targets[0],  # Start trailing after first target
            'trailing_stop_distance': entry_price * 0.03  # 3% trailing distance
        }
        
        return exit_strategy