import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CANSLIMPatternGenerator:
    """
    William O'Neil's CANSLIM pattern recognition system
    C - Current Quarterly Earnings
    A - Annual Earnings Growth  
    N - New Products/Management/Highs
    S - Supply and Demand (Shares Outstanding)
    L - Leader or Laggard
    I - Institutional Sponsorship
    M - Market Direction
    """
    
    def __init__(self):
        self.pattern_weights = {
            'cup_with_handle': 0.25,
            'flat_base': 0.20,
            'high_tight_flag': 0.20,
            'double_bottom': 0.15,
            'ascending_base': 0.10,
            'pocket_pivot': 0.10
        }
    
    def generate_canslim_patterns(self, 
                                 price_data: pd.DataFrame,
                                 volume_data: pd.DataFrame,
                                 earnings_data: pd.DataFrame,
                                 institutional_data: pd.DataFrame,
                                 market_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive CANSLIM pattern vectors
        """
        patterns = {}
        
        # C - Current Quarterly Earnings Patterns
        patterns['current_earnings'] = self._generate_earnings_patterns(earnings_data)
        
        # A - Annual Earnings Growth Patterns
        patterns['annual_growth'] = self._generate_annual_growth_patterns(earnings_data)
        
        # N - New High Patterns
        patterns['new_highs'] = self._generate_new_high_patterns(price_data, volume_data)
        
        # S - Supply/Demand Patterns
        patterns['supply_demand'] = self._generate_supply_demand_patterns(price_data, volume_data)
        
        # L - Leadership Patterns
        patterns['leadership'] = self._generate_leadership_patterns(price_data, market_data)
        
        # I - Institutional Patterns
        patterns['institutional'] = self._generate_institutional_patterns(institutional_data, volume_data)
        
        # M - Market Direction Patterns
        patterns['market_direction'] = self._generate_market_direction_patterns(market_data)
        
        # Base Pattern Recognition
        patterns['base_patterns'] = self._generate_base_patterns(price_data, volume_data)
        
        return patterns
    
    def _generate_earnings_patterns(self, earnings_data: pd.DataFrame) -> np.ndarray:
        """
        C - Current quarterly earnings acceleration patterns
        Look for 18-25% minimum growth, ideally accelerating
        """
        features = []
        
        if 'eps_growth' in earnings_data.columns:
            # Quarterly EPS growth rate
            current_growth = earnings_data['eps_growth'].iloc[-1] if len(earnings_data) > 0 else 0
            
            # Acceleration pattern (last 3 quarters)
            if len(earnings_data) >= 3:
                recent_growth = earnings_data['eps_growth'].iloc[-3:].values
                acceleration = np.diff(recent_growth).mean()  # Average acceleration
                consistency = 1.0 - np.std(recent_growth) / (np.mean(recent_growth) + 1e-8)
            else:
                acceleration = 0
                consistency = 0
            
            # Surprise factor
            surprise = earnings_data.get('eps_surprise', pd.Series([0])).iloc[-1]
            
            features.extend([
                min(current_growth / 25.0, 3.0),  # Normalized growth (cap at 75%)
                acceleration / 10.0,  # Acceleration factor
                consistency,
                surprise / 20.0  # Surprise factor
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def _generate_annual_growth_patterns(self, earnings_data: pd.DataFrame) -> np.ndarray:
        """
        A - Annual earnings growth patterns (25% minimum over 3 years)
        """
        features = []
        
        if 'annual_eps_growth' in earnings_data.columns and len(earnings_data) >= 3:
            annual_growth = earnings_data['annual_eps_growth'].iloc[-3:].values
            
            # Growth consistency over 3 years
            avg_growth = np.mean(annual_growth)
            growth_consistency = 1.0 - np.std(annual_growth) / (avg_growth + 1e-8)
            
            # ROE patterns
            roe = earnings_data.get('roe', pd.Series([0])).iloc[-1]
            
            features.extend([
                min(avg_growth / 25.0, 2.0),  # Normalized average growth
                growth_consistency,
                min(roe / 17.0, 2.0),  # ROE should be 17%+
                1.0 if avg_growth >= 25.0 else avg_growth / 25.0  # Binary + scaled
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def _generate_new_high_patterns(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> np.ndarray:
        """
        N - New product, management, or price high patterns
        """
        features = []
        
        # New 52-week highs with volume
        if len(price_data) >= 252:  # 1 year of data
            recent_high = price_data['Close'].iloc[-1]
            year_high = price_data['Close'].iloc[-252:].max()
            is_new_high = 1.0 if recent_high >= year_high * 0.99 else 0.0
            
            # Volume confirmation on new highs
            if is_new_high > 0:
                recent_volume = volume_data['Volume'].iloc[-5:].mean()
                avg_volume = volume_data['Volume'].iloc[-50:].mean()
                volume_confirmation = min(recent_volume / avg_volume, 3.0)
            else:
                volume_confirmation = 0.0
        else:
            is_new_high = 0.0
            volume_confirmation = 0.0
        
        # Breakout strength
        if len(price_data) >= 50:
            breakout_level = price_data['Close'].iloc[-50:-10].max()
            current_price = price_data['Close'].iloc[-1]
            breakout_strength = max(0, (current_price - breakout_level) / breakout_level)
        else:
            breakout_strength = 0.0
        
        features.extend([
            is_new_high,
            min(volume_confirmation, 2.0),
            min(breakout_strength * 10, 2.0),  # Scale breakout strength
            1.0 if breakout_strength > 0.02 else 0.0  # 2%+ breakout
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _generate_supply_demand_patterns(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> np.ndarray:
        """
        S - Supply and demand patterns (tight float, low institutional ownership initially)
        """
        features = []
        
        # Volume dry-up patterns (decreasing volume in consolidation)
        if len(volume_data) >= 30:
            recent_volume = volume_data['Volume'].iloc[-10:].mean()
            earlier_volume = volume_data['Volume'].iloc[-30:-20].mean()
            volume_dryup = max(0, (earlier_volume - recent_volume) / earlier_volume)
        else:
            volume_dryup = 0
        
        # Price tightness (low volatility in consolidation)
        if len(price_data) >= 20:
            price_range = price_data['High'].iloc[-20:] - price_data['Low'].iloc[-20:]
            avg_range = price_range.mean()
            recent_range = price_range.iloc[-5:].mean()
            tightness = max(0, (avg_range - recent_range) / avg_range) if avg_range > 0 else 0
        else:
            tightness = 0
        
        # Accumulation pattern (price holding up on light volume)
        if len(price_data) >= 20:
            price_change = (price_data['Close'].iloc[-1] - price_data['Close'].iloc[-20]) / price_data['Close'].iloc[-20]
            volume_ratio = volume_data['Volume'].iloc[-20:].mean() / volume_data['Volume'].iloc[-60:-40].mean()
            accumulation = price_change / (volume_ratio + 0.5)  # Good if price up on lower volume
        else:
            accumulation = 0
        
        features.extend([
            min(volume_dryup * 2, 1.0),
            min(tightness * 3, 1.0), 
            min(accumulation * 5, 2.0),
            1.0 if volume_dryup > 0.2 and tightness > 0.15 else 0.0
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _generate_leadership_patterns(self, price_data: pd.DataFrame, market_data: pd.DataFrame) -> np.ndarray:
        """
        L - Leadership patterns (outperforming market)
        """
        features = []
        
        if len(price_data) >= 50 and len(market_data) >= 50:
            # Relative strength vs market
            stock_return = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[-50] - 1)
            market_return = (market_data['Close'].iloc[-1] / market_data['Close'].iloc[-50] - 1)
            relative_strength = stock_return - market_return
            
            # Consistency of outperformance
            stock_weekly = price_data['Close'].pct_change(5).iloc[-10:]
            market_weekly = market_data['Close'].pct_change(5).iloc[-10:]
            outperform_weeks = (stock_weekly > market_weekly).sum()
            outperform_ratio = outperform_weeks / 10.0
            
        else:
            relative_strength = 0
            outperform_ratio = 0.5
        
        features.extend([
            min(relative_strength * 2, 1.5),  # Cap relative strength
            outperform_ratio,
            1.0 if relative_strength > 0.1 else 0.0,  # Strong outperformance
            1.0 if outperform_ratio > 0.7 else 0.0  # Consistent winner
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _generate_institutional_patterns(self, institutional_data: pd.DataFrame, volume_data: pd.DataFrame) -> np.ndarray:
        """
        I - Institutional sponsorship patterns
        """
        features = []
        
        # Institutional ownership changes
        if 'institutional_ownership' in institutional_data.columns and len(institutional_data) >= 2:
            current_inst = institutional_data['institutional_ownership'].iloc[-1]
            previous_inst = institutional_data['institutional_ownership'].iloc[-2]
            inst_change = (current_inst - previous_inst) / previous_inst if previous_inst > 0 else 0
            
            # Number of new institutional positions
            new_positions = institutional_data.get('new_positions', pd.Series([0])).iloc[-1]
            
        else:
            inst_change = 0
            new_positions = 0
            current_inst = 0
        
        # Volume patterns suggesting institutional activity
        if len(volume_data) >= 50:
            recent_volume = volume_data['Volume'].iloc[-20:].mean()
            base_volume = volume_data['Volume'].iloc[-50:-30].mean()
            volume_increase = (recent_volume / base_volume - 1) if base_volume > 0 else 0
        else:
            volume_increase = 0
        
        features.extend([
            min(inst_change * 5, 1.0),  # Institutional buying
            min(new_positions / 5.0, 1.0),  # New institutional interest
            min(volume_increase, 2.0),  # Volume increase
            1.0 if inst_change > 0.05 and volume_increase > 0.5 else 0.0  # Strong institutional buying
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _generate_market_direction_patterns(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        M - Market direction patterns
        """
        features = []
        
        if len(market_data) >= 50:
            # Market trend (above/below key moving averages)
            sma_21 = market_data['Close'].rolling(21).mean().iloc[-1]
            sma_50 = market_data['Close'].rolling(50).mean().iloc[-1]
            current_price = market_data['Close'].iloc[-1]
            
            above_21sma = 1.0 if current_price > sma_21 else 0.0
            above_50sma = 1.0 if current_price > sma_50 else 0.0
            
            # Market momentum
            momentum_10d = (current_price / market_data['Close'].iloc[-10] - 1)
            
            # Distribution days (heavy volume down days)
            recent_data = market_data.iloc[-25:]  # Last 25 days
            down_days = recent_data[recent_data['Close']  0:
                if 'Volume' in recent_data.columns:
                    # Use true distribution day definition when volume is available
                    volume_avg = recent_data['Volume'].mean()
                    heavy_volume_down = len(down_days[down_days['Volume'] > volume_avg * 1.2])
                    distribution_days = heavy_volume_down / max(len(recent_data), 1)
                else:
                    # Fallback: price-only proxy (count down days, scaled conservatively)
                    distribution_days = (len(down_days) / max(len(recent_data), 1)) * 0.5
            else:
                distribution_days = 0
            
        else:
            above_21sma = above_50sma = 0.5
            momentum_10d = 0
            distribution_days = 0
        
        features.extend([
            above_21sma,
            above_50sma, 
            min(momentum_10d * 5, 1.0),  # Scale momentum
            max(0, 1.0 - distribution_days * 2)  # Fewer distribution days = better
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _generate_base_patterns(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Generate specific CANSLIM base patterns
        """
        patterns = {}
        
        patterns['cup_with_handle'] = self._detect_cup_with_handle(price_data, volume_data)
        patterns['flat_base'] = self._detect_flat_base(price_data, volume_data)
        patterns['high_tight_flag'] = self._detect_high_tight_flag(price_data, volume_data)
        patterns['double_bottom'] = self._detect_double_bottom(price_data, volume_data)
        
        return patterns
    
    def _detect_cup_with_handle(self, price_data: pd.DataFrame, volume_data: pd.DataFrame, 
                               min_weeks=7, max_weeks=65) -> Dict:
        """
        Detect Cup with Handle pattern - O'Neil's favorite pattern
        """
        if len(price_data)  left_rim_idx else (lows[1] if len(lows) > 1 else lows[0])
        
        left_rim_price = window['High'].iloc[left_rim_idx]
        cup_bottom_price = window['Low'].iloc[cup_bottom_idx]
        current_price = window['Close'].iloc[-1]
        
        # Cup depth (should be 12-33%)
        cup_depth = (left_rim_price - cup_bottom_price) / left_rim_price
        depth_score = 1.0 if 0.12 = lookback:
            vol_window = volume_data.iloc[-lookback:]
            cup_volume = vol_window.iloc[left_rim_idx:cup_bottom_idx]['Volume'].mean()
            handle_volume = vol_window.iloc[handle_start:]['Volume'].mean() if handle_start  0 else 0
            breakout_volume = recent_volume / handle_volume if handle_volume > 0 else 1
        else:
            volume_dryup = 0
            breakout_volume = 1
        
        # Pattern vector
        vector = np.array([
            depth_score,
            handle_score,
            min(volume_dryup * 2, 1.0),
            min(breakout_volume, 3.0),
            1.0 if current_price >= left_rim_price * 0.98 else 0.0,  # Near breakout
            cup_depth,
            handle_depth if len(handle_data) >= 5 else 0,
            (current_price - cup_bottom_price) / (left_rim_price - cup_bottom_price)  # Recovery level
        ], dtype=np.float32)
        
        confidence = (depth_score + handle_score + min(volume_dryup, 1.0)) / 3.0
        
        return {
            'detected': confidence > 0.5,
            'confidence': confidence,
            'vector': vector,
            'left_rim_price': left_rim_price,
            'cup_bottom_price': cup_bottom_price,
            'buy_point': left_rim_price * 1.02  # 2% above left rim
        }
    
    def _detect_flat_base(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Detect Flat Base pattern (15% max depth, tight sideways action)
        """
        if len(price_data) = 50 else price_data
        
        # Check for prior uptrend
        if len(window) >= 30:
            uptrend_gain = (window['Close'].iloc[15] - window['Close'].iloc[0]) / window['Close'].iloc[0]
            uptrend_score = min(uptrend_gain * 2, 1.0) if uptrend_gain > 0.1 else 0
        else:
            uptrend_score = 0
        
        # Flat base characteristics
        base_window = window.iloc[-25:]  # Last 5 weeks for base
        base_high = base_window['High'].max()
        base_low = base_window['Low'].min()
        base_depth = (base_high - base_low) / base_high
        
        # Should be tight (max 15% depth)
        tightness_score = max(0, (0.15 - base_depth) / 0.15) if base_depth = 50:
            vol_window = volume_data.iloc[-50:]
            early_volume = vol_window.iloc[:25]['Volume'].mean()
            base_volume = vol_window.iloc[-25:]['Volume'].mean()
            volume_decline = (early_volume - base_volume) / early_volume if early_volume > 0 else 0
        else:
            volume_decline = 0
        
        vector = np.array([
            uptrend_score,
            tightness_score,
            min(volume_decline * 2, 1.0),
            base_depth,
            1.0 if base_depth  0.5 and base_depth  Dict:
        """
        Detect High Tight Flag (3-5 weeks tight consolidation after 100%+ move)
        """
        if len(price_data) = 60:
            vol_window = volume_data.iloc[-60:]
            breakout_volume = vol_window.iloc[move_start:move_end]['Volume'].mean()
            flag_volume = vol_window.iloc[-15:]['Volume'].mean()
            volume_dryup = (breakout_volume - flag_volume) / breakout_volume if breakout_volume > 0 else 0
        else:
            volume_dryup = 0
        
        vector = np.array([
            min(prior_gain, 3.0),  # Prior move strength
            1.0 if prior_gain >= 1.0 else prior_gain,  # 100%+ move check
            max(0, (0.25 - flag_tightness) / 0.25),  # Tightness score
            min(volume_dryup * 1.5, 1.0),
            flag_tightness,
            prior_gain,
            len(flag_window) / 15.0  # Duration
        ], dtype=np.float32)
        
        confidence = 0.0
        if prior_gain >= 1.0 and flag_tightness  0.3:
            confidence = (min(prior_gain / 2, 1.0) + max(0, (0.25 - flag_tightness) / 0.25) + min(volume_dryup, 1.0)) / 3.0
        
        return {
            'detected': confidence > 0.6,
            'confidence': confidence, 
            'vector': vector
        }
    
    def _detect_double_bottom(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Detect Double Bottom with Handle pattern
        """
        if len(price_data) = 70 else price_data
        
        # Find two distinct bottoms
        lows = argrelextrema(window['Low'].values, np.less, order=5)[0]
        
        if len(lows)  0:
            peak_price = between_data['High'].max()
            rebound_strength = (peak_price - first_low_price) / first_low_price
        else:
            rebound_strength = 0
        
        # Current position vs second low
        current_price = window['Close'].iloc[-1]
        recovery_level = (current_price - second_low_price) / (peak_price - second_low_price) if peak_price > second_low_price else 0
        
        vector = np.array([
            similarity_score,
            min(rebound_strength * 2, 1.0),
            min(recovery_level * 2, 1.0),
            bottom_similarity,
            rebound_strength,
            1.0 if current_price > peak_price * 0.95 else 0.0  # Near breakout
        ], dtype=np.float32)
        
        confidence = (similarity_score + min(rebound_strength, 1.0) + min(recovery_level, 1.0)) / 3.0
        
        return {
            'detected': confidence > 0.5 and bottom_similarity > 0.96,
            'confidence': confidence,
            'vector': vector
        }


class SEPAPatternGenerator:
    """
    Mark Minervini's SEPA (Specific Entry Point Analysis) system
    Focus on Stage 2 uptrends, VCP, and specific entry techniques
    """
    
    def __init__(self):
        self.stage_weights = {
            'stage_2_characteristics': 0.30,
            'vcp_pattern': 0.25,
            'pocket_pivot': 0.20,
            'relative_strength': 0.15,
            'fundamentals_check': 0.10
        }
    
    def generate_sepa_patterns(self,
                              price_data: pd.DataFrame,
                              volume_data: pd.DataFrame,
                              market_data: pd.DataFrame,
                              fundamentals: pd.DataFrame = None) -> Dict:
        """
        Generate SEPA pattern vectors based on Minervini's methodology
        """
        patterns = {}
        
        # Stage Analysis
        patterns['stage_analysis'] = self._generate_stage_patterns(price_data, volume_data)
        
        # VCP (Volatility Contraction Pattern)
        patterns['vcp'] = self._detect_vcp_pattern(price_data, volume_data)
        
        # Pocket Pivots
        patterns['pocket_pivots'] = self._detect_pocket_pivots(price_data, volume_data, market_data)
        
        # Relative Strength Analysis
        patterns['relative_strength'] = self._generate_relative_strength_patterns(price_data, market_data)
        
        # Moving Average Alignment
        patterns['ma_alignment'] = self._generate_ma_alignment_patterns(price_data)
        
        # Fundamentals Integration
        if fundamentals is not None:
            patterns['fundamentals'] = self._generate_fundamental_patterns(fundamentals)
        
        return patterns
    
    def _generate_stage_patterns(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Generate Stage 2 uptrend characteristics
        Stage 1: Base building
        Stage 2: Advancing (target stage)
        Stage 3: Topping
        Stage 4: Declining
        """
        if len(price_data)  sma_150.iloc[-1],  # Above 150-day SMA
            current_price > sma_200.iloc[-1],  # Above 200-day SMA
            sma_150.iloc[-1] > sma_200.iloc[-1],  # 150 SMA > 200 SMA
            sma_200.iloc[-1] > sma_200.iloc[-30],  # 200 SMA trending up (vs 30 days ago)
            current_price > sma_50.iloc[-1],  # Above 50-day SMA
            current_price > sma_21.iloc[-1],  # Above 21-day SMA
            sma_21.iloc[-1] > sma_50.iloc[-1],  # 21 SMA > 50 SMA
            current_price >= price_data['Close'].iloc[-252:].min() * 1.25,  # 25% above 52-week low
        ]
        
        stage_2_score = sum(stage_2_conditions) / len(stage_2_conditions)
        
        # Price relative to 52-week high (should be within 25% for Stage 2)
        week_52_high = price_data['Close'].iloc[-252:].max()
        price_from_high = (week_52_high - current_price) / week_52_high
        proximity_to_high = max(0, (0.25 - price_from_high) / 0.25) if price_from_high  0
        if len(volume_data) >= len(up_days) and len(up_days) >= 20:
            # Use the last 20 days and align indices properly
            recent_up = up_days.iloc[-20:].reset_index(drop=True)
            recent_volume = volume_data['Volume'].iloc[-20:].reset_index(drop=True)
            
            up_volume = recent_volume[recent_up].mean() if recent_up.any() else recent_volume.mean()
            down_volume = recent_volume[~recent_up].mean() if (~recent_up).any() else recent_volume.mean()
            volume_ratio = up_volume / down_volume if down_volume > 0 else 1
        else:
            volume_ratio = 1
        
        # Determine current stage
        if stage_2_score >= 0.75:
            current_stage = 2
            stage_confidence = stage_2_score
        elif current_price  week_52_high * 0.9 and stage_2_score  0 else 0,
            (sma_150.iloc[-1] / sma_200.iloc[-1] - 1) if sma_200.iloc[-1] > 0 else 0,
            (sma_21.iloc[-1] / sma_50.iloc[-1] - 1) if sma_50.iloc[-1] > 0 else 0,
            price_from_high,
            current_stage / 4.0,
            stage_confidence
        ], dtype=np.float32)
        
        return {
            'stage': current_stage,
            'confidence': stage_confidence,
            'vector': vector,
            'stage_2_score': stage_2_score
        }
    
    def _detect_vcp_pattern(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict:
        """
        Detect Volatility Contraction Pattern - contractions getting tighter over time
        """
        if len(price_data) = 3:  # Minimum contraction length
                    contractions.append(current_contraction)
                current_contraction = []
        
        if len(current_contraction) >= 3:
            contractions.append(current_contraction)
        
        if len(contractions) = len(window):
                vol_window = volume_data.iloc[-len(window):]
                contraction_vol = vol_window.iloc[start_idx:end_idx+1]['Volume'].mean()
                base_vol = vol_window.iloc[max(0, start_idx-10):start_idx]['Volume'].mean()
                vol_ratio = contraction_vol / base_vol if base_vol > 0 else 1
                contraction_volumes.append(vol_ratio)
        
        # VCP characteristics: Each contraction should be shallower than the last
        if len(contraction_depths) >= 2:
            contracting_properly = all(contraction_depths[i] > contraction_depths[i+1] 
                                     for i in range(len(contraction_depths)-1))
            depth_ratio = contraction_depths[-1] / contraction_depths[0] if contraction_depths[0] > 0 else 1
        else:
            contracting_properly = False
            depth_ratio = 1
        
        # Volume should decrease with each contraction
        if len(contraction_volumes) >= 2:
            volume_contracting = all(contraction_volumes[i] > contraction_volumes[i+1] 
                                   for i in range(len(contraction_volumes)-1))
            volume_ratio = contraction_volumes[-1] / contraction_volumes[0] if contraction_volumes[0] > 0 else 1
        else:
            volume_contracting = False
            volume_ratio = 1
        
        # Final contraction characteristics
        final_contraction_depth = contraction_depths[-1] if contraction_depths else 0.5
        tightness_score = max(0, (0.15 - final_contraction_depth) / 0.15)  # Should be = 2 and contracting_properly and final_contraction_depth = 3 else 0.7)) / 3.0
        
        return {
            'detected': confidence > 0.6,
            'confidence': confidence,
            'vector': vector,
            'contractions': len(contractions),
            'final_depth': final_contraction_depth
        }
    
    def _detect_pocket_pivots(self, price_data: pd.DataFrame, volume_data: pd.DataFrame, 
                             market_data: pd.DataFrame) -> Dict:
        """
        Detect Pocket Pivot points - buying opportunities during pullbacks in uptrends
        """
        if len(price_data) = len(volume_data)):
                continue
            
            current_volume = volume_data['Volume'].iloc[current_idx]
            
            # Volume must exceed all down days in last 10 sessions
            down_days_volume = []
            for j in range(max(0, current_idx-10), current_idx):
                if (j  all down day volumes
            if current_volume  0 else 1
            
            # Market should not be in heavy distribution
            market_healthy = True
            if len(market_data) > current_idx:
                market_change = (market_data['Close'].iloc[current_idx] - 
                               market_data['Close'].iloc[current_idx-1]) / market_data['Close'].iloc[current_idx-1]
                if market_change 2%
                    market_healthy = False
            
            pivot_quality = min(volume_ratio / 2.0, 2.0) * (1.0 if price_vs_sma >= 0.98 else 0.5) * (1.0 if market_healthy else 0.3)
            
            pocket_pivots.append({
                'index': current_idx,
                'volume_ratio': volume_ratio,
                'price_vs_sma': price_vs_sma,
                'market_healthy': market_healthy,
                'quality': pivot_quality
            })
        
        # Analyze pocket pivots
        if not pocket_pivots:
            return {'detected': False, 'confidence': 0.0, 'vector': np.zeros(7)}
        
        # Most recent pivot
        recent_pivot = max(pocket_pivots, key=lambda x: x['index'])
        
        # Count quality pivots in recent period
        quality_pivots = [p for p in pocket_pivots if p['quality'] > 1.0]
        
        vector = np.array([
            len(pocket_pivots) / 5.0,  # Number of pocket pivots
            len(quality_pivots) / max(len(pocket_pivots), 1),  # Quality ratio
            min(recent_pivot['volume_ratio'] / 2.0, 2.0),
            recent_pivot['price_vs_sma'],
            1.0 if recent_pivot['market_healthy'] else 0.0,
            recent_pivot['quality'],
            (len(price_data) - recent_pivot['index']) / 10.0  # Recency
        ], dtype=np.float32)
        
        confidence = min(recent_pivot['quality'] / 2.0, 1.0) if recent_pivot['quality'] > 1.0 else 0
        
        return {
            'detected': confidence > 0.5,
            'confidence': confidence,
            'vector': vector,
            'pivot_count': len(pocket_pivots),
            'recent_quality': recent_pivot['quality']
        }
    
    def _generate_relative_strength_patterns(self, price_data: pd.DataFrame, market_data: pd.DataFrame) -> np.ndarray:
        """
        Generate relative strength patterns vs market
        """
        if len(price_data) = tf and len(market_data) >= tf:
                stock_return = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[-tf] - 1)
                market_return = (market_data['Close'].iloc[-1] / market_data['Close'].iloc[-tf] - 1)
                rs = stock_return - market_return
                rs_scores.append(rs)
            else:
                rs_scores.append(0)
        
        # RS consistency (outperforming more often than not)
        if len(price_data) >= 21 and len(market_data) >= 21:
            stock_daily_returns = price_data['Close'].pct_change().iloc[-21:]
            market_daily_returns = market_data['Close'].pct_change().iloc[-21:]
            outperform_days = (stock_daily_returns > market_daily_returns).sum()
            consistency = outperform_days / 21.0
        else:
            consistency = 0.5
        
        # Relative strength trend (improving vs deteriorating)
        if len(rs_scores) >= 2:
            rs_trend = rs_scores[-1] - rs_scores[-2]  # Recent vs earlier RS
        else:
            rs_trend = 0
        
        vector = np.array([
            min(rs_scores[0] * 2, 2.0),  # 10-day RS
            min(rs_scores[1] * 1.5, 2.0),  # 21-day RS  
            min(rs_scores[2], 2.0),  # 50-day RS
            consistency,
            min(rs_trend * 5, 1.0),  # RS trend
            1.0 if all(rs > 0 for rs in rs_scores) else 0.0  # All timeframes positive
        ], dtype=np.float32)
        
        return vector
    
    def _generate_ma_alignment_patterns(self, price_data: pd.DataFrame) -> np.ndarray:
        """
        Generate moving average alignment patterns (all MAs in proper order and trending up)
        """
        if len(price_data)  10 > 21 > 50 > 150 > 200)
        alignment_checks = [
            current_price > sma_10.iloc[-1],
            sma_10.iloc[-1] > sma_21.iloc[-1],
            sma_21.iloc[-1] > sma_50.iloc[-1], 
            sma_50.iloc[-1] > sma_150.iloc[-1],
            sma_150.iloc[-1] > sma_200.iloc[-1]
        ]
        alignment_score = sum(alignment_checks) / len(alignment_checks)
        
        # Trending up check (each MA higher than 20 periods ago)
        trending_checks = [
            sma_10.iloc[-1] > sma_10.iloc[-21],
            sma_21.iloc[-1] > sma_21.iloc[-21],
            sma_50.iloc[-1] > sma_50.iloc[-41] if len(sma_50) >= 41 else True,
            sma_150.iloc[-1] > sma_150.iloc[-41] if len(sma_150) >= 41 else True,
            sma_200.iloc[-1] > sma_200.iloc[-41] if len(sma_200) >= 41 else True
        ]
        trending_score = sum(trending_checks) / len(trending_checks)
        
        # Distance from key MAs (closer is better for entries)
        distance_10 = abs(current_price - sma_10.iloc[-1]) / current_price
        distance_21 = abs(current_price - sma_21.iloc[-1]) / current_price
        distance_50 = abs(current_price - sma_50.iloc[-1]) / current_price
        
        # MA separation (not too bunched up)
        ma_separation = (sma_10.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
        
        vector = np.array([
            alignment_score,
            trending_score,
            min(distance_10 * 20, 1.0),  # Normalize distance (closer = higher score)
            min(distance_21 * 15, 1.0),
            min(distance_50 * 10, 1.0),
            min(ma_separation * 5, 1.0),  # Good separation
            1.0 if alignment_score == 1.0 else 0.0,  # Perfect alignment
            1.0 if trending_score >= 0.8 else 0.0  # Strong trending
        ], dtype=np.float32)
        
        return vector
    
    def _generate_fundamental_patterns(self, fundamentals: pd.DataFrame) -> np.ndarray:
        """
        Generate fundamental pattern vectors (basic SEPA fundamental criteria)
        """
        if fundamentals.empty:
            return np.zeros(6, dtype=np.float32)
        
        # Sales growth (25%+ quarterly)
        sales_growth = fundamentals.get('sales_growth_q', pd.Series([0])).iloc[-1]
        sales_score = min(sales_growth / 25.0, 2.0) if sales_growth > 0 else 0
        
        # Earnings growth (25%+ quarterly) 
        eps_growth = fundamentals.get('eps_growth_q', pd.Series([0])).iloc[-1]
        eps_score = min(eps_growth / 25.0, 2.0) if eps_growth > 0 else 0
        
        # ROE (17%+)
        roe = fundamentals.get('roe', pd.Series([0])).iloc[-1]
        roe_score = min(roe / 17.0, 2.0) if roe > 0 else 0
        
        # Debt-to-equity (low is better)
        debt_equity = fundamentals.get('debt_to_equity', pd.Series([1])).iloc[-1]
        debt_score = max(0, 1.0 - debt_equity / 0.5)  # Prefer  25 and eps_growth > 25 and roe > 17) else 0.0
        ], dtype=np.float32)
        
        return vector


class PatternMatchingEngine:
    """
    Engine to combine CANSLIM and SEPA patterns for comprehensive analysis
    """
    
    def __init__(self):
        self.canslim_generator = CANSLIMPatternGenerator()
        self.sepa_generator = SEPAPatternGenerator()
        
    def generate_comprehensive_patterns(self, 
                                      price_data: pd.DataFrame,
                                      volume_data: pd.DataFrame,
                                      market_data: pd.DataFrame,
                                      earnings_data: pd.DataFrame = None,
                                      institutional_data: pd.DataFrame = None,
                                      fundamentals: pd.DataFrame = None) -> Dict:
        """
        Generate both CANSLIM and SEPA patterns for comprehensive analysis
        """
        results = {
            'canslim_patterns': {},
            'sepa_patterns': {},
            'combined_score': 0.0,
            'trading_recommendation': 'HOLD'
        }
        
        # Generate CANSLIM patterns
        if earnings_data is not None and institutional_data is not None:
            results['canslim_patterns'] = self.canslim_generator.generate_canslim_patterns(
                price_data, volume_data, earnings_data, institutional_data, market_data
            )
        
        # Generate SEPA patterns  
        results['sepa_patterns'] = self.sepa_generator.generate_sepa_patterns(
            price_data, volume_data, market_data, fundamentals
        )
        
        # Calculate combined score and recommendation
        results['combined_score'], results['trading_recommendation'] = self._calculate_combined_score(
            results['canslim_patterns'], results['sepa_patterns']
        )
        
        return results
    
    def _calculate_combined_score(self, canslim_patterns: Dict, sepa_patterns: Dict) -> Tuple[float, str]:
        """
        Calculate combined score from both methodologies
        """
        canslim_score = 0.0
        sepa_score = 0.0
        
        # CANSLIM scoring
        if canslim_patterns:
            # Base patterns
            base_scores = []
            if 'base_patterns' in canslim_patterns:
                for pattern_name, pattern_data in canslim_patterns['base_patterns'].items():
                    if isinstance(pattern_data, dict) and 'confidence' in pattern_data:
                        base_scores.append(pattern_data['confidence'])
            
            base_score = max(base_scores) if base_scores else 0.0
            
            # Other CANSLIM factors (simplified scoring)
            factor_scores = []
            for key in ['current_earnings', 'annual_growth', 'new_highs', 'supply_demand', 
                       'leadership', 'institutional', 'market_direction']:
                if key in canslim_patterns:
                    pattern_array = canslim_patterns[key]
                    if len(pattern_array) > 0:
                        factor_scores.append(np.mean(pattern_array))
            
            canslim_score = (base_score * 0.4 + np.mean(factor_scores) * 0.6) if factor_scores else base_score
        
        # SEPA scoring
        if sepa_patterns:
            sepa_scores = []
            
            # Stage analysis (most important)
            if 'stage_analysis' in sepa_patterns:
                stage_data = sepa_patterns['stage_analysis']
                if isinstance(stage_data, dict):
                    stage_score = stage_data.get('stage_2_score', 0) * 0.3
                    sepa_scores.append(stage_score)
            
            # VCP pattern
            if 'vcp' in sepa_patterns:
                vcp_data = sepa_patterns['vcp']
                if isinstance(vcp_data, dict):
                    vcp_score = vcp_data.get('confidence', 0) * 0.25
                    sepa_scores.append(vcp_score)
            
            # Other SEPA factors
            for key in ['pocket_pivots', 'relative_strength', 'ma_alignment']:
                if key in sepa_patterns:
                    if key == 'pocket_pivots' and isinstance(sepa_patterns[key], dict):
                        sepa_scores.append(sepa_patterns[key].get('confidence', 0) * 0.15)
                    elif isinstance(sepa_patterns[key], np.ndarray):
                        sepa_scores.append(np.mean(sepa_patterns[key]) * 0.15)
            
            sepa_score = sum(sepa_scores)
        
        # Combined score (0-100 scale)
        combined_score = (canslim_score * 50 + sepa_score * 50)
        
        # Trading recommendation
        if combined_score >= 75:
            recommendation = 'STRONG_BUY'
        elif combined_score >= 60:
            recommendation = 'BUY'
        elif combined_score >= 40:
            recommendation = 'HOLD'
        elif combined_score >= 25:
            recommendation = 'WEAK_SELL'
        else:
            recommendation = 'SELL'
        
        return combined_score, recommendation


# Example usage and testing
def create_sample_data():
    """Create sample data for testing the pattern generators"""
    import datetime
    
    # Create sample price data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data with trends
    price_base = 100
    prices = [price_base]
    
    for i in range(1, len(dates)):
        # Add some trend and noise
        trend = 0.0005  # Slight upward trend
        noise = np.random.normal(0, 0.02)  # 2% daily volatility
        change = trend + noise
        prices.append(prices[-1] * (1 + change))
    
    price_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices
    })
    
    # Volume data
    volume_data = pd.DataFrame({
        'Date': dates,
        'Volume': [max(100000, int(1000000 + np.random.normal(0, 500000))) for _ in dates]
    })
    
    # Market data (similar structure)
    market_prices = [price_base * 0.8]
    for i in range(1, len(dates)):
        trend = 0.0003
        noise = np.random.normal(0, 0.015)
        change = trend + noise
        market_prices.append(market_prices[-1] * (1 + change))
    
    market_data = pd.DataFrame({
        'Date': dates,
        'Close': market_prices
    })
    
    return price_data, volume_data, market_data

# Test the pattern generators
if __name__ == "__main__":
    # Create sample data
    price_data, volume_data, market_data = create_sample_data()
    
    # Initialize pattern matching engine
    engine = PatternMatchingEngine()
    
    print("Testing CANSLIM and SEPA Pattern Generators...")
    print("=" * 50)
    
    # Test SEPA patterns (don't need earnings data)
    sepa_patterns = engine.sepa_generator.generate_sepa_patterns(
        price_data, volume_data, market_data
    )
    
    print("SEPA Analysis Results:")
    print(f"Stage Analysis: {sepa_patterns.get('stage_analysis', {}).get('stage', 'N/A')}")
    print(f"VCP Detected: {sepa_patterns.get('vcp', {}).get('detected', False)}")
    print(f"Pocket Pivots: {sepa_patterns.get('pocket_pivots', {}).get('pivot_count', 0)}")
    print()
    
    # Create minimal earnings and institutional data for CANSLIM test
    earnings_data = pd.DataFrame({
        'eps_growth': [15, 22, 28, 35],
        'annual_eps_growth': [20, 25, 30],
        'roe': [18, 19, 21],
        'eps_surprise': [5, 8, 12, 15]
    })
    
    institutional_data = pd.DataFrame({
        'institutional_ownership': [0.45, 0.50, 0.55],
        'new_positions': [3, 5, 8]
    })
    
    # Test CANSLIM patterns
    canslim_patterns = engine.canslim_generator.generate_canslim_patterns(
        price_data, volume_data, earnings_data, institutional_data, market_data
    )
    
    print("CANSLIM Analysis Results:")
    for pattern_type, pattern_data in canslim_patterns.get('base_patterns', {}).items():
        if isinstance(pattern_data, dict):
            print(f"{pattern_type.replace('_', ' ').title()}: {pattern_data.get('detected', False)} "
                  f"(Confidence: {pattern_data.get('confidence', 0):.2f})")
    
    print("\nPattern generation completed successfully!")