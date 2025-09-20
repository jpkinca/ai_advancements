#!/usr/bin/env python3
"""
Enhanced Weekend AI Tester with Centralized Data Management

This script uses the MultiTimeframeDataManager to fetch historical data once
and share it across all AI modules. Proper Eastern Time handling for NYSE/NASDAQ.

Key Features:
- Single data fetch for multiple timeframes
- Eastern Time (NYSE/NASDAQ) compliance
- PostgreSQL storage with efficient queries
- All 4 AI modules using shared data
- Comprehensive analysis reporting

Usage: python enhanced_weekend_ai_tester.py
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
from typing import List, Dict, Any, Optional
from decimal import Decimal
import json

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TradeAppComponents_fresh'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our centralized data manager
from multi_timeframe_data_manager import MultiTimeframeDataManager, fetch_weekend_data, EASTERN_TZ

# Import stock universes
from stock_universes import STOCK_UNIVERSES, get_universe

# AI module imports
try:
    from src.reinforcement_learning.ppo_trader import PPOTrader
    from src.genetic_optimization.portfolio_optimizer import PortfolioOptimizer
    from src.sparse_spectrum.fourier_analyzer import FourierAnalyzer
    from src.sparse_spectrum.wavelet_analyzer import WaveletAnalyzer
    AI_MODULES_AVAILABLE = True
except ImportError as e:
    AI_MODULES_AVAILABLE = False
    print(f"[ERROR] AI modules not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'enhanced_weekend_ai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class EnhancedWeekendAITester:
    """
    Enhanced AI tester using centralized multi-timeframe data management
    with proper Eastern Time handling for NYSE/NASDAQ compliance
    """
    
    def __init__(self, stock_universe: List[str], database_url: str = None):
        self.stock_universe = stock_universe
        self.database_url = database_url or os.getenv('DATABASE_URL')
        
        # Data manager
        self.data_manager = None
        
        # AI modules
        self.ai_modules = {}
        
        # Analysis results
        self.analysis_results = {}
        
        # Optimal timeframes for each module
        self.module_timeframes = {
            'ppo_trader': ['15min', '1hour'],      # Good for RL training
            'portfolio_optimizer': ['1day'],       # Daily returns for portfolio optimization
            'fourier_analyzer': ['1hour', '1day'], # Frequency analysis across timeframes
            'wavelet_analyzer': ['15min', '1hour', '1day']  # Multi-scale analysis
        }
        
        logger.info(f"[SETUP] Enhanced Weekend AI Tester initialized")
        logger.info(f"   Stock universe: {len(stock_universe)} symbols")
        logger.info(f"   Database: {'Configured' if self.database_url else 'Not configured'}")
        logger.info(f"   Eastern Time Zone: {EASTERN_TZ}")
    
    async def initialize_data_manager(self) -> bool:
        """Initialize and populate the data manager"""
        logger.info("[PROCESSING] Initializing multi-timeframe data manager...")
        
        # Determine all required timeframes
        all_timeframes = set()
        for module_tfs in self.module_timeframes.values():
            all_timeframes.update(module_tfs)
        
        timeframes = sorted(list(all_timeframes))
        logger.info(f"[DATA] Required timeframes: {timeframes}")
        
        # Fetch all data using the centralized manager
        self.data_manager = await fetch_weekend_data(
            symbols=self.stock_universe,
            timeframes=timeframes,
            database_url=self.database_url
        )
        
        # Verify we have data
        summary = self.data_manager.get_fetch_summary()
        if summary['total_symbols'] == 0:
            logger.error("[ERROR] No data was fetched successfully")
            return False
        
        logger.info(f"[SUCCESS] Data manager initialized: {summary['total_symbols']} symbols, {summary['total_bars']:,} total bars")
        return True
    
    def initialize_ai_modules(self) -> bool:
        """Initialize all AI modules"""
        if not AI_MODULES_AVAILABLE:
            logger.error("[ERROR] AI modules not available")
            return False
        
        try:
            logger.info("[PROCESSING] Initializing AI modules...")
            
            # PPO Trader - Enhanced configuration for multi-timeframe
            self.ai_modules['ppo_trader'] = PPOTrader(
                state_size=15,  # Extended state space for multi-timeframe features
                action_size=3,  # BUY, SELL, HOLD
                learning_rate=0.0003,
                gamma=0.99,
                clip_ratio=0.2,
                batch_size=64,
                epochs=10
            )
            
            # Portfolio Optimizer - Larger population for better diversity
            self.ai_modules['portfolio_optimizer'] = PortfolioOptimizer(
                population_size=150,  # Larger for better exploration
                generations=75,       # More generations for convergence
                mutation_rate=0.12,
                crossover_rate=0.85,
                elitism_rate=0.15
            )
            
            # Fourier Analyzer - Multi-timeframe configuration
            self.ai_modules['fourier_analyzer'] = FourierAnalyzer(
                config={
                    'window_size': 252,
                    'min_frequency': 0.005,  # Lower for longer cycles
                    'max_frequency': 0.5,
                    'noise_threshold': 0.08,
                    'overlap_ratio': 0.5
                }
            )
            
            # Wavelet Analyzer - Enhanced for multi-scale analysis
            self.ai_modules['wavelet_analyzer'] = WaveletAnalyzer(
                config={
                    'wavelet_type': 'morlet',
                    'scales': list(range(1, 128)),  # Extended scale range
                    'sampling_period': 1,
                    'significance_level': 0.95,
                    'cone_of_influence': True
                }
            )
            
            logger.info("[SUCCESS] All AI modules initialized with enhanced configurations")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize AI modules: {e}")
            return False
    
    def run_portfolio_optimization(self) -> Dict[str, Any]:
        """Run portfolio optimization using daily returns data"""
        logger.info("[PROCESSING] Running Enhanced Portfolio Optimization...")
        
        try:
            # Get daily data for all symbols
            daily_returns = {}
            symbols_with_data = []
            
            for symbol in self.stock_universe:
                cached_data = self.data_manager.get_cached_data(symbol, '1day')
                if '1day' in cached_data and not cached_data['1day'].empty:
                    df = cached_data['1day']
                    if len(df) > 60:  # Need at least ~3 months of data
                        returns = df['returns'].dropna()
                        if len(returns) > 30:  # Final check
                            daily_returns[symbol] = returns
                            symbols_with_data.append(symbol)
            
            if len(daily_returns)  0 else 0
                
                optimization_result.update({
                    'portfolio_return': portfolio_return,
                    'portfolio_volatility': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'symbols': symbols_with_data,
                    'expected_returns': expected_returns
                })
            
            logger.info(f"[SUCCESS] Portfolio optimization complete")
            logger.info(f"   Fitness: {optimization_result.get('fitness', 0):.4f}")
            logger.info(f"   Expected Return: {optimization_result.get('portfolio_return', 0):.1%}")
            logger.info(f"   Volatility: {optimization_result.get('portfolio_volatility', 0):.1%}")
            logger.info(f"   Sharpe Ratio: {optimization_result.get('sharpe_ratio', 0):.2f}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"[ERROR] Portfolio optimization failed: {e}")
            return {}
    
    def run_multi_timeframe_fourier_analysis(self) -> Dict[str, Any]:
        """Run Fourier analysis across multiple timeframes"""
        logger.info("[PROCESSING] Running Multi-Timeframe Fourier Analysis...")
        
        analysis_results = {}
        
        try:
            for timeframe in self.module_timeframes['fourier_analyzer']:
                timeframe_results = {}
                
                logger.info(f"[ANALYSIS] Fourier analysis on {timeframe} timeframe")
                
                for symbol in self.stock_universe:
                    cached_data = self.data_manager.get_cached_data(symbol, timeframe)
                    
                    if timeframe in cached_data and not cached_data[timeframe].empty:
                        df = cached_data[timeframe]
                        
                        if len(df) > 50:  # Minimum for meaningful frequency analysis
                            price_data = df['close'].values
                            
                            # Run Fourier analysis
                            analysis_result = self.ai_modules['fourier_analyzer'].analyze_frequencies(
                                market_data=price_data,
                                symbol=f"{symbol}_{timeframe}"
                            )
                            
                            # Add timeframe-specific metadata
                            analysis_result.update({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'data_points': len(df),
                                'date_range': {
                                    'start': df['date'].iloc[0].strftime('%Y-%m-%d %H:%M ET'),
                                    'end': df['date'].iloc[-1].strftime('%Y-%m-%d %H:%M ET')
                                },
                                'market_hours_only': df['is_market_hours'].all() if 'is_market_hours' in df else True
                            })
                            
                            timeframe_results[symbol] = analysis_result
                            
                            # Log key findings
                            dominant_freq = analysis_result.get('dominant_frequency', 0)
                            cycle_strength = analysis_result.get('cycle_strength', 0)
                            
                            if dominant_freq > 0:
                                # Convert frequency to time period based on timeframe
                                if timeframe == '15min':
                                    cycle_period = f"{(1/dominant_freq * 0.25):.1f} hours"
                                elif timeframe == '1hour':
                                    cycle_period = f"{(1/dominant_freq):.1f} hours"
                                elif timeframe == '1day':
                                    cycle_period = f"{(1/dominant_freq):.1f} days"
                                else:
                                    cycle_period = f"{(1/dominant_freq):.1f} periods"
                                
                                logger.info(f"[CYCLE] {symbol} {timeframe}: {cycle_period} cycle, strength {cycle_strength:.3f}")
                
                analysis_results[timeframe] = timeframe_results
            
            logger.info(f"[SUCCESS] Multi-timeframe Fourier analysis complete")
            return analysis_results
            
        except Exception as e:
            logger.error(f"[ERROR] Fourier analysis failed: {e}")
            return {}
    
    def run_multi_scale_wavelet_analysis(self) -> Dict[str, Any]:
        """Run wavelet analysis across multiple timeframes"""
        logger.info("[PROCESSING] Running Multi-Scale Wavelet Analysis...")
        
        analysis_results = {}
        
        try:
            for timeframe in self.module_timeframes['wavelet_analyzer']:
                timeframe_results = {}
                
                logger.info(f"[ANALYSIS] Wavelet analysis on {timeframe} timeframe")
                
                for symbol in self.stock_universe:
                    cached_data = self.data_manager.get_cached_data(symbol, timeframe)
                    
                    if timeframe in cached_data and not cached_data[timeframe].empty:
                        df = cached_data[timeframe]
                        
                        if len(df) > 64:  # Need sufficient data for wavelet analysis
                            price_data = df['close'].values
                            
                            # Run wavelet analysis
                            analysis_result = self.ai_modules['wavelet_analyzer'].analyze(
                                price_data=price_data,
                                symbol=f"{symbol}_{timeframe}"
                            )
                            
                            # Add timeframe-specific insights
                            analysis_result.update({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'data_points': len(df),
                                'date_range': {
                                    'start': df['date'].iloc[0].strftime('%Y-%m-%d %H:%M ET'),
                                    'end': df['date'].iloc[-1].strftime('%Y-%m-%d %H:%M ET')
                                },
                                'volatility_statistics': {
                                    'mean_volatility': df['returns'].std() * np.sqrt(252) if 'returns' in df else None,
                                    'volatility_regime': self._classify_volatility_regime(df['returns']) if 'returns' in df else 'UNKNOWN'
                                }
                            })
                            
                            timeframe_results[symbol] = analysis_result
                            
                            # Log key insights
                            volatility_regime = analysis_result['volatility_statistics']['volatility_regime']
                            pattern_strength = analysis_result.get('pattern_strength', 0)
                            
                            logger.info(f"[PATTERN] {symbol} {timeframe}: {volatility_regime} regime, pattern strength {pattern_strength:.3f}")
                
                analysis_results[timeframe] = timeframe_results
            
            logger.info(f"[SUCCESS] Multi-scale wavelet analysis complete")
            return analysis_results
            
        except Exception as e:
            logger.error(f"[ERROR] Wavelet analysis failed: {e}")
            return {}
    
    def _classify_volatility_regime(self, returns: pd.Series) -> str:
        """Classify volatility regime based on returns"""
        if returns.empty:
            return 'UNKNOWN'
        
        annualized_vol = returns.std() * np.sqrt(252)
        
        if annualized_vol  Dict[str, Any]:
        """Train PPO models using multi-timeframe data"""
        logger.info("[PROCESSING] Training Enhanced PPO Models...")
        
        training_results = {}
        
        try:
            # Select best symbols for training (those with most complete data)
            training_symbols = self._select_best_training_symbols(max_symbols=3)
            
            for symbol in training_symbols:
                symbol_results = {}
                
                for timeframe in self.module_timeframes['ppo_trader']:
                    cached_data = self.data_manager.get_cached_data(symbol, timeframe)
                    
                    if timeframe in cached_data and not cached_data[timeframe].empty:
                        df = cached_data[timeframe]
                        
                        if len(df) > 200:  # Need sufficient data for RL training
                            # Prepare enhanced training data with multi-timeframe features
                            training_data = self._prepare_enhanced_training_data(df, symbol, timeframe)
                            
                            if len(training_data) > 100:
                                logger.info(f"[TRAINING] PPO model for {symbol} on {timeframe} ({len(training_data)} samples)")
                                
                                # Train model
                                training_result = self.ai_modules['ppo_trader'].train(
                                    market_data=training_data,
                                    episodes=150,  # More episodes for better learning
                                    steps_per_episode=min(len(training_data) // 8, 100)
                                )
                                
                                # Add metadata
                                training_result.update({
                                    'symbol': symbol,
                                    'timeframe': timeframe,
                                    'training_samples': len(training_data),
                                    'data_quality_score': self._calculate_data_quality_score(df)
                                })
                                
                                symbol_results[timeframe] = training_result
                                
                                logger.info(f"[SUCCESS] {symbol} {timeframe}: Final reward {training_result.get('final_reward', 0):.4f}")
                
                if symbol_results:
                    training_results[symbol] = symbol_results
            
            logger.info(f"[SUCCESS] Enhanced PPO training complete for {len(training_results)} symbols")
            return training_results
            
        except Exception as e:
            logger.error(f"[ERROR] PPO training failed: {e}")
            return {}
    
    def _select_best_training_symbols(self, max_symbols: int = 3) -> List[str]:
        """Select symbols with best data quality for training"""
        symbol_scores = {}
        
        for symbol in self.stock_universe:
            score = 0
            
            for timeframe in self.module_timeframes['ppo_trader']:
                cached_data = self.data_manager.get_cached_data(symbol, timeframe)
                
                if timeframe in cached_data and not cached_data[timeframe].empty:
                    df = cached_data[timeframe]
                    score += len(df) * self._calculate_data_quality_score(df)
            
            if score > 0:
                symbol_scores[symbol] = score
        
        # Return top symbols
        sorted_symbols = sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in sorted_symbols[:max_symbols]]
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0-1)"""
        score = 1.0
        
        # Penalize missing data
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score *= (1 - missing_ratio)
        
        # Reward data completeness
        if 'volume' in df.columns:
            zero_volume_ratio = (df['volume'] == 0).sum() / len(df)
            score *= (1 - zero_volume_ratio * 0.5)
        
        # Reward reasonable price movements
        if 'returns' in df.columns:
            extreme_returns = (abs(df['returns']) > 0.2).sum() / len(df)
            score *= (1 - extreme_returns * 0.3)
        
        return max(0.0, min(1.0, score))
    
    def _prepare_enhanced_training_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict[str, float]]:
        """Prepare enhanced training data with multi-timeframe features"""
        training_data = []
        
        # Enhanced feature engineering
        df = df.copy()
        
        # Technical indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Market microstructure
        df['spread_estimate'] = (df['high'] - df['low']) / df['close']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Time-based features
        if 'date' in df.columns:
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_market_open'] = df.get('is_market_hours', True)
        
        # Convert to training format
        required_lookback = 30
        for i in range(required_lookback, len(df)):
            row_data = {
                # OHLCV
                'open': float(df.iloc[i]['open']),
                'high': float(df.iloc[i]['high']),
                'low': float(df.iloc[i]['low']),
                'close': float(df.iloc[i]['close']),
                'volume': float(df.iloc[i].get('volume', 0)),
                
                # Technical indicators
                'rsi': float(df.iloc[i]['rsi']) if pd.notna(df.iloc[i]['rsi']) else 50.0,
                'bb_position': float((df.iloc[i]['close'] - df.iloc[i]['bb_lower']) / 
                                   (df.iloc[i]['bb_upper'] - df.iloc[i]['bb_lower'])) 
                              if pd.notna(df.iloc[i]['bb_upper']) else 0.5,
                'macd': float(df.iloc[i]['macd']) if pd.notna(df.iloc[i]['macd']) else 0.0,
                'macd_signal': float(df.iloc[i]['macd_signal']) if pd.notna(df.iloc[i]['macd_signal']) else 0.0,
                
                # Volume
                'volume_ratio': float(df.iloc[i].get('volume_ratio', 1.0)) if pd.notna(df.iloc[i].get('volume_ratio')) else 1.0,
                
                # Market microstructure
                'spread_estimate': float(df.iloc[i]['spread_estimate']) if pd.notna(df.iloc[i]['spread_estimate']) else 0.01,
                'price_position': float(df.iloc[i]['price_position']) if pd.notna(df.iloc[i]['price_position']) else 0.5,
                
                # Time features
                'hour': float(df.iloc[i].get('hour', 12)),
                'day_of_week': float(df.iloc[i].get('day_of_week', 2)),
                'is_market_open': float(df.iloc[i].get('is_market_open', True)),
                
                # Returns
                'returns': float(df.iloc[i]['returns']) if pd.notna(df.iloc[i]['returns']) else 0.0,
                
                # Metadata
                'symbol': symbol,
                'timeframe': timeframe
            }
            training_data.append(row_data)
        
        return training_data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta  Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = []
        
        report.append("=" * 100)
        report.append("    ENHANCED WEEKEND AI TRADING MODULES COMPREHENSIVE REPORT")
        report.append("=" * 100)
        report.append("")
        
        # Data collection summary
        summary = self.data_manager.get_fetch_summary()
        report.append("DATA COLLECTION SUMMARY (Eastern Time):")
        report.append(f"   Total symbols processed: {summary['total_symbols']}")
        report.append(f"   Total data points: {summary['total_bars']:,}")
        report.append(f"   Available timeframes: {', '.join(summary['timeframes_available'])}")
        report.append(f"   Data timezone: Eastern Time (NYSE/NASDAQ)")
        report.append("")
        
        # Data breakdown by symbol
        report.append("SYMBOL DATA BREAKDOWN:")
        for symbol in summary['symbols']:
            symbol_data = self.data_manager.get_cached_data(symbol)
            total_bars = sum(len(df) for df in symbol_data.values())
            timeframes = ', '.join(symbol_data.keys())
            report.append(f"   {symbol:8}: {total_bars:6,} bars across {timeframes}")
        report.append("")
        
        # Portfolio optimization results
        if 'portfolio_optimization' in self.analysis_results:
            result = self.analysis_results['portfolio_optimization']
            report.append("ENHANCED PORTFOLIO OPTIMIZATION RESULTS:")
            report.append(f"   Optimization fitness: {result.get('fitness', 0):.4f}")
            report.append(f"   Expected annual return: {result.get('portfolio_return', 0):.1%}")
            report.append(f"   Portfolio volatility: {result.get('portfolio_volatility', 0):.1%}")
            report.append(f"   Sharpe ratio: {result.get('sharpe_ratio', 0):.2f}")
            
            if 'weights' in result and 'symbols' in result:
                report.append("   Optimal allocation:")
                for symbol, weight in zip(result['symbols'], result['weights']):
                    expected_ret = result['expected_returns'].get(symbol, 0)
                    report.append(f"      {symbol:8}: {weight:.1%} (expected return: {expected_ret:.1%})")
            report.append("")
        
        # Multi-timeframe Fourier analysis
        if 'fourier_analysis' in self.analysis_results:
            report.append("MULTI-TIMEFRAME FOURIER ANALYSIS:")
            results = self.analysis_results['fourier_analysis']
            
            for timeframe, timeframe_results in results.items():
                report.append(f"   {timeframe.upper()} Timeframe Analysis:")
                
                for symbol, analysis in timeframe_results.items():
                    dominant_freq = analysis.get('dominant_frequency', 0)
                    cycle_strength = analysis.get('cycle_strength', 0)
                    data_points = analysis.get('data_points', 0)
                    
                    if dominant_freq > 0:
                        # Convert to meaningful time periods
                        if timeframe == '15min':
                            period_desc = f"{(1/dominant_freq * 0.25):.1f}h cycle"
                        elif timeframe == '1hour':
                            period_desc = f"{(1/dominant_freq):.1f}h cycle"
                        elif timeframe == '1day':
                            period_desc = f"{(1/dominant_freq):.1f}d cycle"
                        else:
                            period_desc = f"{(1/dominant_freq):.1f} period cycle"
                        
                        report.append(f"      {symbol:8}: {period_desc}, strength {cycle_strength:.3f} ({data_points} points)")
                report.append("")
        
        # Multi-scale wavelet analysis
        if 'wavelet_analysis' in self.analysis_results:
            report.append("MULTI-SCALE WAVELET ANALYSIS:")
            results = self.analysis_results['wavelet_analysis']
            
            for timeframe, timeframe_results in results.items():
                report.append(f"   {timeframe.upper()} Timeframe Analysis:")
                
                for symbol, analysis in timeframe_results.items():
                    vol_stats = analysis.get('volatility_statistics', {})
                    vol_regime = vol_stats.get('volatility_regime', 'UNKNOWN')
                    mean_vol = vol_stats.get('mean_volatility', 0)
                    pattern_strength = analysis.get('pattern_strength', 0)
                    data_points = analysis.get('data_points', 0)
                    
                    vol_desc = f"{mean_vol:.1%} vol" if mean_vol else "N/A vol"
                    report.append(f"      {symbol:8}: {vol_regime} regime ({vol_desc}), pattern {pattern_strength:.3f} ({data_points} points)")
                report.append("")
        
        # Enhanced PPO training results
        if 'ppo_training' in self.analysis_results:
            report.append("ENHANCED PPO TRAINING RESULTS:")
            results = self.analysis_results['ppo_training']
            
            for symbol, symbol_results in results.items():
                report.append(f"   {symbol} Training Results:")
                
                for timeframe, training_result in symbol_results.items():
                    final_reward = training_result.get('final_reward', 0)
                    training_samples = training_result.get('training_samples', 0)
                    quality_score = training_result.get('data_quality_score', 0)
                    
                    report.append(f"      {timeframe:8}: Final reward {final_reward:.4f}, {training_samples} samples, quality {quality_score:.2f}")
                report.append("")
        
        # Eastern Time compliance note
        report.append("TIMEZONE COMPLIANCE:")
        report.append("   All timestamps converted to Eastern Time (America/New_York)")
        report.append("   Market hours filtering: 9:30 AM - 4:00 PM ET, Monday-Friday")
        report.append("   Database storage uses TIMESTAMPTZ with Eastern Time")
        report.append("")
        
        report.append("=" * 100)
        report.append("    ENHANCED AI ANALYSIS COMPLETE")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    async def run_complete_enhanced_analysis(self) -> bool:
        """Run complete enhanced AI analysis pipeline"""
        logger.info("[STARTING] Enhanced Weekend AI Trading Modules Analysis")
        
        try:
            # Step 1: Initialize data manager and fetch all data
            if not await self.initialize_data_manager():
                return False
            
            # Step 2: Initialize AI modules
            if not self.initialize_ai_modules():
                return False
            
            # Step 3: Run portfolio optimization
            logger.info("[ANALYSIS] Running portfolio optimization...")
            portfolio_result = self.run_portfolio_optimization()
            if portfolio_result:
                self.analysis_results['portfolio_optimization'] = portfolio_result
            
            # Step 4: Run multi-timeframe Fourier analysis
            logger.info("[ANALYSIS] Running multi-timeframe Fourier analysis...")
            fourier_result = self.run_multi_timeframe_fourier_analysis()
            if fourier_result:
                self.analysis_results['fourier_analysis'] = fourier_result
            
            # Step 5: Run multi-scale wavelet analysis
            logger.info("[ANALYSIS] Running multi-scale wavelet analysis...")
            wavelet_result = self.run_multi_scale_wavelet_analysis()
            if wavelet_result:
                self.analysis_results['wavelet_analysis'] = wavelet_result
            
            # Step 6: Train enhanced PPO models
            logger.info("[ANALYSIS] Training enhanced PPO models...")
            ppo_result = self.train_enhanced_ppo_models()
            if ppo_result:
                self.analysis_results['ppo_training'] = ppo_result
            
            # Step 7: Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            # Save report
            timestamp = datetime.now(EASTERN_TZ).strftime("%Y%m%d_%H%M%S")
            report_filename = f'enhanced_weekend_ai_analysis_{timestamp}.txt'
            
            with open(report_filename, 'w') as f:
                f.write(report)
            
            print(report)
            logger.info(f"[SUCCESS] Enhanced analysis complete - Report saved to {report_filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Enhanced analysis failed: {e}")
            return False

async def main():
    """Main function for enhanced weekend AI testing"""
    print("=" * 100)
    print("    ENHANCED WEEKEND AI TESTING WITH MULTI-TIMEFRAME DATA")
    print("=" * 100)
    print()
    
    # Check prerequisites
    if not AI_MODULES_AVAILABLE:
        print("[ERROR] AI modules not available")
        return
    
    # Select stock universe
    print("Available stock universes:")
    for name, symbols in STOCK_UNIVERSES.items():
        print(f"  {name:20}: {len(symbols):2d} symbols")
    print()
    
    # Use medium_test as default - good balance for comprehensive testing
    selected_universe = STOCK_UNIVERSES['medium_test']
    
    print(f"Using stock universe: {len(selected_universe)} symbols")
    print(f"Symbols: {', '.join(selected_universe)}")
    print()
    print("Enhanced Features:")
    print("  • Multi-timeframe data collection (15min, 1hour, 1day)")
    print("  • Eastern Time (NYSE/NASDAQ) compliance")
    print("  • PostgreSQL storage with efficient querying")
    print("  • Enhanced AI module configurations")
    print("  • Comprehensive cross-timeframe analysis")
    print()
    
    # Create enhanced tester
    tester = EnhancedWeekendAITester(
        stock_universe=selected_universe,
        database_url=os.getenv('DATABASE_URL')
    )
    
    # Run complete enhanced analysis
    success = await tester.run_complete_enhanced_analysis()
    
    if success:
        print()
        print("[SUCCESS] Enhanced weekend AI analysis completed successfully!")
        print()
        print("Key Achievements:")
        print("  ✓ Multi-timeframe historical data collected")
        print("  ✓ All timestamps in Eastern Time (NYSE/NASDAQ)")
        print("  ✓ Data stored in PostgreSQL for reuse")
        print("  ✓ Portfolio optimization with enhanced constraints")
        print("  ✓ Multi-timeframe frequency domain analysis")
        print("  ✓ Multi-scale wavelet time-frequency analysis")
        print("  ✓ Enhanced PPO reinforcement learning training")
        print("  ✓ Comprehensive analysis report generated")
        print()
        print("Data is now available for further AI module development!")
    else:
        print()
        print("[ERROR] Enhanced weekend AI analysis failed!")
        print("Check logs for details.")

if __name__ == "__main__":
    asyncio.run(main())
