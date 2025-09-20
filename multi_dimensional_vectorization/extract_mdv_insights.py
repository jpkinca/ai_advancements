
Content is user-generated and unverified.
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class StockTradingInsightExtractor:
    """
    Specialized multi-dimensional insight extractor for stock trading data.
    
    This class analyzes technical indicators, market conditions, and price movements
    to generate actionable trading insights and identify market patterns.
    """
    
    def __init__(self, target_column: str = 'future_return'):
        """
        Initialize the stock trading insight extractor.
        
        Args:
            target_column: Name of target variable (e.g., 'future_return', 'price_direction')
        """
        self.target_column = target_column
        self.scaler = RobustScaler()  # More robust to outliers in financial data
        self.insights = {}
        self.processed_data = None
        self.original_data = None
        self.trading_signals = {}
        
    def create_technical_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical indicators from basic OHLCV data.
        
        Args:
            price_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with technical indicators
        """
        df = price_data.copy()
        
        # Price-based indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta  0).astype(int)
        df['up_days_ratio'] = df['up_days'].rolling(10).mean()
        
        return df
    
    def prepare_trading_data(self, price_data: pd.DataFrame, future_periods: int = 5) -> pd.DataFrame:
        """
        Prepare trading data with technical indicators and future returns.
        
        Args:
            price_data: OHLCV data
            future_periods: Days ahead to predict
            
        Returns:
            DataFrame ready for analysis
        """
        # Create technical indicators
        df = self.create_technical_indicators(price_data)
        
        # Create target variables
        df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1
        df['future_direction'] = (df['future_return'] > 0).astype(int)
        df['future_volatility'] = df['close'].rolling(future_periods).std().shift(-future_periods)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def analyze_trading_data(self, trading_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Main method to extract trading insights from prepared data.
        
        Args:
            trading_data: DataFrame with technical indicators and targets
            
        Returns:
            Dictionary containing trading insights
        """
        self.original_data = trading_data.copy()
        print("📈 Starting Stock Trading Multi-Dimensional Analysis...")
        
        # Identify feature columns (exclude target and basic price data)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'future_return', 
                       'future_direction', 'future_volatility']
        feature_cols = [col for col in trading_data.columns if col not in exclude_cols]
        
        # Prepare data
        self.processed_features = trading_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
        self.target = trading_data[self.target_column]
        
        # Scale features
        self.processed_features_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.processed_features),
            columns=self.processed_features.columns,
            index=self.processed_features.index
        )
        
        # Step 1: Market Regime Analysis
        self._analyze_market_regimes()
        
        # Step 2: Technical Indicator Effectiveness
        self._analyze_indicator_effectiveness()
        
        # Step 3: Trading Pattern Discovery
        self._discover_trading_patterns()
        
        # Step 4: Risk Analysis
        self._analyze_trading_risks()
        
        # Step 5: Signal Generation
        self._generate_trading_signals()
        
        # Step 6: Performance Analysis
        self._analyze_trading_performance()
        
        # Step 7: Generate Trading Recommendations
        self._generate_trading_recommendations()
        
        print("✅ Trading analysis completed!")
        return self.insights
    
    def _analyze_market_regimes(self):
        """Identify different market regimes using clustering."""
        print("📊 Analyzing market regimes...")
        
        # Use key indicators for regime identification
        regime_features = ['volatility', 'momentum_20', 'rsi', 'volume_ratio', 'bb_width']
        regime_data = self.processed_features_scaled[regime_features]
        
        # Find optimal number of regimes
        silhouette_scores = []
        k_range = range(2, 8)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(regime_data)
            score = silhouette_score(regime_data, cluster_labels)
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Final clustering for market regimes
        kmeans_regimes = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        regime_labels = kmeans_regimes.fit_predict(regime_data)
        
        # Analyze regime characteristics
        regime_profiles = {}
        regime_returns = {}
        
        for i in range(optimal_k):
            regime_mask = regime_labels == i
            regime_returns[f'regime_{i}'] = self.target[regime_mask].mean()
            
            regime_profile = {
                'frequency': float(regime_mask.sum() / len(regime_labels)),
                'avg_return': float(self.target[regime_mask].mean()),
                'volatility': float(self.original_data.loc[regime_mask, 'volatility'].mean()),
                'volume_activity': float(self.original_data.loc[regime_mask, 'volume_ratio'].mean()),
                'momentum': float(self.original_data.loc[regime_mask, 'momentum_20'].mean()),
                'rsi_level': float(self.original_data.loc[regime_mask, 'rsi'].mean())
            }
            
            # Classify regime type
            if regime_profile['avg_return'] > 0.005 and regime_profile['momentum'] > 0:
                regime_type = 'Bull Market'
            elif regime_profile['avg_return']  self.original_data['volatility'].median():
                regime_type = 'High Volatility'
            else:
                regime_type = 'Consolidation'
            
            regime_profile['regime_type'] = regime_type
            regime_profiles[f'regime_{i}'] = regime_profile
        
        self.insights['market_regimes'] = {
            'optimal_regimes': optimal_k,
            'regime_profiles': regime_profiles,
            'current_regime': int(regime_labels[-1]),  # Most recent regime
            'regime_stability': float(silhouette_scores[optimal_k - 2])  # Silhouette score
        }
        
        self.regime_labels = regime_labels
    
    def _analyze_indicator_effectiveness(self):
        """Analyze the effectiveness of technical indicators."""
        print("⭐ Analyzing technical indicator effectiveness...")
        
        # Calculate correlation with future returns
        indicator_effectiveness = {}
        
        for indicator in self.processed_features.columns:
            corr_with_return = abs(self.processed_features[indicator].corr(self.target))
            
            # Information coefficient (rank correlation)
            ic = stats.spearmanr(self.processed_features[indicator], self.target)[0]
            
            # Hit rate (directional accuracy)
            indicator_signal = self.processed_features[indicator] > self.processed_features[indicator].median()
            return_signal = self.target > 0
            hit_rate = (indicator_signal == return_signal).mean()
            
            indicator_effectiveness[indicator] = {
                'correlation': float(corr_with_return),
                'information_coefficient': float(abs(ic)) if not np.isnan(ic) else 0,
                'hit_rate': float(hit_rate),
                'effectiveness_score': float((corr_with_return + abs(ic) + abs(hit_rate - 0.5) * 2) / 3)
            }
        
        # Sort by effectiveness
        sorted_indicators = sorted(
            indicator_effectiveness.items(),
            key=lambda x: x[1]['effectiveness_score'],
            reverse=True
        )
        
        self.insights['indicator_effectiveness'] = {
            'top_indicators': dict(sorted_indicators[:10]),
            'least_effective': dict(sorted_indicators[-5:]),
            'overall_effectiveness': {
                'avg_correlation': float(np.mean([v['correlation'] for v in indicator_effectiveness.values()])),
                'avg_hit_rate': float(np.mean([v['hit_rate'] for v in indicator_effectiveness.values()])),
                'top_performer': sorted_indicators[0][0]
            }
        }
    
    def _discover_trading_patterns(self):
        """Discover recurring trading patterns."""
        print("🎯 Discovering trading patterns...")
        
        # Cluster based on technical patterns
        pattern_features = ['rsi', 'macd', 'bb_position', 'momentum_10', 'stoch_k', 'volume_ratio']
        pattern_data = self.processed_features_scaled[pattern_features]
        
        # Use DBSCAN for pattern discovery
        dbscan = DBSCAN(eps=0.3, min_samples=20)
        pattern_labels = dbscan.fit_predict(pattern_data)
        
        n_patterns = len(set(pattern_labels)) - (1 if -1 in pattern_labels else 0)
        
        # Analyze each pattern
        pattern_analysis = {}
        
        for pattern_id in set(pattern_labels):
            if pattern_id == -1:  # Skip noise
                continue
                
            pattern_mask = pattern_labels == pattern_id
            pattern_returns = self.target[pattern_mask]
            
            pattern_analysis[f'pattern_{pattern_id}'] = {
                'frequency': int(pattern_mask.sum()),
                'avg_return': float(pattern_returns.mean()),
                'return_std': float(pattern_returns.std()),
                'win_rate': float((pattern_returns > 0).mean()),
                'sharpe_ratio': float(pattern_returns.mean() / pattern_returns.std()) if pattern_returns.std() > 0 else 0,
                'max_return': float(pattern_returns.max()),
                'min_return': float(pattern_returns.min())
            }
        
        # Identify best patterns
        best_patterns = sorted(
            pattern_analysis.items(),
            key=lambda x: x[1]['sharpe_ratio'],
            reverse=True
        )
        
        self.insights['trading_patterns'] = {
            'total_patterns': n_patterns,
            'pattern_analysis': pattern_analysis,
            'best_patterns': dict(best_patterns[:3]) if best_patterns else {},
            'noise_percentage': float((pattern_labels == -1).mean() * 100)
        }
        
        self.pattern_labels = pattern_labels
    
    def _analyze_trading_risks(self):
        """Analyze various trading risks."""
        print("🚨 Analyzing trading risks...")
        
        # Volatility clustering
        volatility_series = self.original_data['daily_return'].abs()
        volatility_autocorr = volatility_series.autocorr(lag=1)
        
        # Drawdown analysis
        cumulative_returns = (1 + self.target).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown  signal_threshold
        sell_signals = composite_signal  0 else 0,
                'win_rate': float((buy_returns > 0).mean()) if len(buy_returns) > 0 else 0,
                'sharpe_ratio': float(buy_returns.mean() / buy_returns.std()) if len(buy_returns) > 0 and buy_returns.std() > 0 else 0
            },
            'sell_performance': {
                'total_trades': int(sell_signals.sum()),
                'avg_return': float(sell_returns.mean()) if len(sell_returns) > 0 else 0,
                'win_rate': float((sell_returns > 0).mean()) if len(sell_returns) > 0 else 0,
                'sharpe_ratio': float(sell_returns.mean() / sell_returns.std()) if len(sell_returns) > 0 and sell_returns.std() > 0 else 0
            }
        }
        
        self.insights['trading_signals'] = self.trading_signals
    
    def _analyze_trading_performance(self):
        """Analyze overall trading performance."""
        print("📈 Analyzing trading performance...")
        
        # Strategy performance metrics
        all_trades = pd.concat([
            self.target[self.trading_signals['buy_signals']],
            -self.target[self.trading_signals['sell_signals']]
        ])
        
        if len(all_trades) > 0:
            total_return = (1 + all_trades).prod() - 1
            annualized_return = (1 + all_trades.mean()) ** 252 - 1  # Assuming daily data
            annualized_volatility = all_trades.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Maximum consecutive wins/losses
            trade_results = (all_trades > 0).astype(int)
            consecutive_wins = self._max_consecutive(trade_results, 1)
            consecutive_losses = self._max_consecutive(trade_results, 0)
        else:
            total_return = 0
            annualized_return = 0
            annualized_volatility = 0
            sharpe_ratio = 0
            consecutive_wins = 0
            consecutive_losses = 0
        
        self.insights['trading_performance'] = {
            'total_trades': len(all_trades),
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'annualized_volatility': float(annualized_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_consecutive_wins': int(consecutive_wins),
            'max_consecutive_losses': int(consecutive_losses),
            'profit_factor': float(all_trades[all_trades > 0].sum() / abs(all_trades[all_trades  0 else float('inf')
        }
    
    def _max_consecutive(self, series, value):
        """Calculate maximum consecutive occurrences of a value."""
        if len(series) == 0:
            return 0
        
        max_count = 0
        current_count = 0
        
        for val in series:
            if val == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _generate_trading_recommendations(self):
        """Generate actionable trading recommendations."""
        print("💡 Generating trading recommendations...")
        
        recommendations = []
        
        # Market regime recommendations
        current_regime = self.insights['market_regimes']['current_regime']
        regime_profile = self.insights['market_regimes']['regime_profiles'][f'regime_{current_regime}']
        
        recommendations.append({
            'category': 'Market Regime',
            'priority': 'High',
            'action': f"Current market is in {regime_profile['regime_type']} regime",
            'strategy': self._get_regime_strategy(regime_profile['regime_type']),
            'confidence': float(self.insights['market_regimes']['regime_stability'])
        })
        
        # Indicator recommendations
        top_indicator = self.insights['indicator_effectiveness']['overall_effectiveness']['top_performer']
        top_effectiveness = self.insights['indicator_effectiveness']['top_indicators'][top_indicator]['effectiveness_score']
        
        if top_effectiveness > 0.6:
            recommendations.append({
                'category': 'Technical Analysis',
                'priority': 'High',
                'action': f"Focus on {top_indicator} - highest effectiveness indicator",
                'strategy': f"Use {top_indicator} for primary entry/exit signals",
                'confidence': top_effectiveness
            })
        
        # Risk management recommendations
        max_dd = abs(self.insights['trading_risks']['max_drawdown'])
        if max_dd > 0.15:
            recommendations.append({
                'category': 'Risk Management',
                'priority': 'Critical',
                'action': f"Implement strict risk controls - max drawdown is {max_dd:.1%}",
                'strategy': "Use stop-losses and position sizing to limit drawdowns",
                'confidence': 0.9
            })
        
        # Performance recommendations
        sharpe = self.insights['trading_performance']['sharpe_ratio']
        if sharpe > 1.0:
            recommendations.append({
                'category': 'Strategy Performance',
                'priority': 'Medium',
                'action': f"Strong strategy performance (Sharpe: {sharpe:.2f})",
                'strategy': "Consider increasing position size or leverage carefully",
                'confidence': min(sharpe / 2, 1.0)
            })
        elif sharpe  1.5:
                recommendations.append({
                    'category': 'Pattern Trading',
                    'priority': 'Medium',
                    'action': f"Exploit {best_pattern_id} - high Sharpe ratio pattern",
                    'strategy': f"Focus on conditions that create this pattern (occurs {best_pattern['frequency']} times)",
                    'confidence': min(best_pattern['sharpe_ratio'] / 2, 1.0)
                })
        
        self.insights['trading_recommendations'] = recommendations
    
    def _get_regime_strategy(self, regime_type: str) -> str:
        """Get trading strategy for different market regimes."""
        strategies = {
            'Bull Market': 'Focus on momentum strategies and trend following',
            'Bear Market': 'Use defensive strategies, short selling, or hedging',
            'High Volatility': 'Reduce position sizes, use options strategies',
            'Consolidation': 'Range trading strategies, mean reversion'
        }
        return strategies.get(regime_type, 'Use adaptive strategy based on conditions')
    
    def get_trading_summary(self) -> str:
        """Generate comprehensive trading analysis summary."""
        report = "📊 STOCK TRADING MULTI-DIMENSIONAL ANALYSIS SUMMARY\n"
        report += "=" * 60 + "\n\n"
        
        # Market overview
        current_regime = self.insights['market_regimes']['current_regime']
        regime_profile = self.insights['market_regimes']['regime_profiles'][f'regime_{current_regime}']
        
        report += f"🎯 Current Market Regime: {regime_profile['regime_type']}\n"
        report += f"   • Expected Return: {regime_profile['avg_return']:.2%}\n"
        report += f"   • Volatility Level: {regime_profile['volatility']:.4f}\n"
        report += f"   • Volume Activity: {regime_profile['volume_activity']:.2f}x normal\n\n"
        
        # Top performing indicators
        report += "⭐ Most Effective Technical Indicators:\n"
        top_indicators = self.insights['indicator_effectiveness']['top_indicators']
        for i, (indicator, metrics) in enumerate(list(top_indicators.items())[:3], 1):
            report += f"   {i}. {indicator}: {metrics['effectiveness_score']:.3f} effectiveness\n"
            report += f"      → Hit Rate: {metrics['hit_rate']:.1%}, Correlation: {metrics['correlation']:.3f}\n"
        
        report += "\n"
        
        # Trading signals performance
        buy_perf = self.insights['trading_signals']['buy_performance']
        sell_perf = self.insights['trading_signals']['sell_performance']
        
        report += "📡 Trading Signals Performance:\n"
        report += f"   • Buy Signals: {buy_perf['total_trades']} trades, {buy_perf['avg_return']:.2%} avg return, {buy_perf['win_rate']:.1%} win rate\n"
        report += f"   • Sell Signals: {sell_perf['total_trades']} trades, {sell_perf['avg_return']:.2%} avg return, {sell_perf['win_rate']:.1%} win rate\n"
        
        # Overall performance
        performance = self.insights['trading_performance']
        report += f"   • Overall Sharpe Ratio: {performance['sharpe_ratio']:.2f}\n"
        report += f"   • Annualized Return: {performance['annualized_return']:.1%}\n\n"
        
        # Risk metrics
        risks = self.insights['trading_risks']
        report += "⚠️  Risk Analysis:\n"
        report += f"   • Maximum Drawdown: {risks['max_drawdown']:.1%}\n"
        report += f"   • Value at Risk (95%): {risks['value_at_risk_95']:.2%}\n"
        report += f"   • High Risk Periods: {risks['risk_anomaly_percentage']:.1f}% of time\n\n"
        
        # Top recommendations
        report += "💡 Top Trading Recommendations:\n"
        for i, rec in enumerate(self.insights['trading_recommendations'][:3], 1):
            report += f"   {i}. [{rec['priority']}] {rec['action']}\n"
            report += f"      → Strategy: {rec['strategy']}\n"
            report += f"      → Confidence: {rec['confidence']:.1%}\n"
        
        return report
    
    def plot_trading_insights(self, figsize: Tuple[int, int] = (16, 12)):
        """Create comprehensive trading analysis visualizations."""
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('Stock Trading Multi-Dimensional Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Market Regimes
        regime_data = self.insights['market_regimes']['regime_profiles']
        regime_names = [v['regime_type'] for v in regime_data.values()]
        regime_returns = [v['avg_return'] for v in regime_data.values()]
        regime_freqs = [v['frequency'] for v in regime_data.values()]
        
        bars = axes[0, 0].bar(regime_names, regime_returns)
        for i, (bar, freq) in enumerate(zip(bars, regime_freqs)):
            bar.set_color('green' if regime_returns[i] > 0 else 'red')
            bar.set_alpha(0.7)
        axes[0, 0].set_ylabel('Average Return')
        axes[0, 0].set_title('Market Regimes Performance')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Top Indicator Effectiveness
        top_indicators = self.insights['indicator_effectiveness']['top_indicators']
        indicator_names = list(top_indicators.keys())[:8]
        effectiveness_scores = [top_indicators[name]['effectiveness_score'] for name in indicator_names]
        
        axes[0, 1].barh(range(len(indicator_names)), effectiveness_scores)
        axes[0, 1].set_yticks(range(len(indicator_names)))
        axes[0, 1].set_yticklabels(indicator_names)
        axes[0, 1].set_xlabel('Effectiveness Score')
        axes[0, 1].set_title('Top Technical Indicators')
        
        # 3. Trading Signals Timeline
        time_index = range(len(self.trading_signals['composite_signal']))
        axes[0, 2].plot(time_index, self.trading_signals['composite_signal'], label='Composite Signal', alpha=0.7)
        
        buy_points = np.where(self.trading_signals['buy_signals'])[0]
        sell_points = np.where(self.trading_signals['sell_signals'])[0]
        
        axes[0, 2].scatter(buy_points, [self.trading_signals['composite_signal'][i] for i in buy_points], 
                          color='green', marker='^', s=50, label='Buy Signals')
        axes[0, 2].scatter(sell_points, [self.trading_signals['composite_signal'][i] for i in sell_points], 
                          color='red', marker='v', s=50, label='Sell Signals')
        
        axes[0, 2].axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
        axes[0, 2].axhline(y=-0.5, color='red', linestyle='--', alpha=0.5)
        axes[0, 2].set_xlabel('Time Period')
        axes[0, 2].set_ylabel('Signal Strength')
        axes[0, 2].set_title('Trading Signals Timeline')
        axes[0, 2].legend()
        
        # 4. Risk Analysis
        drawdown_data = []
        cumulative_returns = (1 + self.target).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        axes[1, 0].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color='red')
        axes[1, 0].set_xlabel('Time Period')
        axes[1, 0].set_ylabel('Drawdown')
        axes[1, 0].set_title('Portfolio Drawdown Analysis')
        
        # 5. Return Distribution
        axes[1, 1].hist(self.target, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=self.insights['trading_risks']['value_at_risk_95'], 
                          color='red', linestyle='--', label='VaR 95%')
        axes[1, 1].axvline(x=self.target.mean(), color='blue', linestyle='-', label='Mean Return')
        axes[1, 1].set_xlabel('Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Return Distribution')
        axes[1, 1].legend()
        
        # 6. Pattern Analysis
        if self.insights['trading_patterns']['best_patterns']:
            pattern_data = self.insights['trading_patterns']['pattern_analysis']
            pattern_names = list(pattern_data.keys())[:5]
            pattern_sharpe = [pattern_data[name]['sharpe_ratio'] for name in pattern_names]
            
            bars = axes[1, 2].bar(range(len(pattern_names)), pattern_sharpe)
            for i, bar in enumerate(bars):
                bar.set_color('green' if pattern_sharpe[i] > 0 else 'red')
            axes[1, 2].set_xticks(range(len(pattern_names)))
            axes[1, 2].set_xticklabels(pattern_names, rotation=45)
            axes[1, 2].set_ylabel('Sharpe Ratio')
            axes[1, 2].set_title('Trading Pattern Performance')
        
        # 7. Feature Correlation Network (simplified)
        top_features = list(self.insights['indicator_effectiveness']['top_indicators'].keys())[:6]
        corr_matrix = self.processed_features[top_features].corr()
        
        im = axes[2, 0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2, 0].set_xticks(range(len(top_features)))
        axes[2, 0].set_yticks(range(len(top_features)))
        axes[2, 0].set_xticklabels(top_features, rotation=45, ha='right')
        axes[2, 0].set_yticklabels(top_features)
        axes[2, 0].set_title('Top Indicators Correlation')
        plt.colorbar(im, ax=axes[2, 0])
        
        # 8. Volatility Clustering
        volatility = self.original_data['daily_return'].rolling(5).std()
        axes[2, 1].plot(volatility, alpha=0.7)
        axes[2, 1].set_xlabel('Time Period')
        axes[2, 1].set_ylabel('5-Day Rolling Volatility')
        axes[2, 1].set_title('Volatility Clustering')
        
        # 9. Signal Performance Comparison
        buy_metrics = ['avg_return', 'win_rate', 'sharpe_ratio']
        buy_values = [self.trading_signals['buy_performance'][metric] for metric in buy_metrics]
        sell_values = [self.trading_signals['sell_performance'][metric] for metric in buy_metrics]
        
        x = np.arange(len(buy_metrics))
        width = 0.35
        
        axes[2, 2].bar(x - width/2, buy_values, width, label='Buy Signals', alpha=0.7, color='green')
        axes[2, 2].bar(x + width/2, sell_values, width, label='Sell Signals', alpha=0.7, color='red')
        axes[2, 2].set_xlabel('Metrics')
        axes[2, 2].set_ylabel('Values')
        axes[2, 2].set_title('Signal Performance Comparison')
        axes[2, 2].set_xticks(x)
        axes[2, 2].set_xticklabels(buy_metrics)
        axes[2, 2].legend()
        
        plt.tight_layout()
        return fig
    
    def get_current_market_signal(self) -> Dict[str, Any]:
        """Get current market signal and recommendation."""
        latest_signal = self.trading_signals['composite_signal'].iloc[-1]
        current_regime = self.insights['market_regimes']['current_regime']
        regime_profile = self.insights['market_regimes']['regime_profiles'][f'regime_{current_regime}']
        
        # Determine signal strength
        if latest_signal > 0.7:
            signal_strength = 'Strong Buy'
            action = 'BUY'
        elif latest_signal > 0.3:
            signal_strength = 'Moderate Buy'
            action = 'BUY'
        elif latest_signal  str:
        """Calculate recommended position size based on signal and regime."""
        base_size = abs(signal_strength)
        
        # Adjust for market regime
        if regime_profile['regime_type'] == 'High Volatility':
            base_size *= 0.5
        elif regime_profile['regime_type'] == 'Bull Market':
            base_size *= 1.2
        elif regime_profile['regime_type'] == 'Bear Market':
            base_size *= 0.7
        
        # Convert to percentage
        position_pct = min(base_size * 100, 100)
        
        if position_pct > 75:
            return 'Large (>75%)'
        elif position_pct > 50:
            return 'Medium (50-75%)'
        elif position_pct > 25:
            return 'Small (25-50%)'
        else:
            return 'Minimal ( change) - 1]
        
        # Regime-specific parameters
        if current_regime == 'bull':
            drift = 0.001
            vol = 0.015
        elif current_regime == 'bear':
            drift = -0.002
            vol = 0.025
        elif current_regime == 'recovery':
            drift = 0.0015
            vol = 0.020
        else:  # consolidation
            drift = 0.0002
            vol = 0.012
        
        # Price movement with volatility clustering
        if i > 1:
            vol_cluster = vol * (1 + 0.3 * abs(prices[-1]/prices[-2] - 1))
        else:
            vol_cluster = vol
            
        price_change = drift + np.random.normal(0, vol_cluster)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        # Volume based on volatility and price movement
        base_volume = 1000000
        volume_mult = 1 + abs(price_change) * 5 + np.random.normal(0, 0.3)
        volumes.append(int(base_volume * max(volume_mult, 0.1)))
    
    # Create OHLC data
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # Simulate intraday movements
    highs = []
    lows = []
    opens = []
    closes = prices[1:]  # Skip initial price
    
    for i, close_price in enumerate(closes):
        prev_close = prices[i] if i > 0 else base_price
        
        # Open with small gap
        open_gap = np.random.normal(0, 0.005)
        open_price = prev_close * (1 + open_gap)
        opens.append(open_price)
        
        # Daily range
        daily_range = abs(np.random.normal(0, 0.02))
        high_price = max(open_price, close_price) * (1 + daily_range/2)
        low_price = min(open_price, close_price) * (1 - daily_range/2)
        
        highs.append(high_price)
        lows.append(low_price)
    
    # Create DataFrame
    stock_data = pd.DataFrame({
        'date': dates[1:],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    stock_data.set_index('date', inplace=True)
    
    # Initialize trading analyzer
    analyzer = StockTradingInsightExtractor(target_column='future_return')
    
    # Prepare data with technical indicators
    trading_data = analyzer.prepare_trading_data(stock_data, future_periods=5)
    
    # Run analysis
    insights = analyzer.analyze_trading_data(trading_data)
    
    # Print results
    print("\n" + analyzer.get_trading_summary())
    
    # Get current market signal
    current_signal = analyzer.get_current_market_signal()
    print(f"\n🎯 CURRENT TRADING RECOMMENDATION:")
    print(f"Action: {current_signal['action']}")
    print(f"Signal Strength: {current_signal['signal_strength']}")
    print(f"Market Regime: {current_signal['market_regime']}")
    print(f"Confidence: {current_signal['confidence']:.1%}")
    print(f"Position Size: {current_signal['recommended_position_size']}")
    print(f"Risk Level: {current_signal['risk_level']}")
    
    # Create visualizations
    fig = analyzer.plot_trading_insights()
    plt.show()
    
    return analyzer, insights, stock_data

# Advanced trading strategy backtesting
def backtest_trading_strategy(analyzer, stock_data: pd.DataFrame):
    """
    Perform backtesting of the generated trading strategy.
    """
    print("\n🔄 Backtesting Trading Strategy...")
    
    # Get trading signals
    buy_signals = analyzer.trading_signals['buy_signals']
    sell_signals = analyzer.trading_signals['sell_signals']
    
    # Simple backtesting
    portfolio_value = [10000]  # Starting with $10k
    positions = []  # Track open positions
    trades = []
    
    for i in range(len(buy_signals)):
        current_value = portfolio_value[-1]
        
        if buy_signals.iloc[i] and len(positions) == 0:  # Buy signal and no position
            # Calculate position size based on signal strength
            signal_strength = analyzer.trading_signals['composite_signal'][i]
            position_size = min(abs(signal_strength) * 0.5, 0.3)  # Max 30% of portfolio
            
            position_value = current_value * position_size
            shares = position_value / stock_data['close'].iloc[i]
            positions.append({
                'entry_price': stock_data['close'].iloc[i],
                'shares': shares,
                'entry_date': i
            })
            
        elif sell_signals.iloc[i] and len(positions) > 0:  # Sell signal and have position
            position = positions.pop(0)  # Close position
            
            exit_price = stock_data['close'].iloc[i]
            trade_return = (exit_price - position['entry_price']) / position['entry_price']
            trade_pnl = position['shares'] * (exit_price - position['entry_price'])
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': i,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'return': trade_return,
                'pnl': trade_pnl
            })
            
            current_value += trade_pnl
        
        portfolio_value.append(current_value)
    
    # Calculate performance metrics
    total_return = (portfolio_value[-1] - portfolio_value[0]) / portfolio_value[0]
    trade_returns = [trade['return'] for trade in trades]
    
    if trade_returns:
        win_rate = sum(1 for ret in trade_returns if ret > 0) / len(trade_returns)
        avg_win = np.mean([ret for ret in trade_returns if ret > 0]) if any(ret > 0 for ret in trade_returns) else 0
        avg_loss = np.mean([ret for ret in trade_returns if ret  0) / sum(ret for ret in trade_returns if ret  Dict[str, Any]:
        """
        Update trading signal with new market data point.
        
        Args:
            new_data_point: Dictionary with latest market data
            
        Returns:
            Updated trading signal and recommendation
        """
        # Create technical indicators for new data point
        # (In practice, you'd calculate these from recent price history)
        
        # Get top indicators from analysis
        top_indicators = list(self.analyzer.insights['indicator_effectiveness']['top_indicators'].keys())[:5]
        
        # Normalize new data point
        signal_components = []
        for indicator in top_indicators:
            if indicator in new_data_point:
                # Simple normalization (in practice, use historical ranges)
                normalized_value = (new_data_point[indicator] - 0.5) * 2  # Scale to [-1, 1]
                signal_components.append(normalized_value)
        
        # Calculate composite signal
        if signal_components:
            composite_signal = np.mean(signal_components)
        else:
            composite_signal = 0
        
        # Generate recommendation
        if composite_signal > 0.5:
            recommendation = {
                'action': 'BUY',
                'strength': 'Strong' if composite_signal > 0.7 else 'Moderate',
                'confidence': min(abs(composite_signal), 1.0)
            }
        elif composite_signal  analyzer.original_data['volatility'].quantile(0.8)
    high_vol_performance = analyzer.target[high_vol_mask]
    
    print(f"High volatility periods: {high_vol_mask.sum()} days")
    print(f"Average return in high vol: {high_vol_performance.mean():.2%}")
    print(f"Win rate in high vol: {(high_vol_performance > 0).mean():.1%}")
    
    # Scenario 2: Strong momentum periods
    print("\n🚀 Scenario 2: Strong Momentum Market")
    strong_momentum_mask = analyzer.original_data['momentum_20'] > analyzer.original_data['momentum_20'].quantile(0.8)
    momentum_performance = analyzer.target[strong_momentum_mask]
    
    print(f"Strong momentum periods: {strong_momentum_mask.sum()} days")
    print(f"Average return in momentum: {momentum_performance.mean():.2%}")
    print(f"Win rate in momentum: {(momentum_performance > 0).mean():.1%}")
    
    # Scenario 3: Oversold conditions
    print("\n📉 Scenario 3: Oversold Market Conditions")
    oversold_mask = analyzer.original_data['rsi']  0:
        print(f"Oversold periods: {oversold_mask.sum()} days")
        print(f"Average return when oversold: {oversold_performance.mean():.2%}")
        print(f"Win rate when oversold: {(oversold_performance > 0).mean():.1%}")
    else:
        print("No significant oversold periods in dataset")
    
    # Backtest the strategy
    backtest_results, trades = backtest_trading_strategy(analyzer, analyzer.original_data)
    
    # Real-time signal example
    print("\n📡 Real-Time Signal Generator Example:")
    real_time_generator = RealTimeTradingSignals(analyzer)
    
    # Simulate new market data
    latest_data = {
        'rsi': 65.0,
        'macd': 0.02,
        'bb_position': 0.8,
        'momentum_10': 0.03,
        'volume_ratio': 1.5
    }
    
    signal = real_time_generator.update_signal(latest_data)
    print(f"Latest Signal: {signal['recommendation']['action']} ({signal['recommendation']['strength']})")
    print(f"Confidence: {signal['recommendation']['confidence']:.1%}")
    
    return analyzer, insights, backtest_results

# Risk management utilities
def calculate_portfolio_risk_metrics(returns: pd.Series, risk_free_rate: float = 0.02):
    """
    Calculate comprehensive risk metrics for a trading strategy.
    
    Args:
        returns: Series of strategy returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary of risk metrics
    """
    if len(returns) == 0 or returns.std() == 0:
        return {'error': 'Insufficient data for risk calculation'}
    
    # Annualized metrics (assuming daily returns)
    annual_return = (1 + returns.mean()) ** 252 - 1
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    max_drawdown = drawdown.min()
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Tail risk
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns  0 else 0
    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'win_rate': (returns > 0).mean(),
        'profit_factor': returns[returns > 0].sum() / abs(returns[returns  0 else float('inf')
    }

# Run the demonstration
if __name__ == "__main__":
    analyzer, insights, backtest_results = trading_scenario_analysis()
