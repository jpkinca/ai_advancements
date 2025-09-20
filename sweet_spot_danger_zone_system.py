"""
Sweet Spot & Danger Zone Detection System
Production-Grade Dual-Signal Algorithmic Trading Framework

This module implements a sophisticated trading system that combines:
- Sweet Spot Detection: ML-based identification of optimal entry points
- Danger Zone Detection: Risk-focused identification of high-danger market conditions
- Vectorized Operations: High-performance processing for institutional scale
- Integrated Decision Making: Combined signals with risk-adjusted position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
import joblib
import os
from pathlib import Path

# ML and statistical libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from scipy import stats
import talib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SweetSpotConfig:
    """Configuration for Sweet Spot detection model"""
    # Feature parameters
    sequence_length: int = 60  # Historical periods to analyze
    prediction_horizon: int = 10  # Periods ahead to predict
    min_sweetness_threshold: float = 0.65  # Minimum probability for signal

    # ML parameters
    model_type: str = 'xgboost'  # 'xgboost', 'random_forest', 'gradient_boosting'
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1

    # Feature selection
    use_technical_indicators: bool = True
    use_volume_features: bool = True
    use_price_patterns: bool = True
    use_market_context: bool = True

@dataclass
class DangerZoneConfig:
    """Configuration for Danger Zone detection model"""
    # Risk parameters
    sequence_length: int = 30  # Shorter lookback for risk signals
    prediction_horizon: int = 5  # Near-term risk prediction
    max_danger_threshold: float = 0.35  # Maximum danger probability for trading

    # ML parameters
    model_type: str = 'random_forest'
    n_estimators: int = 150
    max_depth: int = 5
    class_weight = 'balanced'  # Handle imbalanced risk events

    # Risk features
    use_volatility_expansion: bool = True
    use_momentum_breakdown: bool = True
    use_support_resistance: bool = True
    use_order_book_signals: bool = True

class TechnicalIndicators:
    """Vectorized technical indicator calculations"""

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using TA-Lib for vectorized performance"""
        prices_array = prices.values.astype('float64')
        return pd.Series(talib.RSI(prices_array, timeperiod=period), index=prices.index)

    @staticmethod
    def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD components"""
        prices_array = prices.values.astype('float64')
        macd, macdsignal, macdhist = talib.MACD(prices_array)
        return (pd.Series(macd, index=prices.index),
                pd.Series(macdsignal, index=prices.index),
                pd.Series(macdhist, index=prices.index))

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0):
        """Calculate Bollinger Bands"""
        prices_array = prices.values.astype('float64')
        upper, middle, lower = talib.BBANDS(prices_array, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
        return (pd.Series(upper, index=prices.index),
                pd.Series(middle, index=prices.index),
                pd.Series(lower, index=prices.index))

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_array = high.values.astype('float64')
        low_array = low.values.astype('float64')
        close_array = close.values.astype('float64')
        return pd.Series(talib.ATR(high_array, low_array, close_array, timeperiod=period),
                        index=close.index)

class SweetSpotDetector:
    """
    ML-based Sweet Spot Detection System
    Identifies optimal entry points with high probability of profit
    """

    def __init__(self, config: SweetSpotConfig = None):
        self.config = config or SweetSpotConfig()
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.is_trained = False

    def create_sweet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for sweet spot detection
        Focuses on momentum, volume, and market strength indicators
        """
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features['price_change_5'] = df['close'].pct_change(5)
        features['price_change_10'] = df['close'].pct_change(10)
        features['price_change_20'] = df['close'].pct_change(20)

        # Momentum features
        if self.config.use_technical_indicators:
            features['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
            features['rsi_slope'] = features['rsi'].diff(3)  # RSI momentum

            macd, macdsignal, macdhist = TechnicalIndicators.calculate_macd(df['close'])
            features['macd'] = macd
            features['macd_signal'] = macdsignal
            features['macd_hist'] = macdhist
            features['macd_crossover'] = (macd > macdsignal).astype(int)

        # Volume features
        if self.config.use_volume_features and 'volume' in df.columns:
            features['volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
            features['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
            features['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()

        # Bollinger Band features
        if self.config.use_price_patterns:
            upper_bb, middle_bb, lower_bb = TechnicalIndicators.calculate_bollinger_bands(df['close'])
            features['bb_position'] = (df['close'] - lower_bb) / (upper_bb - lower_bb)
            features['bb_width'] = (upper_bb - lower_bb) / middle_bb
            features['bb_squeeze'] = features['bb_width'].rolling(10).std() < features['bb_width'].rolling(50).std()

        # Volatility features
        features['atr'] = TechnicalIndicators.calculate_atr(df['high'], df['low'], df['close'])
        features['volatility_10'] = df['close'].pct_change().rolling(10).std()
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()

        # Market regime features
        if self.config.use_market_context:
            features['trend_strength'] = abs(df['close'].rolling(20).mean() - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
            features['momentum_divergence'] = features['price_change_10'] - features['price_change_20']

        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour_of_day'] = df.index.hour / 24.0
            features['day_of_week'] = df.index.dayofweek / 7.0

        # Ensure all features are float64 and handle NaN
        features = features.astype('float64').fillna(0)

        return features

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create target labels for sweet spot detection
        Label = 1 if price rises by >2% in next N periods, else 0
        """
        future_returns = df['close'].pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)
        labels = (future_returns > 0.02).astype(int)  # 2% threshold
        return labels

    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the sweet spot detection model
        """
        logger.info("Training Sweet Spot Detection Model...")

        # Create features and labels
        features = self.create_sweet_features(df)
        labels = self.create_labels(df)

        # Align features and labels, remove NaN
        combined = pd.concat([features, labels], axis=1).dropna()
        features_clean = combined.iloc[:, :-1]
        labels_clean = combined.iloc[:, -1]

        self.feature_columns = features_clean.columns.tolist()

        # Split data (time-aware split)
        split_idx = int(len(features_clean) * (1 - validation_split))
        X_train = features_clean.iloc[:split_idx]
        X_val = features_clean.iloc[split_idx:]
        y_train = labels_clean.iloc[:split_idx]
        y_val = labels_clean.iloc[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Initialize model
        if self.config.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                objective='binary:logistic',
                eval_metric='logloss'
            )
        elif self.config.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42
            )
        else:  # gradient_boosting
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=42
            )

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)

        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'train_precision': precision_score(y_train, train_pred),
            'val_precision': precision_score(y_val, val_pred),
            'train_recall': recall_score(y_train, train_pred),
            'val_recall': recall_score(y_val, val_pred),
            'train_f1': f1_score(y_train, train_pred),
            'val_f1': f1_score(y_val, val_pred)
        }

        self.is_trained = True
        logger.info(f"Model trained. Validation Accuracy: {metrics['val_accuracy']:.3f}")
        return metrics

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sweet spot predictions
        Returns: (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        features = self.create_sweet_features(df)
        features_clean = features.dropna()

        if len(features_clean) == 0:
            return np.array([]), np.array([])

        # Ensure feature alignment
        features_aligned = features_clean[self.feature_columns]
        features_scaled = self.scaler.transform(features_aligned)

        # Get predictions and probabilities
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]  # Probability of class 1 (sweet spot)

        return predictions, probabilities

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            return {}

        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return {}

class DangerZoneDetector:
    """
    ML-based Danger Zone Detection System
    Identifies high-risk market conditions to preserve capital
    """

    def __init__(self, config: DangerZoneConfig = None):
        self.config = config or DangerZoneConfig()
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.is_trained = False

    def create_danger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive risk-focused feature set
        Emphasizes downside momentum, volatility expansion, and breakdown patterns
        """
        features = pd.DataFrame(index=df.index)

        # Negative momentum features (emphasize downside)
        features['neg_momentum_5'] = df['close'].pct_change(5).clip(upper=0)  # Only negative returns
        features['neg_momentum_10'] = df['close'].pct_change(10).clip(upper=0)
        features['downside_volatility'] = df['close'].pct_change().clip(upper=0).rolling(10).std()

        # Volatility expansion (key risk indicator)
        if self.config.use_volatility_expansion:
            features['atr'] = TechnicalIndicators.calculate_atr(df['high'], df['low'], df['close'])
            features['atr_ma_20'] = features['atr'].rolling(20).mean()
            features['volatility_expansion'] = features['atr'] / features['atr_ma_20']
            features['price_volatility'] = df['close'].pct_change().rolling(20).std()

        # Support and resistance breaks
        if self.config.use_support_resistance:
            features['close_50_low'] = df['low'].rolling(50).min()
            features['support_break'] = (df['low'] < features['close_50_low'].shift(1)).astype(int)
            features['resistance_break'] = (df['high'] > df['high'].rolling(50).max().shift(1)).astype(int)

        # Volume on downside
        if 'volume' in df.columns and self.config.use_order_book_signals:
            features['volume_ma_20'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_ma_20']
            # Volume significance on downside moves
            price_down = (df['close'].pct_change() < 0).astype(int)
            features['high_volume_downside'] = features['volume_ratio'] * price_down

        # RSI extremes (overbought/oversold conditions)
        if self.config.use_momentum_breakdown:
            features['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
            features['rsi_extreme_low'] = (features['rsi'] < 30).astype(int)
            features['rsi_extreme_high'] = (features['rsi'] > 70).astype(int)
            features['rsi_divergence'] = features['rsi'].diff(5) * df['close'].pct_change(5) < 0  # RSI vs price divergence

        # Bollinger Band extremes
        upper_bb, middle_bb, lower_bb = TechnicalIndicators.calculate_bollinger_bands(df['close'])
        features['bb_lower_touch'] = ((df['close'] - lower_bb) / (upper_bb - lower_bb)) < 0.1
        features['bb_upper_touch'] = ((df['close'] - lower_bb) / (upper_bb - lower_bb)) > 0.9

        # Gap detection
        if 'open' in df.columns:
            features['gap_down'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) < -0.02
            features['gap_size'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Market stress indicators
        features['price_acceleration'] = df['close'].pct_change().diff()  # Second derivative
        volume_ratio_series = features.get('volume_ratio', pd.Series(1, index=features.index))
        features['volume_surge'] = volume_ratio_series.rolling(5).max() > 2.0

        return features

    def create_risk_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create target labels for danger zone detection
        Label = 1 if significant loss occurs in next N periods, else 0
        """
        future_returns = df['close'].pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)
        # Significant loss: >2% drop or >3% drawdown
        significant_loss = (future_returns < -0.02).astype(int)
        return significant_loss

    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the danger zone detection model
        """
        logger.info("Training Danger Zone Detection Model...")

        # Create features and labels
        features = self.create_danger_features(df)
        labels = self.create_risk_labels(df)

        # Align features and labels, remove NaN
        combined = pd.concat([features, labels], axis=1).dropna()
        features_clean = combined.iloc[:, :-1]
        labels_clean = combined.iloc[:, -1]

        self.feature_columns = features_clean.columns.tolist()

        # Split data (time-aware split)
        split_idx = int(len(features_clean) * (1 - validation_split))
        X_train = features_clean.iloc[:split_idx]
        X_val = features_clean.iloc[split_idx:]
        y_train = labels_clean.iloc[:split_idx]
        y_val = labels_clean.iloc[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Initialize model with class balancing for rare risk events
        if self.config.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                class_weight=self.config.class_weight,
                random_state=42
            )
        elif self.config.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=0.1,
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle imbalance
                objective='binary:logistic',
                eval_metric='logloss'
            )
        else:  # gradient_boosting
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42
            )

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)

        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'train_precision': precision_score(y_train, train_pred),
            'val_precision': precision_score(y_val, val_pred),
            'train_recall': recall_score(y_train, train_pred),
            'val_recall': recall_score(y_val, val_pred),
            'train_f1': f1_score(y_train, train_pred),
            'val_f1': f1_score(y_val, val_pred)
        }

        self.is_trained = True
        logger.info(f"Danger model trained. Validation Accuracy: {metrics['val_accuracy']:.3f}")
        return metrics

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate danger zone predictions
        Returns: (predictions, danger_probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        features = self.create_danger_features(df)
        features_clean = features.dropna()

        if len(features_clean) == 0:
            return np.array([]), np.array([])

        # Ensure feature alignment
        features_aligned = features_clean[self.feature_columns]
        features_scaled = self.scaler.transform(features_aligned)

        # Get predictions and probabilities
        predictions = self.model.predict(features_scaled)
        danger_probabilities = self.model.predict_proba(features_scaled)[:, 1]  # Probability of danger

        return predictions, danger_probabilities

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            return {}

        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return {}

class DualSignalTradingSystem:
    """
    Integrated Sweet Spot + Danger Zone Trading System
    Combines opportunity identification with risk management
    """

    def __init__(self,
                 sweet_config: SweetSpotConfig = None,
                 danger_config: DangerZoneConfig = None):

        self.sweet_detector = SweetSpotDetector(sweet_config)
        self.danger_detector = DangerZoneDetector(danger_config)

        self.sweet_config = sweet_config or SweetSpotConfig()
        self.danger_config = danger_config or DangerZoneConfig()

        self.is_trained = False

    def train(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Train both sweet spot and danger zone models
        """
        logger.info("Training Dual-Signal Trading System...")

        # Train sweet spot detector
        sweet_metrics = self.sweet_detector.train(df)

        # Train danger zone detector
        danger_metrics = self.danger_detector.train(df)

        self.is_trained = True

        return {
            'sweet_spot': sweet_metrics,
            'danger_zone': danger_metrics
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate combined trading signals
        Returns DataFrame with signals and confidence scores
        """
        if not self.is_trained:
            raise ValueError("System not trained. Call train() first.")

        signals = pd.DataFrame(index=df.index)

        # Get sweet spot signals
        sweet_pred, sweet_prob = self.sweet_detector.predict(df)
        signals['sweet_prediction'] = np.nan
        signals['sweet_probability'] = np.nan

        # Align predictions with available data
        sweet_features = self.sweet_detector.create_sweet_features(df)
        valid_sweet_idx = sweet_features.dropna().index
        signals.loc[valid_sweet_idx, 'sweet_prediction'] = sweet_pred
        signals.loc[valid_sweet_idx, 'sweet_probability'] = sweet_prob

        # Get danger zone signals
        danger_pred, danger_prob = self.danger_detector.predict(df)
        signals['danger_prediction'] = np.nan
        signals['danger_probability'] = np.nan

        # Align predictions with available data
        danger_features = self.danger_detector.create_danger_features(df)
        valid_danger_idx = danger_features.dropna().index
        signals.loc[valid_danger_idx, 'danger_prediction'] = danger_pred
        signals.loc[valid_danger_idx, 'danger_probability'] = danger_prob

        # Generate combined decision
        signals['combined_signal'] = self._generate_combined_signal(signals)
        signals['position_size'] = self._calculate_position_size(signals)
        signals['confidence_score'] = self._calculate_confidence(signals)

        return signals

    def _generate_combined_signal(self, signals: pd.DataFrame) -> pd.Series:
        """
        Generate combined trading signal based on sweet spot and danger zone
        """
        sweet_prob = signals['sweet_probability']
        danger_prob = signals['danger_probability']

        # Decision logic
        conditions = [
            # Strong buy: High sweetness, low danger
            (sweet_prob > self.sweet_config.min_sweetness_threshold) & (danger_prob < self.danger_config.max_danger_threshold),
            # Weak buy: Moderate conditions
            (sweet_prob > 0.5) & (danger_prob < 0.5),
            # Avoid: High danger or low sweetness
            (danger_prob > self.danger_config.max_danger_threshold) | (sweet_prob < 0.4)
        ]

        choices = [2, 1, 0]  # 2=Strong Buy, 1=Weak Buy, 0=Avoid
        return pd.Series(np.select(conditions, choices, default=0), index=signals.index)

    def _calculate_position_size(self, signals: pd.DataFrame) -> pd.Series:
        """
        Calculate dynamic position sizing based on signal strength
        """
        sweet_prob = signals['sweet_probability'].fillna(0.5)
        danger_prob = signals['danger_probability'].fillna(0.5)

        # Position size formula: confidence-weighted allocation
        confidence = sweet_prob * (1 - danger_prob)
        position_size = np.clip(confidence * 2, 0, 1)  # Scale to 0-1 range

        return pd.Series(position_size, index=signals.index)

    def _calculate_confidence(self, signals: pd.DataFrame) -> pd.Series:
        """
        Calculate overall confidence score for the signal
        """
        sweet_prob = signals['sweet_probability'].fillna(0.5)
        danger_prob = signals['danger_probability'].fillna(0.5)

        # Combined confidence metric
        confidence = sweet_prob * (1 - danger_prob) * signals['combined_signal']
        return confidence

    def backtest(self, df: pd.DataFrame, initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Backtest the dual-signal system
        """
        logger.info("Running backtest...")

        signals = self.generate_signals(df)

        # Calculate returns
        price_returns = df['close'].pct_change().fillna(0)
        strategy_returns = signals['position_size'].shift(1).fillna(0) * price_returns

        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns

        # Calculate performance metrics
        total_return = portfolio_value.iloc[-1] / initial_capital - 1
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        max_drawdown = drawdown.min()

        # Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = (signals['combined_signal'] > 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'portfolio_value': portfolio_value,
            'signals': signals
        }

    def save_models(self, directory: str = "models"):
        """Save trained models"""
        os.makedirs(directory, exist_ok=True)

        if self.sweet_detector.is_trained:
            joblib.dump(self.sweet_detector, f"{directory}/sweet_spot_detector.pkl")

        if self.danger_detector.is_trained:
            joblib.dump(self.danger_detector, f"{directory}/danger_zone_detector.pkl")

        logger.info(f"Models saved to {directory}")

    def load_models(self, directory: str = "models"):
        """Load trained models"""
        sweet_path = f"{directory}/sweet_spot_detector.pkl"
        danger_path = f"{directory}/danger_zone_detector.pkl"

        if os.path.exists(sweet_path):
            self.sweet_detector = joblib.load(sweet_path)

        if os.path.exists(danger_path):
            self.danger_detector = joblib.load(danger_path)

        self.is_trained = True
        logger.info(f"Models loaded from {directory}")

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for both models"""
        return {
            'sweet_spot': self.sweet_detector.get_feature_importance(),
            'danger_zone': self.danger_detector.get_feature_importance()
        }

# Convenience functions for quick usage
def create_sweet_spot_detector(model_type: str = 'xgboost') -> SweetSpotDetector:
    """Create a pre-configured sweet spot detector"""
    config = SweetSpotConfig(model_type=model_type)
    return SweetSpotDetector(config)

def create_danger_zone_detector(model_type: str = 'random_forest') -> DangerZoneDetector:
    """Create a pre-configured danger zone detector"""
    config = DangerZoneConfig(model_type=model_type)
    return DangerZoneDetector(config)

def create_dual_signal_system() -> DualSignalTradingSystem:
    """Create a complete dual-signal trading system"""
    return DualSignalTradingSystem()

# Example usage and testing functions
def example_usage():
    """
    Example of how to use the Sweet Spot & Danger Zone system
    """
    # Create sample data (replace with real market data)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    np.random.seed(42)

    # Generate synthetic price data with trends and volatility
    n_periods = len(dates)
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n_periods)))
    volume = np.random.lognormal(15, 0.5, n_periods)

    df = pd.DataFrame({
        'open': price * (1 + np.random.normal(0, 0.005, n_periods)),
        'high': price * (1 + np.random.normal(0.002, 0.01, n_periods)),
        'low': price * (1 + np.random.normal(-0.002, 0.01, n_periods)),
        'close': price,
        'volume': volume
    }, index=dates)

    # Create and train the system
    system = create_dual_signal_system()
    metrics = system.train(df)

    print("Training Metrics:")
    print(f"Sweet Spot - Val Accuracy: {metrics['sweet_spot']['val_accuracy']:.3f}")
    print(f"Danger Zone - Val Accuracy: {metrics['danger_zone']['val_accuracy']:.3f}")

    # Generate signals
    signals = system.generate_signals(df)

    # Run backtest
    backtest_results = system.backtest(df)

    print("\nBacktest Results:")
    print(f"Total Return: {backtest_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    print(f"Win Rate: {backtest_results['win_rate']:.2%}")

    return system, signals, backtest_results

if __name__ == "__main__":
    # Run example
    system, signals, results = example_usage()