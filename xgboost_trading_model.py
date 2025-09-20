#!/usr/bin/env python3
"""
XGBoost Trading Model

This module implements XGBoost-based predictive models for algorithmic trading,
including regression for price forecasting and classification for trading signals.

Features:
- Price prediction using gradient boosting regression
- Buy/sell/hold signal classification
- Technical indicator feature engineering
- Hyperparameter tuning and cross-validation
- Backtesting framework
- Integration with existing data sources
"""

import asyncio
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from decimal import Decimal
import pickle
import json

# XGBoost and ML imports
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Project imports
from ai_data_accessor import AIDataAccessor, create_ai_data_accessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostTradingModel:
    """
    XGBoost-based trading model for price prediction and signal generation
    """

    def __init__(self, symbol: str = 'AAPL', start_date: str = '2020-01-01',
                 end_date: Optional[str] = None, timeframe: str = '1day'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.timeframe = timeframe

        # Model components
        self.regression_model = None
        self.classification_model = None
        self.scaler = StandardScaler()

        # Data
        self.raw_data = None
        self.feature_data = None
        self.target_data = None

        # Model paths
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)

        logger.info(f"[INIT] XGBoost Trading Model for {symbol} ({timeframe})")

    async def load_data(self) -> pd.DataFrame:
        """Load historical market data"""
        try:
            accessor = await create_ai_data_accessor()

            start_dt = pd.to_datetime(self.start_date)
            end_dt = pd.to_datetime(self.end_date)

            self.raw_data = await accessor.get_symbol_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=start_dt,
                end_date=end_dt
            )

            await accessor.close()

            if self.raw_data.empty:
                logger.error(f"[ERROR] No data loaded for {self.symbol}")
                return pd.DataFrame()

            logger.info(f"[SUCCESS] Loaded {len(self.raw_data)} data points for {self.symbol}")
            return self.raw_data

        except Exception as e:
            logger.error(f"[ERROR] Failed to load data: {e}")
            return pd.DataFrame()

    def add_technical_indicators(self) -> pd.DataFrame:
        """Add technical indicators as features"""
        if self.raw_data is None or self.raw_data.empty:
            logger.error("[ERROR] No raw data available")
            return pd.DataFrame()

        df = self.raw_data.copy()

        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # Volatility measures
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_50'] = df['returns'].rolling(window=50).std()

        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # RSI (Relative Strength Index)
        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['rsi_14'] = calculate_rsi(df['close'], 14)

        # MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_middle'] = sma_20
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Price position within Bollinger Bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Momentum indicators
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        # Drop NaN values created by indicators
        df = df.dropna()

        self.feature_data = df
        logger.info(f"[SUCCESS] Added technical indicators, {len(df)} rows remaining")
        return df

    def prepare_targets(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare target variables for regression and classification"""
        if self.feature_data is None or self.feature_data.empty:
            logger.error("[ERROR] No feature data available")
            return pd.DataFrame(), pd.Series(), pd.Series()

        df = self.feature_data.copy()

        # Regression target: next day's close price
        df['target_price'] = df['close'].shift(-1)

        # Classification target: price movement direction
        df['price_change'] = df['target_price'] - df['close']
        df['target_signal'] = np.where(df['price_change'] > 0, 1,  # Buy
                                     np.where(df['price_change'] < 0, -1, 0))  # Sell/Hold

        # Drop rows with NaN targets
        df = df.dropna()

        # Features for modeling (exclude target columns)
        feature_cols = [col for col in df.columns if not col.startswith('target_') and col != 'price_change']
        X = df[feature_cols]

        # Targets
        y_reg = df['target_price']
        y_cls = df['target_signal']

        logger.info(f"[SUCCESS] Prepared {len(X)} samples for modeling")
        logger.info(f"  Regression target: next day price")
        logger.info(f"  Classification target: {y_cls.value_counts().to_dict()}")

        return X, y_reg, y_cls

    def train_regression_model(self, hyperparameter_tune: bool = True) -> Tuple[float, Dict[str, float]]:
        """Train XGBoost regression model for price prediction"""
        if self.feature_data is None:
            logger.error("[ERROR] No feature data available")
            return 0.0, {}

        X, y_reg, _ = self.prepare_targets()

        if X.empty or y_reg.empty:
            logger.error("[ERROR] No valid training data")
            return 0.0, {}

        # Split data (time series split)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y_reg[:train_size], y_reg[train_size:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if hyperparameter_tune:
            # Hyperparameter tuning
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }

            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                param_grid,
                cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )

            grid_search.fit(X_train_scaled, y_train)
            self.regression_model = grid_search.best_estimator_

            logger.info(f"[TUNING] Best regression params: {grid_search.best_params_}")

        else:
            # Default model
            self.regression_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=5,
                learning_rate=0.1,
                n_estimators=200,
                random_state=42
            )
            self.regression_model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.regression_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)

        # Feature importance
        feature_importance = dict(zip(X.columns, self.regression_model.feature_importances_))

        logger.info(f"[SUCCESS] Regression model trained - MAE: {mae:.4f}")

        return mae, feature_importance

    def train_classification_model(self, hyperparameter_tune: bool = True) -> Tuple[float, Dict[str, float]]:
        """Train XGBoost classification model for trading signals"""
        if self.feature_data is None:
            logger.error("[ERROR] No feature data available")
            return 0.0, {}

        X, _, y_cls = self.prepare_targets()

        if X.empty or y_cls.empty:
            logger.error("[ERROR] No valid training data")
            return 0.0, {}

        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y_cls[:train_size], y_cls[train_size:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if hyperparameter_tune:
            # Hyperparameter tuning
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }

            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42),
                param_grid,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1
            )

            grid_search.fit(X_train_scaled, y_train)
            self.classification_model = grid_search.best_estimator_

            logger.info(f"[TUNING] Best classification params: {grid_search.best_params_}")

        else:
            # Default model
            self.classification_model = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=3,
                max_depth=5,
                learning_rate=0.1,
                n_estimators=200,
                random_state=42
            )
            self.classification_model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.classification_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Feature importance
        feature_importance = dict(zip(X.columns, self.classification_model.feature_importances_))

        logger.info(f"[SUCCESS] Classification model trained - Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Sell', 'Hold', 'Buy'])}")

        return accuracy, feature_importance

    def predict_price(self, features: pd.DataFrame) -> np.ndarray:
        """Predict next day price using regression model"""
        if self.regression_model is None:
            logger.error("[ERROR] Regression model not trained")
            return np.array([])

        features_scaled = self.scaler.transform(features)
        predictions = self.regression_model.predict(features_scaled)

        return predictions

    def predict_signals(self, features: pd.DataFrame) -> np.ndarray:
        """Predict trading signals using classification model"""
        if self.classification_model is None:
            logger.error("[ERROR] Classification model not trained")
            return np.array([])

        features_scaled = self.scaler.transform(features)
        predictions = self.classification_model.predict(features_scaled)

        return predictions

    def backtest_strategy(self, initial_capital: float = 10000.0) -> Tuple[float, List[Dict[str, Any]]]:
        """Backtest the trading strategy"""
        if self.feature_data is None or self.classification_model is None:
            logger.error("[ERROR] No data or model available for backtesting")
            return 0.0, []

        df = self.feature_data.copy()
        X, _, _ = self.prepare_targets()

        # Get predictions for the entire dataset
        signals = self.predict_signals(X)

        # Simulate trading
        capital = initial_capital
        position = 0  # 0: no position, 1: long
        trades = []

        for i, (idx, row) in enumerate(df.iterrows()):
            if i >= len(signals):
                break

            signal = signals[i]
            current_price = row['close']

            # Execute trades based on signals
            if signal == 1 and position == 0:  # Buy signal
                position = 1
                entry_price = current_price
                trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'date': idx,
                    'capital': capital
                })
                logger.debug(f"BUY at {current_price:.2f}")

            elif signal == -1 and position == 1:  # Sell signal
                position = 0
                exit_price = current_price
                pnl = (exit_price - entry_price) / entry_price * capital
                capital += pnl
                trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'date': idx,
                    'pnl': pnl,
                    'capital': capital
                })
                logger.debug(f"SELL at {current_price:.2f}, PnL: {pnl:.2f}")

        # Calculate total return
        total_return = (capital - initial_capital) / initial_capital * 100

        logger.info(f"[BACKTEST] Initial: ${initial_capital:.2f}, Final: ${capital:.2f}, Return: {total_return:.2f}%")
        logger.info(f"[BACKTEST] Total trades: {len(trades)}")

        return total_return, trades

    def save_models(self):
        """Save trained models to disk"""
        if self.regression_model:
            with open(os.path.join(self.models_dir, f'{self.symbol}_regression.pkl'), 'wb') as f:
                pickle.dump(self.regression_model, f)

        if self.classification_model:
            with open(os.path.join(self.models_dir, f'{self.symbol}_classification.pkl'), 'wb') as f:
                pickle.dump(self.classification_model, f)

        # Save scaler
        with open(os.path.join(self.models_dir, f'{self.symbol}_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        logger.info(f"[SAVE] Models saved for {self.symbol}")

    def load_models(self):
        """Load trained models from disk"""
        try:
            with open(os.path.join(self.models_dir, f'{self.symbol}_regression.pkl'), 'rb') as f:
                self.regression_model = pickle.load(f)

            with open(os.path.join(self.models_dir, f'{self.symbol}_classification.pkl'), 'rb') as f:
                self.classification_model = pickle.load(f)

            with open(os.path.join(self.models_dir, f'{self.symbol}_scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)

            logger.info(f"[LOAD] Models loaded for {self.symbol}")
            return True

        except FileNotFoundError:
            logger.warning(f"[LOAD] No saved models found for {self.symbol}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained models"""
        info = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'data_points': len(self.feature_data) if self.feature_data is not None else 0,
            'regression_trained': self.regression_model is not None,
            'classification_trained': self.classification_model is not None,
            'features': list(self.feature_data.columns) if self.feature_data is not None else []
        }

        return info

# Standalone functions for easy usage

async def create_xgboost_model(symbol: str, start_date: str = '2020-01-01') -> XGBoostTradingModel:
    """Factory function to create and initialize XGBoost trading model"""
    model = XGBoostTradingModel(symbol=symbol, start_date=start_date)

    # Load and prepare data
    await model.load_data()
    if not model.raw_data.empty:
        model.add_technical_indicators()
        model.prepare_targets()

    return model

def xgboost_example():
    """Example usage of XGBoost trading model"""
    async def run_example():
        # Initialize XGBoost trading model
        model = XGBoostTradingModel(symbol='AAPL', start_date='2020-01-01')

        # Load and prepare data
        await model.load_data()
        model.add_technical_indicators()
        model.prepare_targets()

        # Train regression model for price prediction
        mae, reg_importance = model.train_regression_model()

        # Train classification model for trading signals
        accuracy, cls_importance = model.train_classification_model()

        # Backtest the strategy
        total_return, trades = model.backtest_strategy()

        print(f"Regression MAE: {mae:.4f}")
        print(f"Classification Accuracy: {accuracy:.4f}")
        print(f"Backtest Return: {total_return:.2f}%")
        print(f"Number of trades: {len(trades)}")

        # Save models
        model.save_models()

        return model

    # Run the example
    return asyncio.run(run_example())

if __name__ == "__main__":
    # Run example
    model = xgboost_example()</content>
<parameter name="filePath">c:\Users\nzcon\VSPython\ai_advancements\xgboost_trading_model.py