"""
XGBoost Trading Model Implementation

This module implements XGBoost-based predictive models for algorithmic trading,
including regression for price prediction and classification for buy/sell signals.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import xgboost as xgb
import talib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class XGBoostTradingModel:
    """
    XGBoost-based trading model for stock price prediction and signal generation.
    """

    def __init__(self, symbol='AAPL', start_date='2020-01-01', end_date=None):
        """
        Initialize the trading model.

        Args:
            symbol (str): Stock ticker symbol
            start_date (str): Start date for data collection
            end_date (str): End date for data collection (default: today)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.features = None
        self.regressor = None
        self.classifier = None

    def load_data(self):
        """
        Load historical stock data using yfinance.
        """
        print(f"Loading data for {self.symbol} from {self.start_date} to {self.end_date}")
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)

        if self.data.empty:
            raise ValueError(f"No data found for symbol {self.symbol}")

        print(f"Loaded {len(self.data)} data points")
        return self.data

    def add_technical_indicators(self):
        """
        Add technical indicators as features using TA-Lib.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.data.copy()

        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()

        # Technical indicators
        # RSI
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'],
                                                                   fastperiod=12,
                                                                   slowperiod=26,
                                                                   signalperiod=9)

        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'],
                                                                       timeperiod=20,
                                                                       nbdevup=2,
                                                                       nbdevdn=2,
                                                                       matype=0)

        # Moving Averages
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)

        # Stochastic Oscillator
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'],
                                                   fastk_period=14, slowk_period=3,
                                                   slowd_period=3)

        # Williams %R
        df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

        # Average True Range (volatility)
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)

        # Drop NaN values created by indicators and lags
        df.dropna(inplace=True)

        self.features = df
        print(f"Added {len(df.columns)} features including technical indicators")
        return self.features

    def prepare_targets(self):
        """
        Prepare target variables for regression and classification.
        """
        if self.features is None:
            raise ValueError("Features not prepared. Call add_technical_indicators() first.")

        # Regression target: Next day return
        self.features['Target_Return'] = self.features['Returns'].shift(-1)

        # Classification target: Buy/Sell/Hold signal
        # Buy if next day return > 1%, Sell if < -1%, Hold otherwise
        conditions = [
            (self.features['Target_Return'] > 0.01),
            (self.features['Target_Return'] < -0.01)
        ]
        choices = [1, -1]  # 1: Buy, -1: Sell
        self.features['Target_Signal'] = np.select(conditions, choices, default=0)  # 0: Hold

        # Drop the last row since we don't have target for it
        self.features.dropna(inplace=True)

        print(f"Prepared targets: {len(self.features)} samples")
        print(f"Signal distribution: {self.features['Target_Signal'].value_counts()}")

    def train_regression_model(self, test_size=0.2):
        """
        Train XGBoost regressor for price prediction.

        Args:
            test_size (float): Proportion of data for testing
        """
        if self.features is None:
            raise ValueError("Features not prepared. Call prepare_targets() first.")

        # Feature columns (exclude target and non-feature columns)
        exclude_cols = ['Target_Return', 'Target_Signal']
        feature_cols = [col for col in self.features.columns if col not in exclude_cols]

        X = self.features[feature_cols]
        y = self.features['Target_Return']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=42, shuffle=False)

        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

        self.regressor = xgb.XGBRegressor(**params)

        print("Training XGBoost Regressor...")
        self.regressor.fit(X_train, y_train,
                          eval_set=[(X_test, y_test)],
                          early_stopping_rounds=10,
                          verbose=False)

        # Predictions
        y_pred = self.regressor.predict(X_test)

        # Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        print(".4f")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.regressor.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 important features for regression:")
        print(feature_importance.head(10))

        return mae, feature_importance

    def train_classification_model(self, test_size=0.2):
        """
        Train XGBoost classifier for buy/sell signals.

        Args:
            test_size (float): Proportion of data for testing
        """
        if self.features is None:
            raise ValueError("Features not prepared. Call prepare_targets() first.")

        # Feature columns
        exclude_cols = ['Target_Return', 'Target_Signal']
        feature_cols = [col for col in self.features.columns if col not in exclude_cols]

        X = self.features[feature_cols]
        y = self.features['Target_Signal']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=42, shuffle=False)

        # XGBoost parameters
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,  # Buy, Hold, Sell
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

        self.classifier = xgb.XGBClassifier(**params)

        print("Training XGBoost Classifier...")
        self.classifier.fit(X_train, y_train,
                           eval_set=[(X_test, y_test)],
                           early_stopping_rounds=10,
                           verbose=False)

        # Predictions
        y_pred = self.classifier.predict(X_test)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(".4f")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Sell', 'Hold', 'Buy']))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 important features for classification:")
        print(feature_importance.head(10))

        return accuracy, feature_importance

    def plot_feature_importance(self, importance_df, title, top_n=20):
        """
        Plot feature importance.

        Args:
            importance_df (pd.DataFrame): Feature importance dataframe
            title (str): Plot title
            top_n (int): Number of top features to plot
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature',
                   data=importance_df.head(top_n),
                   palette='viridis')
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    def backtest_strategy(self, initial_capital=10000):
        """
        Simple backtest of the trading strategy using classifier predictions.

        Args:
            initial_capital (float): Starting capital for backtesting
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_classification_model() first.")

        # Use test data for backtesting
        exclude_cols = ['Target_Return', 'Target_Signal']
        feature_cols = [col for col in self.features.columns if col not in exclude_cols]

        # Get the last part as test data (assuming chronological split)
        test_size = 0.2
        split_idx = int(len(self.features) * (1 - test_size))

        X_test = self.features[feature_cols].iloc[split_idx:]
        y_test = self.features['Target_Signal'].iloc[split_idx:]
        prices = self.features['Close'].iloc[split_idx:]

        # Predictions
        signals = self.classifier.predict(X_test)

        # Simple strategy: Buy on signal 1, Sell on signal -1, Hold on 0
        capital = initial_capital
        position = 0  # 0: no position, 1: long
        trades = []

        for i, (signal, price, actual_return) in enumerate(zip(signals, prices, y_test)):
            if signal == 1 and position == 0:  # Buy signal
                position = capital / price
                capital = 0
                trades.append(('BUY', price, i))
            elif signal == -1 and position > 0:  # Sell signal
                capital = position * price
                position = 0
                trades.append(('SELL', price, i))

        # Final position
        if position > 0:
            capital = position * prices.iloc[-1]
            trades.append(('FINAL_SELL', prices.iloc[-1], len(prices)-1))

        # Calculate returns
        total_return = (capital - initial_capital) / initial_capital * 100

        print(f"\nBacktest Results:")
        print(f"Initial Capital: ${initial_capital}")
        print(f"Final Capital: ${capital:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {len(trades)}")

        return total_return, trades

def main():
    """
    Main function to demonstrate the XGBoost trading model.
    """
    # Initialize model
    model = XGBoostTradingModel(symbol='AAPL', start_date='2020-01-01')

    # Load and prepare data
    model.load_data()
    model.add_technical_indicators()
    model.prepare_targets()

    # Train models
    reg_mae, reg_importance = model.train_regression_model()
    cls_accuracy, cls_importance = model.train_classification_model()

    # Backtest
    total_return, trades = model.backtest_strategy()

    # Plot feature importance (optional)
    # model.plot_feature_importance(reg_importance, 'Regression Feature Importance')
    # model.plot_feature_importance(cls_importance, 'Classification Feature Importance')

    print("\nXGBoost Trading Model Implementation Complete!")

if __name__ == "__main__":
    main()