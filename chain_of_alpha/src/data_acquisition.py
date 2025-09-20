"""
Data Acquisition Module for Chain-of-Alpha MVP

Handles fetching and preprocessing market data for alpha factor generation.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class DataAcquisition:
    """
    Handles market data acquisition and preprocessing
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_cache = {}

        # Create output directory if it doesn't exist
        os.makedirs(config.get('output_dir', 'outputs'), exist_ok=True)

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch market data for all configured tickers

        Returns:
            Combined DataFrame with MultiIndex (ticker, date)
        """
        logger.info(f"Fetching data for {len(self.config['tickers'])} tickers...")

        market_data = {}

        for ticker in self.config['tickers']:
            try:
                logger.info(f"Fetching data for {ticker}...")
                df = self._fetch_single_ticker(ticker)

                if df is not None and len(df) > 0:
                    # Preprocess the data
                    df = self._preprocess_data(df, ticker)
                    market_data[ticker] = df
                    logger.info(f"✓ {ticker}: {len(df)} data points")
                else:
                    logger.warning(f"✗ No data retrieved for {ticker}")

            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                continue

        logger.info(f"Successfully fetched data for {len(market_data)}/{len(self.config['tickers'])} tickers")

        # Combine all data into a single DataFrame with MultiIndex
        if market_data:
            combined_data = self._combine_market_data(market_data)
            return combined_data
        else:
            logger.error("No data retrieved for any tickers")
            return pd.DataFrame()

    def _combine_market_data(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine individual ticker DataFrames into a single MultiIndex DataFrame
        """
        try:
            # Add ticker index to each DataFrame
            dfs_with_ticker = []
            for ticker, df in market_data.items():
                df_copy = df.copy()
                df_copy['ticker'] = ticker
                df_copy.set_index('ticker', append=True, inplace=True)
                df_copy = df_copy.reorder_levels(['ticker', df_copy.index.names[0]])
                dfs_with_ticker.append(df_copy)

            # Concatenate all DataFrames - use keys parameter to avoid MultiIndex columns
            combined = pd.concat(dfs_with_ticker, axis=0, sort=False)

            # Ensure we don't have MultiIndex columns
            if isinstance(combined.columns, pd.MultiIndex):
                # Flatten column names by taking the first level (the actual column names)
                combined.columns = [col[0] if isinstance(col, tuple) else col for col in combined.columns]

            # Sort by ticker and date
            combined = combined.sort_index()

            return combined

        except Exception as e:
            logger.error(f"Error combining market data: {e}")
            return pd.DataFrame()

    def _fetch_single_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single ticker using yfinance
        """
        try:
            # Download data
            data = yf.download(
                ticker,
                start=self.config['start_date'],
                end=self.config['end_date'],
                progress=False,
                auto_adjust=True,
                prepost=False
            )

            if data is None or data.empty:
                logger.warning(f"No data available for {ticker}")
                return None

            # Flatten MultiIndex columns if they exist (common with yfinance)
            if isinstance(data.columns, pd.MultiIndex):
                # Take the first level of the MultiIndex (the actual column names)
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                logger.warning(f"Missing required columns for {ticker}")
                return None

            return data

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None

    def _preprocess_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Preprocess raw market data for factor generation
        """
        # Rename columns to lowercase for consistency
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Remove any rows with NaN values
        df = df.dropna()

        # Ensure we have enough data
        if len(df) < 100:
            logger.warning(f"Insufficient data for {ticker}: {len(df)} rows")
            return df

        # Add basic derived features
        df = self._add_basic_features(df)

        # Sort by date
        df = df.sort_index()

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical features for factor generation
        """
        try:
            # Price returns
            df = df.copy()  # Work on a copy to avoid issues
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Volume features
            df['volume_ma_5'] = df['volume'].rolling(5).mean()
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            # Calculate volume_ratio safely
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']

            # Price momentum
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

            # Volatility
            df['volatility_20'] = df['returns'].rolling(20).std()

            # Simple moving averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()

            # RSI (Relative Strength Index)
            df['rsi'] = self._calculate_rsi(df['close'])

            # MACD
            macd, signal, hist = self._calculate_macd(df['close'])
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist

            return df

        except Exception as e:
            logger.error(f"Error adding basic features: {e}")
            return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff().astype(float)
        gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series,
                        fast_period: int = 12,
                        slow_period: int = 26,
                        signal_period: int = 9) -> tuple:
        """Calculate MACD"""
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def save_data(self, market_data: Dict[str, pd.DataFrame], filename: str = "market_data.pkl"):
        """Save market data to disk"""
        import pickle
        filepath = os.path.join(self.config.get('output_dir', 'outputs'), filename)

        with open(filepath, 'wb') as f:
            pickle.dump(market_data, f)

        logger.info(f"Market data saved to {filepath}")

    def load_data(self, filename: str = "market_data.pkl") -> Dict[str, pd.DataFrame]:
        """Load market data from disk"""
        import pickle
        filepath = os.path.join(self.config.get('output_dir', 'outputs'), filename)

        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Market data loaded from {filepath}")
            return data
        else:
            logger.warning(f"Data file {filepath} not found")
            return {}

    def get_data_summary(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get summary statistics of the market data"""
        summary = {
            'num_tickers': len(market_data),
            'tickers': list(market_data.keys()),
            'date_range': {},
            'data_points': {},
            'features': set()
        }

        for ticker, df in market_data.items():
            summary['data_points'][ticker] = len(df)
            summary['date_range'][ticker] = {
                'start': df.index.min().strftime('%Y-%m-%d'),
                'end': df.index.max().strftime('%Y-%m-%d')
            }

            if len(summary['features']) == 0:
                summary['features'] = set(df.columns)
            else:
                summary['features'] = summary['features'].intersection(set(df.columns))

        summary['features'] = list(summary['features'])
        return summary