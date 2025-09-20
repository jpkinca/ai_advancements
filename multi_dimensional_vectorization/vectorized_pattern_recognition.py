# Stock Pattern Recognition using Vectorization
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import ta
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class StockPatternRecognizer:
    def __init__(self, lookback_window=20):
        self.lookback_window = lookback_window
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.pattern_clusters = None
        
    def create_technical_vectors(self, df):
        """Convert OHLCV data into technical indicator vectors"""
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Technical indicators using ta library
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        df['bb_high'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['bb_low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        df['bb_position'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        
        # Volatility measures
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df
    
    def extract_pattern_windows(self, df, target_column='Close'):
        """Extract sliding windows of patterns as vectors"""
        vectors = []
        dates = []
        
        feature_cols = ['returns', 'high_low_ratio', 'close_open_ratio', 'rsi', 
                       'macd', 'bb_position', 'volume_ratio', 'atr', 'volatility']
        
        for i in range(self.lookback_window, len(df)):
            window_data = df[feature_cols].iloc[i-self.lookback_window:i].values.flatten()
            if not np.isnan(window_data).any():  # Skip windows with NaN values
                vectors.append(window_data)
                dates.append(df.index[i])
                
        return np.array(vectors), dates
    
    def fit_pattern_clusters(self, symbols, start_date='2020-01-01'):
        """Train the pattern recognition model on multiple stocks"""
        all_vectors = []
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date)
            df = self.create_technical_vectors(df)
            vectors, _ = self.extract_pattern_windows(df)
            all_vectors.extend(vectors)
        
        all_vectors = np.array(all_vectors)
        
        # Normalize features
        all_vectors_scaled = self.scaler.fit_transform(all_vectors)
        
        # Reduce dimensionality
        all_vectors_pca = self.pca.fit_transform(all_vectors_scaled)
        
        # Cluster patterns
        self.pattern_clusters = KMeans(n_clusters=15, random_state=42)
        self.pattern_clusters.fit(all_vectors_pca)
        
        print(f"Trained on {len(all_vectors)} pattern windows")
        return self
    
    def find_similar_patterns(self, target_symbol, current_date=None, top_k=5):
        """Find historical patterns similar to current market conditions"""
        if current_date is None:
            current_date = datetime.now()
        
        # Get recent data for target symbol
        stock = yf.Ticker(target_symbol)
        df = stock.history(start=current_date - timedelta(days=100))
        df = self.create_technical_vectors(df)
        
        vectors, dates = self.extract_pattern_windows(df)
        
        if len(vectors) == 0:
            return "Insufficient data for pattern analysis"
        
        # Get the most recent pattern
        current_pattern = vectors[-1].reshape(1, -1)
        current_pattern_scaled = self.scaler.transform(current_pattern)
        current_pattern_pca = self.pca.transform(current_pattern_scaled)
        
        # Find cluster
        cluster_id = self.pattern_clusters.predict(current_pattern_pca)[0]
        
        # Calculate similarities with all historical patterns
        all_patterns_scaled = self.scaler.transform(vectors)
        all_patterns_pca = self.pca.transform(all_patterns_scaled)
        
        similarities = cosine_similarity(current_pattern_pca, all_patterns_pca)[0]
        
        # Get top similar patterns
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'date': dates[idx],
                'similarity': similarities[idx],
                'cluster': self.pattern_clusters.predict(all_patterns_pca[idx:idx+1])[0]
            })
        
        return results
    
    def predict_next_move(self, target_symbol, lookforward_days=5):
        """Predict likely price movement based on historical similar patterns"""
        similar_patterns = self.find_similar_patterns(target_symbol)
        
        # Analyze what happened after similar patterns in the past
        stock = yf.Ticker(target_symbol)
        df = stock.history(start='2020-01-01')
        
        future_returns = []
        for pattern in similar_patterns:
            pattern_date = pattern['date']
            try:
                future_date = pattern_date + timedelta(days=lookforward_days)
                current_price = df.loc[pattern_date, 'Close']
                future_price = df.loc[future_date, 'Close']
                return_pct = (future_price - current_price) / current_price * 100
                future_returns.append(return_pct)
            except:
                continue
        
        if future_returns:
            avg_return = np.mean(future_returns)
            return_std = np.std(future_returns)
            confidence = len(future_returns) / len(similar_patterns)
            
            return {
                'predicted_return': avg_return,
                'uncertainty': return_std,
                'confidence': confidence,
                'historical_examples': len(future_returns)
            }
        
        return "Insufficient historical data for prediction"

# Example usage
if __name__ == "__main__":
    # Initialize pattern recognizer
    recognizer = StockPatternRecognizer(lookback_window=15)
    
    # Train on multiple stocks
    training_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META']
    recognizer.fit_pattern_clusters(training_symbols)
    
    # Find similar patterns for a specific stock
    target_stock = 'AAPL'
    similar_patterns = recognizer.find_similar_patterns(target_stock)
    
    print(f"\nSimilar patterns for {target_stock}:")
    for i, pattern in enumerate(similar_patterns):
        print(f"{i+1}. Date: {pattern['date'].strftime('%Y-%m-%d')}, "
              f"Similarity: {pattern['similarity']:.3f}, "
              f"Cluster: {pattern['cluster']}")
    
    # Predict next move
    prediction = recognizer.predict_next_move(target_stock)
    print(f"\nPrediction for {target_stock}:")
    print(prediction)