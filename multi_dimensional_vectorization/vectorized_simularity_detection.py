# Stock Similarity Analysis using Multidimensional Vectorization
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import faiss
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockSimilarityAnalyzer:
    def __init__(self):
        self.stock_vectors = {}
        self.stock_symbols = []
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=20)
        self.faiss_index = None
        self.vector_dim = None
        
    def create_fundamental_vector(self, symbol):
        """Create vector from fundamental data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key fundamental metrics
            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'profit_margins': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'return_on_assets': info.get('returnOnAssets', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'beta': info.get('beta', 1.0),
                'dividend_yield': info.get('dividendYield', 0),
            }
            
            return np.array(list(fundamentals.values()))
        except:
            return np.zeros(16)  # Return zero vector if data unavailable
    
    def create_technical_vector(self, symbol, period='1y'):
        """Create vector from technical analysis features"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if len(df)  0).sum() / len(returns)
            big_moves = (abs(returns) > returns.std() * 2).sum() / len(returns)
            
            # Gap analysis
            gaps = df['Open'] - df['Close'].shift(1)
            gap_frequency = (abs(gaps) > df['Close'].shift(1) * 0.02).sum() / len(gaps)
            avg_gap_size = gaps.mean() / df['Close'].mean()
            
            technical_features = [
                price_volatility, avg_return, skewness, kurtosis, max_drawdown,
                avg_volume_change, volume_volatility, price_trend, recent_momentum,
                current_position, up_days, big_moves, gap_frequency, avg_gap_size,
                len(df)  # data availability
            ]
            
            return np.array(technical_features)
        except:
            return np.zeros(15)
    
    def create_sector_industry_vector(self, symbol):
        """Create one-hot encoded vector for sector and industry"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Common sectors
            sectors = [
                'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
                'Communication Services', 'Industrials', 'Consumer Defensive',
                'Energy', 'Utilities', 'Real Estate', 'Materials', 'Basic Materials'
            ]
            
            # Common industries (simplified)
            industries = [
                'Software', 'Semiconductors', 'Banks', 'Biotechnology', 
                'Retail', 'Automotive', 'Oil & Gas', 'Pharmaceuticals',
                'Insurance', 'Real Estate', 'Aerospace', 'Media'
            ]
            
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            
            # One-hot encode sector
            sector_vector = np.zeros(len(sectors))
            for i, s in enumerate(sectors):
                if s in sector:
                    sector_vector[i] = 1
                    break
            
            # One-hot encode industry
            industry_vector = np.zeros(len(industries))
            for i, ind in enumerate(industries):
                if ind in industry:
                    industry_vector[i] = 1
                    break
            
            return np.concatenate([sector_vector, industry_vector])
        except:
            return np.zeros(24)  # 12 sectors + 12 industries
    
    def build_stock_vectors(self, symbols):
        """Build comprehensive vectors for all stocks"""
        print("Building stock vectors...")
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            
            # Create different types of vectors
            fundamental_vec = self.create_fundamental_vector(symbol)
            technical_vec = self.create_technical_vector(symbol)
            sector_vec = self.create_sector_industry_vector(symbol)
            
            # Combine all vectors
            combined_vector = np.concatenate([
                fundamental_vec, technical_vec, sector_vec
            ])
            
            # Handle NaN values
            combined_vector = np.nan_to_num(combined_vector, 0)
            
            self.stock_vectors[symbol] = combined_vector
        
        self.stock_symbols = symbols
        print(f"Built vectors for {len(symbols)} stocks")
    
    def normalize_and_reduce_dimensions(self):
        """Normalize vectors and optionally reduce dimensions"""
        # Convert to matrix
        vector_matrix = np.array([self.stock_vectors[symbol] for symbol in self.stock_symbols])
        
        # Normalize
        vector_matrix_scaled = self.scaler.fit_transform(vector_matrix)
        
        # Optional: Reduce dimensions with PCA
        if vector_matrix_scaled.shape[1] > 20:
            vector_matrix_pca = self.pca.fit_transform(vector_matrix_scaled)
        else:
            vector_matrix_pca = vector_matrix_scaled
        
        # Update stock vectors with normalized/reduced versions
        for i, symbol in enumerate(self.stock_symbols):
            self.stock_vectors[symbol] = vector_matrix_pca[i]
        
        self.vector_dim = vector_matrix_pca.shape[1]
        print(f"Vector dimension: {self.vector_dim}")
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        # Convert vectors to FAISS format
        vectors = np.array([self.stock_vectors[symbol] for symbol in self.stock_symbols])
        vectors = vectors.astype('float32')
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatL2(self.vector_dim)  # L2 distance
        self.faiss_index.add(vectors)
        
        print(f"Built FAISS index with {len(vectors)} stocks")
    
    def find_similar_stocks(self, target_symbol, top_k=10, method='cosine'):
        """Find stocks most similar to target stock"""
        if target_symbol not in self.stock_vectors:
            return f"Stock {target_symbol} not found in database"
        
        target_vector = self.stock_vectors[target_symbol]
        
        if method == 'cosine':
            # Calculate cosine similarities
            similarities = {}
            for symbol in self.stock_symbols:
                if symbol != target_symbol:
                    sim = cosine_similarity(
                        target_vector.reshape(1, -1),
                        self.stock_vectors[symbol].reshape(1, -1)
                    )[0][0]
                    similarities[symbol] = sim
            
            # Sort by similarity
            sorted_stocks = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            return sorted_stocks[:top_k]
        
        elif method == 'faiss' and self.faiss_index is not None:
            # Use FAISS for fast search
            target_vector_faiss = target_vector.reshape(1, -1).astype('float32')
            distances, indices = self.faiss_index.search(target_vector_faiss, top_k + 1)  # +1 because it includes itself
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                symbol = self.stock_symbols[idx]
                if symbol != target_symbol:  # Skip the stock itself
                    similarity = 1 / (1 + distance)  # Convert distance to similarity
                    results.append((symbol, similarity))
            
            return results[:top_k]
    
    def create_similarity_matrix(self, subset_symbols=None):
        """Create similarity matrix for visualization"""
        if subset_symbols is None:
            subset_symbols = self.stock_symbols
        
        n = len(subset_symbols)
        similarity_matrix = np.zeros((n, n))
        
        for i, symbol1 in enumerate(subset_symbols):
            for j, symbol2 in enumerate(subset_symbols):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    sim = cosine_similarity(
                        self.stock_vectors[symbol1].reshape(1, -1),
                        self.stock_vectors[symbol2].reshape(1, -1)
                    )[0][0]
                    similarity_matrix[i][j] = sim
        
        return pd.DataFrame(similarity_matrix, index=subset_symbols, columns=subset_symbols)
    
    def find_portfolio_diversification_candidates(self, current_portfolio, candidate_pool, top_k=5):
        """Find stocks that would diversify the current portfolio"""
        # Calculate average vector of current portfolio
        portfolio_vectors = [self.stock_vectors[symbol] for symbol in current_portfolio if symbol in self.stock_vectors]
        
        if not portfolio_vectors:
            return "No valid stocks in current portfolio"
        
        avg_portfolio_vector = np.mean(portfolio_vectors, axis=0)
        
        # Find candidates most dissimilar to portfolio average
        dissimilarities = {}
        for symbol in candidate_pool:
            if symbol not in current_portfolio and symbol in self.stock_vectors:
                sim = cosine_similarity(
                    avg_portfolio_vector.reshape(1, -1),
                    self.stock_vectors[symbol].reshape(1, -1)
                )[0][0]
                dissimilarities[symbol] = 1 - sim  # Convert to dissimilarity
        
        # Sort by dissimilarity (highest first)
        sorted_candidates = sorted(dissimilarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:top_k]

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = StockSimilarityAnalyzer()
    
    # Define stock universe
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'CRM', 'ADBE']
    finance_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP', 'BRK-B', 'C']
    healthcare_stocks = ['JNJ', 'PFE', 'ABT', 'TMO', 'UNH', 'MDT', 'ABBV', 'BMY', 'MRK', 'LLY']
    
    all_stocks = tech_stocks + finance_stocks + healthcare_stocks
    
    # Build vectors
    analyzer.build_stock_vectors(all_stocks)
    analyzer.normalize_and_reduce_dimensions()
    analyzer.build_faiss_index()
    
    # Find similar stocks
    target_stock = 'AAPL'
    similar_stocks = analyzer.find_similar_stocks(target_stock, top_k=5)
    
    print(f"\nStocks most similar to {target_stock}:")
    for symbol, similarity in similar_stocks:
        print(f"{symbol}: {similarity:.3f}")
    
    # Portfolio diversification example
    current_portfolio = ['AAPL', 'MSFT', 'GOOGL']
    diversification_candidates = analyzer.find_portfolio_diversification_candidates(
        current_portfolio, all_stocks, top_k=5
    )
    
    print(f"\nDiversification candidates for portfolio {current_portfolio}:")
    for symbol, dissimilarity in diversification_candidates:
        print(f"{symbol}: {dissimilarity:.3f} dissimilarity")
    
    # Create similarity matrix for visualization
    tech_similarity = analyzer.create_similarity_matrix(tech_stocks[:5])
    print(f"\nTech stocks similarity matrix:")
    print(tech_similarity.round(3))