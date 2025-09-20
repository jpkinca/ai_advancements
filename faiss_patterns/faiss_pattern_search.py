#!/usr/bin/env python3
"""
FAISS Pattern Search for Trading Patterns
Demonstrates how to use FAISS for similarity search on trading patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

# Import our pattern generators
from canslim_sepa_pattern_generator import PatternMatchingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS is available")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using fallback similarity search")

class FaissPatternSearch:
    """FAISS-based pattern similarity search for trading patterns."""
    
    def __init__(self, dimension: int = 50):
        self.dimension = dimension
        self.index = None
        self.pattern_metadata = []
        
        if FAISS_AVAILABLE:
            # Create FAISS index for cosine similarity
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)
            logger.info(f"Initialized FAISS index with dimension {dimension}")
        else:
            logger.info("Using fallback search without FAISS")
    
    def add_patterns(self, patterns: np.ndarray, metadata: List[Dict]):
        """Add trading patterns to the search index.
        
        Args:
            patterns: Array of shape (n_patterns, dimension)
            metadata: List of pattern metadata dictionaries
        """
        if patterns.shape[1] != self.dimension:
            raise ValueError(f"Pattern dimension {patterns.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Normalize patterns for cosine similarity
        patterns_norm = patterns / (np.linalg.norm(patterns, axis=1, keepdims=True) + 1e-8)
        
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(patterns_norm.astype(np.float32))
        else:
            # Fallback: store patterns in memory
            if not hasattr(self, '_patterns'):
                self._patterns = patterns_norm
            else:
                self._patterns = np.vstack([self._patterns, patterns_norm])
        
        self.pattern_metadata.extend(metadata)
        logger.info(f"Added {len(patterns)} patterns to index")
    
    def search(self, query_pattern: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Dict]]:
        """Search for similar patterns.
        
        Args:
            query_pattern: Pattern to search for (shape: dimension,)
            k: Number of similar patterns to return
            
        Returns:
            Tuple of (similarities, metadata_list)
        """
        query_norm = query_pattern / (np.linalg.norm(query_pattern) + 1e-8)
        query_norm = query_norm.reshape(1, -1).astype(np.float32)
        
        if FAISS_AVAILABLE and self.index is not None:
            similarities, indices = self.index.search(query_norm, k)
            similarities = similarities[0]
            indices = indices[0]
            
            # Filter out invalid indices
            valid_mask = indices >= 0
            similarities = similarities[valid_mask]
            indices = indices[valid_mask]
            
            metadata_results = [self.pattern_metadata[i] for i in indices]
        else:
            # Fallback similarity search
            if not hasattr(self, '_patterns'):
                return np.array([]), []
            
            similarities = np.dot(self._patterns, query_norm.T).flatten()
            top_k_indices = np.argsort(similarities)[::-1][:k]
            similarities = similarities[top_k_indices]
            metadata_results = [self.pattern_metadata[i] for i in top_k_indices]
        
        return similarities, metadata_results

def extract_pattern_features(pattern_data: Dict) -> np.ndarray:
    """Extract numerical features from pattern data for FAISS indexing."""
    features = []
    
    # Extract features from different pattern types
    for pattern_type, pattern_info in pattern_data.items():
        if isinstance(pattern_info, dict):
            for key, value in pattern_info.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, bool):
                    features.append(float(value))
                elif isinstance(value, np.ndarray) and value.size > 0:
                    # Take first few elements of arrays
                    arr_features = value.flatten()[:5]  # Limit array features
                    features.extend(arr_features.tolist())
                elif isinstance(value, list) and len(value) > 0:
                    # Take first few elements of lists
                    list_features = value[:5] if all(isinstance(x, (int, float)) for x in value[:5]) else []
                    features.extend(list_features)
        elif isinstance(pattern_info, (int, float)):
            features.append(float(pattern_info))
        elif isinstance(pattern_info, bool):
            features.append(float(pattern_info))
    
    # Pad or truncate to fixed dimension
    target_dim = 50
    if len(features)  Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate sample trading data for testing."""
    np.random.seed(42)  # For reproducibility
    
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    # Generate price data with trend
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_days)  # Small positive drift
    prices = base_price * np.cumprod(1 + returns)
    
    price_data = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'Close': prices,
        'Adj Close': prices
    })
    
    # Ensure OHLC relationships are correct
    price_data['High'] = price_data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    price_data['Low'] = price_data[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    volume_data = pd.DataFrame({
        'Date': dates,
        'Volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    market_data = pd.DataFrame({
        'Date': dates,
        'SPY_Close': 400 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n_days))
    })
    
    return price_data, volume_data, market_data

def main():
    """Main function demonstrating FAISS pattern search."""
    logger.info("Starting FAISS Pattern Search Demo...")
    
    # Initialize pattern search
    pattern_search = FaissPatternSearch(dimension=50)
    
    # Generate sample data and patterns
    logger.info("Generating sample data...")
    price_data, volume_data, market_data = generate_sample_data(100)
    
    # Initialize pattern engine
    engine = PatternMatchingEngine()
    
    # Generate patterns for multiple "stocks"
    all_patterns = []
    all_metadata = []
    
    for stock_id in range(5):  # Simulate 5 different stocks
        logger.info(f"Generating patterns for stock {stock_id + 1}/5...")
        
        # Add some noise to make different "stocks"
        noise_factor = 0.1 * stock_id
        noisy_price = price_data.copy()
        noisy_price['Close'] *= (1 + np.random.normal(0, noise_factor, len(price_data)))
        
        # Generate CANSLIM patterns
        canslim_patterns = engine.canslim_generator.generate_canslim_patterns(
            noisy_price, volume_data, market_data
        )
        
        # Generate SEPA patterns
        sepa_patterns = engine.sepa_generator.generate_sepa_patterns(
            noisy_price, volume_data, market_data
        )
        
        # Combine patterns
        combined_patterns = {**canslim_patterns, **sepa_patterns}
        
        # Extract features for FAISS
        pattern_features = extract_pattern_features(combined_patterns)
        all_patterns.append(pattern_features)
        
        # Store metadata
        metadata = {
            'stock_id': f'STOCK_{stock_id:03d}',
            'timestamp': pd.Timestamp.now().isoformat(),
            'pattern_type': 'combined_canslim_sepa',
            'canslim_score': np.random.random(),
            'sepa_score': np.random.random(),
            'patterns': combined_patterns
        }
        all_metadata.append(metadata)
    
    # Add patterns to FAISS index
    logger.info("Adding patterns to FAISS index...")
    pattern_array = np.array(all_patterns)
    pattern_search.add_patterns(pattern_array, all_metadata)
    
    # Perform similarity search
    logger.info("Performing similarity search...")
    query_pattern = all_patterns[0]  # Use first pattern as query
    similarities, similar_metadata = pattern_search.search(query_pattern, k=3)
    
    # Display results
    logger.info("Search Results:")
    logger.info("=" * 50)
    for i, (sim, meta) in enumerate(zip(similarities, similar_metadata)):
        logger.info(f"Rank {i+1}: Stock {meta['stock_id']}")
        logger.info(f"  Similarity: {sim:.4f}")
        logger.info(f"  CANSLIM Score: {meta['canslim_score']:.4f}")
        logger.info(f"  SEPA Score: {meta['sepa_score']:.4f}")
        logger.info(f"  Timestamp: {meta['timestamp']}")
        logger.info("-" * 30)
    
    logger.info("FAISS Pattern Search Demo completed successfully!")

if __name__ == "__main__":
    main()
