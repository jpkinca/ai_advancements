#!/usr/bin/env python3
"""
Optimized FAISS Pattern Matching System for Trading
Based on FAISS best practices for financial data

Features:
- Multiple index types for different use cases
- Proper data preprocessing with StandardScaler
- Index training and optimization
- Performance tuning with nprobe settings
- Dimensionality reduction options
- Regular index updates
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import json
import time
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import faiss
    logger.info("[SUCCESS] FAISS imported successfully")
except ImportError:
    logger.error("[ERROR] FAISS not available - install with: pip install faiss-cpu")
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    logger.info("[SUCCESS] sklearn imported for preprocessing")
except ImportError:
    logger.warning("[WARNING] sklearn not available - basic normalization will be used")
    StandardScaler = None
    PCA = None

try:
    from ib_insync import IB, Stock, util
    logger.info("[SUCCESS] ib_insync imported successfully")
except ImportError:
    logger.error("[ERROR] ib_insync not available")
    sys.exit(1)

class OptimizedFAISSPatternMatcher:
    """
    Optimized FAISS pattern matching system following best practices
    """
    
    def __init__(self, 
                 dimension: int = 32,
                 index_type: str = "flat",  # flat, ivf, hnsw
                 use_gpu: bool = False,
                 n_clusters: int = 100,
                 nprobe: int = 10,
                 normalize_features: bool = True,
                 use_pca: bool = False,
                 pca_components: int = 16):
        
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        self.normalize_features = normalize_features
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        # Storage
        self.index = None
        self.quantizer = None
        self.pattern_metadata = []
        self.training_vectors = []
        
        # Preprocessing
        self.scaler = StandardScaler() if StandardScaler else None
        self.pca = PCA(n_components=pca_components) if PCA and use_pca else None
        self.is_trained = False
        
        # Performance tracking
        self.search_times = []
        self.index_size = 0
        
    def _get_effective_dimension(self) -> int:
        """Get the effective dimension after PCA"""
        if self.use_pca and self.pca:
            return self.pca_components
        return self.dimension
    
    def create_index(self, expected_size: int = 1000) -> bool:
        """
        Create optimized FAISS index based on expected dataset size and requirements
        """
        try:
            effective_dim = self._get_effective_dimension()
            
            if self.index_type == "flat":
                # IndexFlatIP for exact cosine similarity (best for small datasets  np.ndarray:
        """
        Apply preprocessing pipeline to feature vector
        """
        try:
            # Ensure float32 for FAISS
            vector = vector.astype(np.float32)
            
            # Handle NaN values
            vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Standardization with sklearn (if available)
            if self.scaler and self.is_trained:
                vector = self.scaler.transform(vector.reshape(1, -1)).flatten()
            elif self.normalize_features:
                # Basic z-score normalization
                mean = np.mean(vector)
                std = np.std(vector)
                if std > 0:
                    vector = (vector - mean) / std
            
            # PCA dimensionality reduction (if configured)
            if self.pca and self.is_trained:
                vector = self.pca.transform(vector.reshape(1, -1)).flatten()
            
            # L2 normalization for cosine similarity
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
                
            return vector
            
        except Exception as e:
            logger.error(f"[ERROR] Vector preprocessing failed: {e}")
            return vector.astype(np.float32)
    
    def train_index(self, training_vectors: List[np.ndarray]) -> bool:
        """
        Train the index with a representative sample of data
        """
        try:
            if len(training_vectors) == 0:
                logger.warning("[WARNING] No training vectors provided")
                return False
            
            # Convert to numpy array
            training_matrix = np.vstack(training_vectors).astype(np.float32)
            logger.info(f"[TRAINING] Training with {len(training_vectors)} vectors")
            
            # Fit preprocessing components
            if self.scaler:
                self.scaler.fit(training_matrix)
                training_matrix = self.scaler.transform(training_matrix)
                logger.info(f"[TRAINING] Fitted StandardScaler")
            
            if self.pca:
                self.pca.fit(training_matrix)
                training_matrix = self.pca.transform(training_matrix)
                explained_variance = np.sum(self.pca.explained_variance_ratio_)
                logger.info(f"[TRAINING] Fitted PCA, explained variance: {explained_variance:.3f}")
            
            # Normalize for cosine similarity
            if self.normalize_features:
                faiss.normalize_L2(training_matrix)
            
            # Train index if required
            if hasattr(self.index, 'train') and not self.index.is_trained:
                self.index.train(training_matrix)
                logger.info(f"[TRAINING] Index trained successfully")
            
            self.is_trained = True
            self.training_vectors = training_vectors
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Index training failed: {e}")
            return False
    
    def add_pattern(self, vector: np.ndarray, metadata: Dict) -> bool:
        """
        Add a pattern to the index with proper preprocessing
        """
        try:
            if self.index is None:
                logger.error("[ERROR] Index not created")
                return False
            
            # Preprocess vector
            processed_vector = self.preprocess_vector(vector)
            
            # Add to index
            processed_vector = processed_vector.reshape(1, -1)
            self.index.add(processed_vector)
            self.pattern_metadata.append(metadata)
            
            self.index_size += 1
            
            logger.info(f"[SUCCESS] Added pattern for {metadata.get('symbol', 'unknown')} "
                       f"(Total: {self.index_size})")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to add pattern: {e}")
            return False
    
    def add_vpa_pattern(self, vpa_features: Dict[str, Any], metadata: Dict) -> bool:
        """
        Add VPA pattern to the index

        Args:
            vpa_features: Dictionary of VPA features
            metadata: Pattern metadata

        Returns:
            Success status
        """
        try:
            # Extract VPA feature vector
            vpa_vector = np.array([
                vpa_features.get('vol_price_ratio', 0),
                vpa_features.get('volume_imbalance', 0),
                vpa_features.get('volume_ratio', 0),
                vpa_features.get('volume_roc', 0),
                vpa_features.get('vpt', 0),
                vpa_features.get('nvi', 0),
                vpa_features.get('pvi', 0),
                vpa_features.get('volume_oscillator', 0),
                vpa_features.get('vpc_confirmed', 0),
                vpa_features.get('volume_divergence', 0)
            ], dtype=np.float32)

            # Add VPA type to metadata
            metadata['pattern_type'] = 'vpa'

            return self.add_pattern(vpa_vector, metadata)

        except Exception as e:
            logger.error(f"[ERROR] Failed to add VPA pattern: {e}")
            return False
    
    def search_similar_patterns(self, 
                               query_vector: np.ndarray, 
                               k: int = 10, 
                               return_distances: bool = True) -> Dict:
        """
        Search for similar patterns with optimized parameters
        """
        try:
            start_time = time.time()
            
            if self.index is None or self.index_size == 0:
                return {"distances": [], "indices": [], "metadata": [], "search_time": 0}
            
            # Preprocess query vector
            processed_query = self.preprocess_vector(query_vector)
            processed_query = processed_query.reshape(1, -1)
            
            # Adjust k to available patterns
            k = min(k, self.index_size)
            
            # Search
            distances, indices = self.index.search(processed_query, k)
            
            # Get metadata for results
            result_metadata = []
            for idx in indices[0]:
                if 0  Dict:
        """
        Optimize search parameters for speed vs accuracy trade-off
        """
        try:
            if self.index_type != "ivf":
                logger.info("[INFO] Parameter optimization only available for IVF indexes")
                return {}
            
            logger.info("[OPTIMIZATION] Testing different nprobe values...")
            
            best_nprobe = self.nprobe
            best_score = 0
            results = {}
            
            # Test different nprobe values
            nprobe_values = [1, 5, 10, 20, 50, min(100, self.n_clusters)]
            
            for nprobe in nprobe_values:
                self.index.nprobe = nprobe
                
                times = []
                for query in query_vectors[:10]:  # Test with subset
                    start = time.time()
                    self.search_similar_patterns(query, k=5)
                    times.append((time.time() - start) * 1000)
                
                avg_time = np.mean(times)
                
                # Simple scoring: balance speed and thoroughness
                score = target_recall / (1 + avg_time / 10)  # Prefer faster searches
                
                results[nprobe] = {
                    "avg_search_time": avg_time,
                    "score": score
                }
                
                if score > best_score:
                    best_score = score
                    best_nprobe = nprobe
                
                logger.info(f"nprobe={nprobe}: {avg_time:.2f}ms avg, score={score:.3f}")
            
            # Set optimal value
            self.index.nprobe = best_nprobe
            self.nprobe = best_nprobe
            
            logger.info(f"[OPTIMIZATION] Optimal nprobe: {best_nprobe}")
            
            return {
                "optimal_nprobe": best_nprobe,
                "results": results,
                "best_score": best_score
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Parameter optimization failed: {e}")
            return {}
    
    def save_index(self, filepath: str) -> bool:
        """
        Save index and metadata to disk
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata and preprocessing components
            metadata = {
                "pattern_metadata": self.pattern_metadata,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "n_clusters": self.n_clusters,
                "nprobe": self.nprobe,
                "normalize_features": self.normalize_features,
                "use_pca": self.use_pca,
                "pca_components": self.pca_components,
                "index_size": self.index_size,
                "is_trained": self.is_trained
            }
            
            # Save sklearn components if available
            if self.scaler:
                with open(f"{filepath}_scaler.pkl", 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            if self.pca:
                with open(f"{filepath}_pca.pkl", 'wb') as f:
                    pickle.dump(self.pca, f)
            
            # Save metadata
            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"[SUCCESS] Index saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to save index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """
        Load index and metadata from disk
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.pattern_metadata = metadata["pattern_metadata"]
            self.dimension = metadata["dimension"]
            self.index_type = metadata["index_type"]
            self.n_clusters = metadata["n_clusters"]
            self.nprobe = metadata["nprobe"]
            self.normalize_features = metadata["normalize_features"]
            self.use_pca = metadata["use_pca"]
            self.pca_components = metadata["pca_components"]
            self.index_size = metadata["index_size"]
            self.is_trained = metadata["is_trained"]
            
            # Load sklearn components
            try:
                with open(f"{filepath}_scaler.pkl", 'rb') as f:
                    self.scaler = pickle.load(f)
            except FileNotFoundError:
                pass
            
            try:
                with open(f"{filepath}_pca.pkl", 'rb') as f:
                    self.pca = pickle.load(f)
            except FileNotFoundError:
                pass
            
            # Set nprobe for IVF indexes
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
            
            logger.info(f"[SUCCESS] Index loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load index: {e}")
            return False
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics
        """
        if not self.search_times:
            return {}
        
        return {
            "total_searches": len(self.search_times),
            "avg_search_time": np.mean(self.search_times),
            "min_search_time": np.min(self.search_times),
            "max_search_time": np.max(self.search_times),
            "median_search_time": np.median(self.search_times),
            "index_size": self.index_size,
            "index_type": self.index_type,
            "searches_per_second": 1000 / np.mean(self.search_times) if self.search_times else 0
        }

def create_optimal_index_for_trading(expected_patterns: int = 1000,
                                   feature_dimension: int = 32,
                                   speed_priority: bool = True) -> OptimizedFAISSPatternMatcher:
    """
    Factory function to create optimally configured FAISS index for trading patterns
    """
    
    if expected_patterns  64,  # Use PCA for high-dimensional data
        pca_components=min(32, feature_dimension // 2)
    )
    
    return matcher

# Example usage and testing
def test_optimized_faiss():
    """Test the optimized FAISS setup"""
    logger.info("🚀 Testing Optimized FAISS Setup")
    
    # Create matcher with different configurations
    configs = [
        ("small_exact", 100, "flat"),
        ("medium_ivf", 5000, "ivf"), 
        ("large_hnsw", 50000, "hnsw")
    ]
    
    for name, size, expected_type in configs:
        logger.info(f"\n--- Testing {name} configuration ---")
        
        matcher = create_optimal_index_for_trading(
            expected_patterns=size,
            feature_dimension=32
        )
        
        if matcher.create_index(expected_size=size):
            logger.info(f"✅ {name}: Index created successfully")
            
            # Generate test data
            training_data = [np.random.randn(32) for _ in range(min(100, size // 10))]
            
            if matcher.train_index(training_data):
                logger.info(f"✅ {name}: Training completed")
                
                # Add patterns
                for i in range(min(10, size // 100)):
                    vector = np.random.randn(32)
                    metadata = {"symbol": f"TEST{i}", "pattern": name}
                    matcher.add_pattern(vector, metadata)
                
                # Test search
                query = np.random.randn(32)
                results = matcher.search_similar_patterns(query, k=5)
                
                if results["distances"]:
                    logger.info(f"✅ {name}: Search completed in {results['search_time']:.2f}ms")
                else:
                    logger.warning(f"⚠️ {name}: No search results")
            
            # Performance stats
            stats = matcher.get_performance_stats()
            if stats:
                logger.info(f"📊 {name}: {stats['searches_per_second']:.0f} searches/sec")
        
        else:
            logger.error(f"❌ {name}: Index creation failed")

if __name__ == "__main__":
    test_optimized_faiss()
