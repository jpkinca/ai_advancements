#!/usr/bin/env python3
"""
FAISS Pattern Search Test

This script tests the core FAISS functionality with generated patterns.
It demonstrates the complete workflow from pattern storage to similarity search.

Author: AI Assistant  
Date: September 2, 2025
Status: Phase 1 Testing
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import faiss
    logger.info("[SUCCESS] FAISS imported successfully")
except ImportError:
    logger.error("[ERROR] FAISS not available - install with: pip install faiss-cpu")
    sys.exit(1)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class FAISSPatternSearcher:
    """FAISS-based pattern similarity search engine"""
    
    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.index = None
        self.pattern_metadata = []
        self.vectors = []
        
    def create_index(self, metric: str = "cosine"):
        """Create FAISS index"""
        try:
            if metric == "cosine":
                # Use Inner Product for cosine similarity (vectors must be normalized)
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"[SUCCESS] Created cosine similarity index (dimension: {self.dimension})")
            elif metric == "l2":
                # Use L2 distance
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"[SUCCESS] Created L2 distance index (dimension: {self.dimension})")
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to create FAISS index: {e}")
            raise
    
    def add_patterns(self, vectors: np.ndarray, metadata: List[Dict]):
        """Add pattern vectors to the index"""
        try:
            if self.index is None:
                self.create_index()
            
            # Ensure vectors are float32 and properly shaped
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            
            vectors = vectors.astype(np.float32)
            
            # Normalize vectors for cosine similarity
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / (norms + 1e-8)  # Avoid division by zero
            
            # Add to index
            self.index.add(vectors)
            
            # Store vectors and metadata
            self.vectors.extend(vectors.tolist())
            self.pattern_metadata.extend(metadata)
            
            logger.info(f"[SUCCESS] Added {len(vectors)} patterns to index")
            logger.info(f"[INFO] Total patterns in index: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to add patterns to index: {e}")
            raise
    
    def search_similar_patterns(self, query_vector: np.ndarray, k: int = 5) -> Dict:
        """Search for similar patterns"""
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("[WARNING] Index is empty")
                return {"distances": [], "indices": [], "metadata": []}
            
            # Prepare query vector
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            query_vector = query_vector.astype(np.float32)
            
            # Normalize query vector
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
            
            # Search
            k = min(k, self.index.ntotal)  # Don't search for more than available
            distances, indices = self.index.search(query_vector, k)
            
            # Get metadata for results
            result_metadata = []
            for idx in indices[0]:
                if 0  Tuple[np.ndarray, List[Dict]]:
    """Create synthetic test patterns for demonstration"""
    logger.info(f"[CREATING] {n_patterns} test patterns with dimension {dimension}")
    
    np.random.seed(42)  # For reproducible results
    
    # Create pattern vectors with different "types"
    patterns = []
    metadata = []
    
    pattern_types = ["bull_flag", "cup_with_handle", "flat_base", "breakout", "reversal"]
    
    for i in range(n_patterns):
        # Create base pattern with some structure
        pattern_type = pattern_types[i % len(pattern_types)]
        
        if pattern_type == "bull_flag":
            # Bull flag patterns - strong momentum followed by consolidation
            base_vector = np.concatenate([
                np.random.normal(0.8, 0.1, dimension//4),  # High momentum indicators
                np.random.normal(0.2, 0.1, dimension//4),  # Low volatility in flag
                np.random.normal(0.6, 0.1, dimension//4),  # Volume characteristics
                np.random.normal(0.4, 0.2, dimension//4)   # Other technical indicators
            ])
        elif pattern_type == "cup_with_handle":
            # Cup with handle - U-shaped recovery
            base_vector = np.concatenate([
                np.random.normal(0.3, 0.1, dimension//4),  # Initial decline
                np.random.normal(0.7, 0.1, dimension//4),  # Recovery strength
                np.random.normal(0.5, 0.1, dimension//4),  # Handle formation
                np.random.normal(0.6, 0.1, dimension//4)   # Volume confirmation
            ])
        elif pattern_type == "breakout":
            # Breakout patterns - volume surge with price acceleration
            base_vector = np.concatenate([
                np.random.normal(0.9, 0.1, dimension//4),  # Price acceleration
                np.random.normal(0.8, 0.1, dimension//4),  # Volume surge
                np.random.normal(0.4, 0.1, dimension//4),  # Prior consolidation
                np.random.normal(0.7, 0.1, dimension//4)   # Momentum indicators
            ])
        else:
            # Random pattern for variety
            base_vector = np.random.normal(0.5, 0.3, dimension)
        
        # Add some noise
        noise = np.random.normal(0, 0.1, dimension)
        pattern_vector = base_vector + noise
        
        # Clip to reasonable range
        pattern_vector = np.clip(pattern_vector, -2.0, 2.0)
        
        patterns.append(pattern_vector)
        
        # Create metadata
        metadata.append({
            "pattern_id": f"test_pattern_{i:03d}",
            "pattern_type": pattern_type,
            "symbol": f"STOCK{i % 10}",
            "timestamp": datetime.now().isoformat(),
            "confidence": float(np.random.uniform(0.6, 0.95)),
            "timeframe": "daily",
            "entry_price": float(np.random.uniform(100, 300)),
            "synthetic": True
        })
    
    pattern_array = np.array(patterns, dtype=np.float32)
    logger.info(f"[SUCCESS] Created test patterns with shape: {pattern_array.shape}")
    
    return pattern_array, metadata

def test_pattern_similarity():
    """Test pattern similarity search functionality"""
    logger.info("\n" + "="*50)
    logger.info("TESTING FAISS PATTERN SIMILARITY SEARCH")
    logger.info("="*50)
    
    # Create test patterns
    dimension = 64
    patterns, metadata = create_test_patterns(n_patterns=20, dimension=dimension)
    
    # Initialize FAISS searcher
    searcher = FAISSPatternSearcher(dimension=dimension)
    
    # Add patterns to index
    searcher.add_patterns(patterns, metadata)
    
    # Test similarity search
    logger.info("\n[TESTING] Similarity search...")
    
    # Use the first pattern as query
    query_pattern = patterns[0]
    query_metadata = metadata[0]
    
    logger.info(f"Query pattern: {query_metadata['pattern_type']} for {query_metadata['symbol']}")
    
    # Search for similar patterns
    results = searcher.search_similar_patterns(query_pattern, k=5)
    
    # Display results
    logger.info(f"\nTop 5 similar patterns:")
    for i, (distance, idx, meta) in enumerate(zip(results['distances'], results['indices'], results['metadata'])):
        similarity = distance  # For cosine similarity (inner product)
        logger.info(f"  {i+1}. {meta['pattern_type']} ({meta['symbol']}) - Similarity: {similarity:.4f}")
        logger.info(f"      Confidence: {meta['confidence']:.3f}, Entry: ${meta['entry_price']:.2f}")
    
    # Test with different query
    logger.info(f"\n[TESTING] Search with different pattern type...")
    
    # Find a different pattern type
    breakout_patterns = [i for i, m in enumerate(metadata) if m['pattern_type'] == 'breakout']
    if breakout_patterns:
        breakout_idx = breakout_patterns[0]
        breakout_query = patterns[breakout_idx]
        breakout_metadata = metadata[breakout_idx]
        
        logger.info(f"Query pattern: {breakout_metadata['pattern_type']} for {breakout_metadata['symbol']}")
        
        results2 = searcher.search_similar_patterns(breakout_query, k=5)
        
        logger.info(f"Top 5 similar patterns:")
        for i, (distance, idx, meta) in enumerate(zip(results2['distances'], results2['indices'], results2['metadata'])):
            similarity = distance
            logger.info(f"  {i+1}. {meta['pattern_type']} ({meta['symbol']}) - Similarity: {similarity:.4f}")
    
    # Test index persistence
    logger.info(f"\n[TESTING] Index save/load functionality...")
    
    # Save index
    test_index_path = "test_faiss_index"
    searcher.save_index(test_index_path)
    
    # Create new searcher and load index
    new_searcher = FAISSPatternSearcher(dimension=dimension)
    new_searcher.load_index(test_index_path)
    
    # Test search with loaded index
    results3 = new_searcher.search_similar_patterns(query_pattern, k=3)
    
    logger.info(f"Search results after reload (should be identical):")
    for i, (distance, idx, meta) in enumerate(zip(results3['distances'], results3['indices'], results3['metadata'])):
        similarity = distance
        logger.info(f"  {i+1}. {meta['pattern_type']} ({meta['symbol']}) - Similarity: {similarity:.4f}")
    
    # Performance test
    logger.info(f"\n[TESTING] Performance test...")
    
    import time
    start_time = time.time()
    
    # Run multiple searches
    for _ in range(100):
        random_query = patterns[np.random.randint(0, len(patterns))]
        searcher.search_similar_patterns(random_query, k=5)
    
    end_time = time.time()
    avg_search_time = (end_time - start_time) / 100 * 1000  # Convert to milliseconds
    
    logger.info(f"Average search time: {avg_search_time:.2f} ms")
    logger.info(f"Target: = 1.5:
        print("\n[SUCCESS] FAISS pattern search is working!")
        print("Next steps:")
        print("1. Run pattern_generation_runner.py to generate real patterns")
        print("2. Build production FAISS index manager")
        print("3. Implement real-time pattern detection")
    else:
        print("\n[WARNING] Issues detected - check error messages above")
        print("Verify FAISS installation and dependencies")

if __name__ == "__main__":
    main()
