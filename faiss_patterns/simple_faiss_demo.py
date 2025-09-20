#!/usr/bin/env python3
"""
Simple FAISS demo for trading patterns.
"""

import numpy as np
import pandas as pd
import logging

# Configure logging to show INFO messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import faiss
    logger.info("FAISS imported successfully")
except ImportError:
    logger.error("FAISS not available")
    exit(1)

def main():
    logger.info("Starting simple FAISS demo...")
    
    # Create some sample pattern data
    n_patterns = 10
    dimension = 20
    
    logger.info(f"Creating {n_patterns} random patterns with {dimension} features each")
    patterns = np.random.random((n_patterns, dimension)).astype(np.float32)
    
    # Normalize for cosine similarity
    patterns = patterns / (np.linalg.norm(patterns, axis=1, keepdims=True) + 1e-8)
    
    # Create FAISS index
    logger.info("Creating FAISS index...")
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Add patterns to index
    logger.info("Adding patterns to index...")
    index.add(patterns)
    
    # Search for similar patterns
    query = patterns[0:1]  # Use first pattern as query
    k = 3  # Find top 3 similar patterns
    
    logger.info(f"Searching for {k} similar patterns...")
    similarities, indices = index.search(query, k)
    
    logger.info("Search results:")
    for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
        logger.info(f"  Rank {i+1}: Pattern {idx} (similarity: {sim:.4f})")
    
    logger.info("Demo completed successfully!")

if __name__ == "__main__":
    main()
