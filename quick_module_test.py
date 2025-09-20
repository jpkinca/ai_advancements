#!/usr/bin/env python3
"""
Quick FAISS and IBKR Module Test
"""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_faiss():
    """Test FAISS module"""
    try:
        import faiss
        import numpy as np
        
        logger.info("[SUCCESS] FAISS imported successfully")
        
        # Create a simple test
        dimension = 32
        n_vectors = 10
        
        # Create random vectors
        vectors = np.random.random((n_vectors, dimension)).astype('float32')
        
        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        # Test search
        query = np.random.random((1, dimension)).astype('float32')
        distances, indices = index.search(query, 3)
        
        logger.info(f"[SUCCESS] FAISS test completed - found {len(indices[0])} neighbors")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] FAISS test failed: {e}")
        return False

def test_ibkr():
    """Test IBKR module"""
    try:
        from ib_insync import IB, Stock
        logger.info("[SUCCESS] ib_insync imported successfully")
        
        # Just test import and basic object creation
        ib = IB()
        contract = Stock('AAPL', 'SMART', 'USD')
        
        logger.info("[SUCCESS] IBKR objects created successfully")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] IBKR test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("QUICK MODULE TEST")
    print("=" * 50)
    
    faiss_ok = test_faiss()
    ibkr_ok = test_ibkr()
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print(f"FAISS: {'PASS' if faiss_ok else 'FAIL'}")
    print(f"IBKR:  {'PASS' if ibkr_ok else 'FAIL'}")
    print("=" * 50)
    
    if faiss_ok and ibkr_ok:
        print("✓ All modules are working - ready for integration test")
        return 0
    else:
        print("✗ Some modules failed - check installation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
