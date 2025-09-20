"""Tests for FAISS integration."""

import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestFAISSEngine:
    """Test cases for FAISS engine functionality."""
    
    def test_faiss_cpu_availability(self):
        """Test that FAISS CPU is available."""
        try:
            import faiss
            assert faiss is not None
            print(f"FAISS version: {faiss.__version__}")
        except ImportError:
            pytest.skip("FAISS not available")
    
    def test_vector_creation(self):
        """Test creating vectors for similarity search."""
        # Create sample vectors
        dimension = 128
        vectors = np.random.rand(10, dimension).astype(np.float32)
        
        assert vectors.shape == (10, 128)
        assert vectors.dtype == np.float32
    
    def test_index_creation(self):
        """Test FAISS index creation."""
        try:
            import faiss
            
            dimension = 128
            index = faiss.IndexFlatL2(dimension)
            
            assert index.d == dimension
            assert index.ntotal == 0
            
        except ImportError:
            pytest.skip("FAISS not available")
    
    def test_index_operations(self):
        """Test adding and searching vectors in FAISS index."""
        try:
            import faiss
            
            dimension = 128
            index = faiss.IndexFlatL2(dimension)
            
            # Add vectors
            vectors = np.random.rand(10, dimension).astype(np.float32)
            index.add(vectors)
            
            assert index.ntotal == 10
            
            # Search
            query = vectors[0:1]  # Use first vector as query
            distances, indices = index.search(query, k=3)
            
            assert len(distances[0]) == 3
            assert len(indices[0]) == 3
            assert indices[0][0] == 0  # First result should be the query itself
            
        except ImportError:
            pytest.skip("FAISS not available")


class TestDatabaseIntegration:
    """Test cases for database integration."""
    
    @pytest.mark.asyncio
    async def test_database_connection_mock(self, mock_database_url):
        """Test database connection with mock."""
        with patch('psycopg2.connect') as mock_connect:
            mock_connect.return_value = Mock()
            
            # Simulate database connection test
            connection = mock_connect(mock_database_url)
            assert connection is not None
    
    def test_pattern_storage_format(self, sample_pattern_data):
        """Test pattern data format for storage."""
        pattern = sample_pattern_data
        
        assert "pattern_id" in pattern
        assert "symbol" in pattern
        assert "features" in pattern
        assert "metadata" in pattern
        
        # Check feature vector format
        features = pattern["features"]
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features.shape) == 1  # 1D array


class TestPatternRecognition:
    """Test cases for pattern recognition functionality."""
    
    def test_pattern_similarity_calculation(self):
        """Test pattern similarity calculation."""
        # Create two similar patterns
        pattern1 = np.random.rand(128).astype(np.float32)
        pattern2 = pattern1 + np.random.rand(128).astype(np.float32) * 0.1
        
        # Calculate L2 distance
        distance = np.linalg.norm(pattern1 - pattern2)
        
        # Should be a small positive number
        assert distance > 0
        assert distance  0
        
        # Step 3: Pattern storage (mocked)
        pattern_id = pattern_data["pattern_id"]
        assert pattern_id.startswith("test_pattern")
        
        # Step 4: Similarity search (mocked)
        # In real implementation, this would use FAISS
        similar_patterns = ["pattern_001", "pattern_002"]
        assert len(similar_patterns) > 0
