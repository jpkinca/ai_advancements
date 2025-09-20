"""Test configuration for AI Trading Advancements."""

import os
import pytest
import asyncio
from pathlib import Path

# Set test environment variables
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"

# Test database URL (use SQLite for testing if PostgreSQL not available)
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "sqlite:///test.db"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_data_dir():
    """Return the test data directory path."""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "symbol": "AAPL",
        "timestamp": "2025-09-01T10:00:00Z",
        "open": 150.0,
        "high": 152.0,
        "low": 149.0,
        "close": 151.0,
        "volume": 1000000
    }

@pytest.fixture
def sample_pattern_data():
    """Sample pattern data for FAISS testing."""
    import numpy as np
    return {
        "pattern_id": "test_pattern_001",
        "symbol": "AAPL",
        "features": np.random.rand(128).astype(np.float32),
        "metadata": {
            "timeframe": "1D",
            "pattern_type": "bullish_flag",
            "confidence": 0.85
        }
    }

@pytest.fixture
def mock_database_url():
    """Mock database URL for testing."""
    return "postgresql://test:test@localhost:5432/test_db"
