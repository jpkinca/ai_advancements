import os
import sys
import pathlib
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

# Ensure src directory is importable as top-level modules
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Use an in-memory SQLite DB for fast tests (override DATABASE_URL usage indirectly)
@pytest.fixture(scope="session")
def engine():
    """Provide a PostgreSQL engine from DATABASE_URL.

    Skips tests if DATABASE_URL is unset or not pointing to PostgreSQL.
    Does NOT print the URL (secrets safety).
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set; PostgreSQL required (SQLite prohibited).")
    if not url.startswith("postgres"):
        pytest.skip("DATABASE_URL must be PostgreSQL (starts with 'postgres').")
    try:
        eng = create_engine(url, future=True)
        # Smoke test connection
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PostgreSQL connection failed: {exc}")
    yield eng

@pytest.fixture
def sample_prices_df():
    data = {
        "symbol": ["AAA", "AAA", "BBB", "BBB"],
        "datetime": ["2024-01-01 00:00:00", "2024-01-02 00:00:00", "2024-01-01 00:00:00", "2024-01-02 00:00:00"],
        "open": [10, 11, 20, 21],
        "high": [11, 12, 21, 22],
        "low": [9, 10, 19, 20],
        "close": [10.5, 11.5, 20.5, 21.5],
        "volume": [1000, 1100, 2000, 2100],
    }
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

@pytest.fixture
def cleanup_prices(engine):
    """Cleanup inserted test instruments after each test for isolation."""
    test_syms = ["AAA", "BBB"]
    yield
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM tt_prices WHERE instrument = ANY(:s)"), {"s": test_syms})
    except Exception:
        pass
