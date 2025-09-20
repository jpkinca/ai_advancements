"""db_utils.py
Lightweight DB utilities for MVP pipeline.

Uses SQLAlchemy for PostgreSQL persistence of historical price data and episode artifacts.
Assumes schema created by `create_tt_tables.py`.
"""
from __future__ import annotations

import os
import logging
from typing import Iterable, List, Dict, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def get_engine(url: str | None = None) -> Engine:
    url = url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not set; export or place in .env")
    # Explicitly set client encoding to UTF-8 to prevent decode errors
    return create_engine(url, future=True, connect_args={'client_encoding': 'utf8'})


def ensure_indexes(engine: Engine) -> None:
    with engine.begin() as conn:
        # Unique index prevents duplicate bar inserts (instrument+timestamp)
        conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_tt_prices_instrument_timestamp ON tt_prices (instrument, timestamp)"
        ))
        # Helpful secondary indexes (optional for later scaling)
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_tt_action_episode_time ON tt_action (episode_id, timestamp)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_tt_reward_episode_time ON tt_reward (episode_id, timestamp)"
        ))


def ensure_tables(engine: Engine) -> None:
    """Create core MVP tables if they do not exist.

    Keeps schema minimal; for full schema run create_tt_tables.py.
    """
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS tt_prices (
                id SERIAL PRIMARY KEY,
                instrument TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION
            )
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS tt_episode (
                id SERIAL PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                stop_reason TEXT,
                max_drawdown DOUBLE PRECISION,
                sharpe_ratio DOUBLE PRECISION,
                turnover DOUBLE PRECISION,
                config_json TEXT
            )
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS tt_action (
                id SERIAL PRIMARY KEY,
                episode_id INTEGER REFERENCES tt_episode(id),
                timestamp TIMESTAMP,
                instrument TEXT,
                action_value DOUBLE PRECISION
            )
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS tt_reward (
                id SERIAL PRIMARY KEY,
                episode_id INTEGER REFERENCES tt_episode(id),
                timestamp TIMESTAMP,
                reward_value DOUBLE PRECISION
            )
            """
        ))
        # New: feature persistence (long form)
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS tt_features (
                id SERIAL PRIMARY KEY,
                instrument TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                feature TEXT NOT NULL,
                value DOUBLE PRECISION,
                UNIQUE (instrument, timestamp, feature)
            )
            """
        ))
        # New: signal scores
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS tt_signal_scores (
                id SERIAL PRIMARY KEY,
                instrument TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                signal_name TEXT NOT NULL,
                score DOUBLE PRECISION,
                meta_json TEXT,
                UNIQUE (instrument, timestamp, signal_name)
            )
            """
        ))
        # New: target weights (include episode for context)
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS tt_target_weights (
                id SERIAL PRIMARY KEY,
                episode_id INTEGER REFERENCES tt_episode(id),
                timestamp TIMESTAMP NOT NULL,
                instrument TEXT NOT NULL,
                target_weight DOUBLE PRECISION,
                strategy TEXT,
                rationale TEXT
            )
            """
        ))
        # New: equity curve per episode
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS tt_equity_curve (
                id SERIAL PRIMARY KEY,
                episode_id INTEGER REFERENCES tt_episode(id),
                timestamp TIMESTAMP NOT NULL,
                net_worth DOUBLE PRECISION
            )
            """
        ))

        # Backfill: add config_json column if missing (legacy deployments)
        try:
            conn.execute(text("ALTER TABLE tt_episode ADD COLUMN IF NOT EXISTS config_json TEXT"))
        except Exception:
            pass


def upsert_price_bars(engine: Engine, df: pd.DataFrame, chunk: int = 1000) -> int:
    """Insert historical bars into tt_prices with ON CONFLICT DO NOTHING semantics.

    DataFrame expected columns: instrument, timestamp, open, high, low, close, volume
    (timestamp must be tz-naive or UTC normalized).
    """
    if df.empty:
        return 0
    required = {"symbol", "datetime", "open", "high", "low", "close", "volume"}
    if not required.issubset({c.lower() for c in df.columns}):
        raise ValueError(f"Missing required columns. Have {df.columns.tolist()}")

    # Normalize / rename
    bars = df.copy()
    bars.rename(columns={"symbol": "instrument", "datetime": "timestamp"}, inplace=True)
    bars = bars[["instrument", "timestamp", "open", "high", "low", "close", "volume"]]
    # Ensure timestamps are Python datetime objects (not pandas Timestamp) for SQLite compatibility
    try:
        import pandas as _pd  # local import to avoid global dependency noise
        if _pd.api.types.is_datetime64_any_dtype(bars['timestamp']):
            bars['timestamp'] = _pd.to_datetime(bars['timestamp']).dt.to_pydatetime()
    except Exception:  # pragma: no cover
        pass

    inserted = 0
    with engine.begin() as conn:
        # Convert DataFrame to a list of dictionaries for executemany
        rows = bars.to_dict("records")
        
        # The core INSERT statement
        stmt = text(
            """
            INSERT INTO tt_prices (instrument, timestamp, open, high, low, close, volume)
            VALUES (:instrument, :timestamp, :open, :high, :low, :close, :volume)
            ON CONFLICT (instrument, timestamp) DO NOTHING
            """
        )
        
        # Execute the insert in chunks to manage memory
        for i in range(0, len(rows), chunk):
            chunk_data = rows[i:i + chunk]
            try:
                conn.execute(stmt, chunk_data)
                inserted += len(chunk_data)
            except Exception as e:
                logging.error(f"Error inserting chunk: {e}")
                # Optionally, you could add fallback logic here to insert row-by-row
                # for the failing chunk to identify the problematic row.
                pass

    return inserted


def fetch_prices(engine: Engine, instruments: Iterable[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    clauses = ["instrument = ANY(:symbols)"]
    params: dict = {"symbols": list(instruments)}
    if start:
        clauses.append("timestamp >= :start")
        params["start"] = start
    if end:
        clauses.append("timestamp  int:
    """Insert a new episode row and return its id."""
    with engine.begin() as conn:
        res = conn.execute(text(
            "INSERT INTO tt_episode (start_time) VALUES (:st) RETURNING id"
        ), {"st": start_time})
        episode_id = res.scalar_one()
    return int(episode_id)


def finalize_episode(engine: Engine, episode_id: int, end_time, stop_reason: str | None = None, max_drawdown: float | None = None, sharpe_ratio: float | None = None, turnover: float | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            """UPDATE tt_episode SET end_time=:et, stop_reason=:sr, max_drawdown=:md, sharpe_ratio=:sh, turnover=:to WHERE id=:eid"""
        ), {"et": end_time, "sr": stop_reason, "md": max_drawdown, "sh": sharpe_ratio, "to": turnover, "eid": episode_id})


def insert_action(engine: Engine, episode_id: int, timestamp, instrument: str, action_value: float) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            """INSERT INTO tt_action (episode_id, timestamp, instrument, action_value) VALUES (:ep,:ts,:ins,:val)"""
        ), {"ep": episode_id, "ts": timestamp, "ins": instrument, "val": float(action_value)})


def insert_reward(engine: Engine, episode_id: int, timestamp, reward_value: float) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            """INSERT INTO tt_reward (episode_id, timestamp, reward_value) VALUES (:ep,:ts,:rv)"""
        ), {"ep": episode_id, "ts": timestamp, "rv": float(reward_value)})


# ---------- New Persistence Helpers ----------

def insert_equity_point(engine: Engine, episode_id: int, timestamp, net_worth: float) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            """INSERT INTO tt_equity_curve (episode_id, timestamp, net_worth) VALUES (:ep,:ts,:nw)"""
        ), {"ep": episode_id, "ts": timestamp, "nw": float(net_worth)})


def upsert_features(engine: Engine, features_df: pd.DataFrame, feature_cols: list[str]) -> int:
    """Persist selected feature columns in long form into tt_features.

    features_df must have columns: symbol, datetime, plus feature_cols.
    """
    if features_df.empty:
        return 0
    required = {"symbol", "datetime"}.union(feature_cols)
    if not required.issubset({c.lower() if c.lower()==c else c for c in features_df.columns}):  # simple check
        raise ValueError("Features DataFrame missing required columns")
    rows = []
    for _, row in features_df.iterrows():
        ts = row["datetime"]
        if hasattr(ts, 'to_pydatetime'):
            ts = ts.to_pydatetime()
        for f in feature_cols:
            if f in row and pd.notna(row[f]):
                rows.append({
                    "instrument": row["symbol"],
                    "timestamp": ts,
                    "feature": f,
                    "value": float(row[f])
                })
    inserted = 0
    with engine.begin() as conn:
        for r in rows:
            conn.execute(text(
                """INSERT INTO tt_features (instrument, timestamp, feature, value)
                VALUES (:instrument,:timestamp,:feature,:value)
                ON CONFLICT (instrument, timestamp, feature) DO UPDATE SET value=EXCLUDED.value"""
            ), r)
            inserted += 1
    return inserted


def upsert_signal_scores(engine: Engine, scores: list[dict]) -> int:
    """Insert or update signal scores.
    Expected keys per dict: instrument, timestamp, signal_name, score, meta_json (optional)
    """
    if not scores:
        return 0
    with engine.begin() as conn:
        for s in scores:
            ts = s.get("timestamp")
            if hasattr(ts, 'to_pydatetime'):
                s["timestamp"] = ts.to_pydatetime()
            conn.execute(text(
                """INSERT INTO tt_signal_scores (instrument, timestamp, signal_name, score, meta_json)
                VALUES (:instrument,:timestamp,:signal_name,:score,:meta_json)
                ON CONFLICT (instrument, timestamp, signal_name) DO UPDATE SET score=EXCLUDED.score, meta_json=EXCLUDED.meta_json"""
            ), s)
    return len(scores)


def insert_target_weights(engine: Engine, episode_id: int, timestamp, weights: dict[str, float], strategy: str, rationale: str | None = None) -> int:
    if not weights:
        return 0
    with engine.begin() as conn:
        for sym, w in weights.items():
            conn.execute(text(
                """INSERT INTO tt_target_weights (episode_id, timestamp, instrument, target_weight, strategy, rationale)
                VALUES (:ep,:ts,:ins,:tw,:strat,:rat)"""
            ), {"ep": episode_id, "ts": timestamp, "ins": sym, "tw": float(w), "strat": strategy, "rat": rationale})
    return len(weights)


def set_episode_config(engine: Engine, episode_id: int, config_json: str) -> None:
    with engine.begin() as conn:
        conn.execute(text("UPDATE tt_episode SET config_json=:cfg WHERE id=:eid"), {"cfg": config_json, "eid": episode_id})


# --------------------------------------------------------------------------------------
# Lightweight higher-level manager expected by new real-time / risk / paper trading code
# --------------------------------------------------------------------------------------

class DatabaseManager:
    """Convenience wrapper around the lower-level helper functions in this module.

    The recently added real-time streaming, risk management, paper trading, and
    signal bridge modules expect a *DatabaseManager* providing a small interface:

        - insert_price_data(list[dict])
        - get_recent_data(symbols, lookback_periods)
        - (optional) store_tick(tick)

    This class adapts to the existing minimalist MVP schema. If a DATABASE_URL
    environment variable is not present it will fall back to a local SQLite file
    so tests can still run without explicit configuration.
    """

    def __init__(self, url: str | None = None, echo: bool = False):
        self.logger = logging.getLogger(__name__)
        db_url = url or os.getenv("DATABASE_URL")
        self._using_default_sqlite = False
        if not db_url:
            # Fallback for local development / tests
            db_url = "sqlite:///tensortrade_dev.db"
            self._using_default_sqlite = True
            self.logger.warning(
                "DATABASE_URL not set – using local SQLite file tensortrade_dev.db (development mode)"
            )
        self.engine: Engine = create_engine(db_url, future=True, echo=echo)
        # Ensure tables (idempotent) so modules can assume availability
        try:
            ensure_tables(self.engine)
            ensure_indexes(self.engine)
        except Exception as e:  # pragma: no cover - defensive
            self.logger.error(f"Error ensuring tables/indexes: {e}")

    # -------- Price / Bar Persistence -------------------------------------------------
    def insert_price_data(self, bars: List[Dict]) -> int:
        """Insert a collection of bar dictionaries.

        Expected keys (additional keys ignored):
            symbol (or instrument), timestamp, open, high, low, close, volume
        """
        if not bars:
            return 0
        inserted = 0
        with self.engine.begin() as conn:
            for bar in bars:
                record = {
                    "instrument": bar.get("symbol") or bar.get("instrument"),
                    "timestamp": bar.get("timestamp"),
                    "open": bar.get("open"),
                    "high": bar.get("high"),
                    "low": bar.get("low"),
                    "close": bar.get("close"),
                    "volume": bar.get("volume"),
                }
                if not record["instrument"] or record["timestamp"] is None:
                    continue  # skip malformed
                try:
                    conn.execute(
                        text(
                            """
                            INSERT INTO tt_prices (instrument, timestamp, open, high, low, close, volume)
                            VALUES (:instrument, :timestamp, :open, :high, :low, :close, :volume)
                            ON CONFLICT (instrument, timestamp) DO NOTHING
                            """
                        ),
                        record,
                    )
                    inserted += 1
                except Exception as e:  # pragma: no cover
                    self.logger.debug(f"Failed to insert bar {record}: {e}")
        return inserted

    # Alias to align with possible legacy name used elsewhere
    upsert_price_data = insert_price_data

    # -------- Recent Data Retrieval ---------------------------------------------------
    def get_recent_data(self, symbols: List[str], lookback_periods: int = 100) -> pd.DataFrame:
        """Return the most recent *lookback_periods* bars per symbol as a single DataFrame.

        Output columns: symbol, datetime, open, high, low, close, volume
        Ordered by symbol then timestamp ascending.
        """
        if not symbols:
            return pd.DataFrame(columns=["symbol", "datetime", "open", "high", "low", "close", "volume"])

        # Query all bars for symbols limited by lookback per symbol using window function (Postgres)
        # For SQLite fallback we will emulate with a simple subquery.
        with self.engine.begin() as conn:
            dialect = self.engine.url.get_backend_name()
            if dialect == "postgresql":
                query = text(
                    """
                    SELECT instrument AS symbol, timestamp AS datetime, open, high, low, close, volume
                    FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY instrument ORDER BY timestamp DESC) AS rn
                        FROM tt_prices
                        WHERE instrument = ANY(:symbols)
                    ) t
                    WHERE rn  None:  # tick is MarketTick dataclass in real_time_streaming
        """Optionally persist raw tick data.

        Not implemented for MVP to avoid high-volume writes. Provided so callers can
        invoke without needing to guard for attribute existence.
        """
        # Intentionally a no-op; could write to a dedicated ticks table later.
        return None

    # -------- Generic helpers --------------------------------------------------------
    def get_engine(self) -> Engine:
        return self.engine

    # Backward compatibility convenience wrappers around module-level functions
    def create_episode(self, start_time) -> int:
        return create_episode(self.engine, start_time)

    def finalize_episode(self, episode_id: int, end_time, **metrics) -> None:
        finalize_episode(self.engine, episode_id, end_time, **metrics)

    def insert_action(self, episode_id: int, timestamp, instrument: str, action_value: float) -> None:
        insert_action(self.engine, episode_id, timestamp, instrument, action_value)

    def insert_reward(self, episode_id: int, timestamp, reward_value: float) -> None:
        insert_reward(self.engine, episode_id, timestamp, reward_value)

    def insert_equity_point(self, episode_id: int, timestamp, net_worth: float) -> None:
        insert_equity_point(self.engine, episode_id, timestamp, net_worth)

    def upsert_features(self, features_df: pd.DataFrame, feature_cols: list[str]) -> int:
        return upsert_features(self.engine, features_df, feature_cols)

    def upsert_signal_scores(self, scores: list[dict]) -> int:
        return upsert_signal_scores(self.engine, scores)

    def insert_target_weights(self, episode_id: int, timestamp, weights: dict[str, float], strategy: str, rationale: str | None = None) -> int:
        return insert_target_weights(self.engine, episode_id, timestamp, weights, strategy, rationale)

    def set_episode_config(self, episode_id: int, config_json: str) -> None:
        set_episode_config(self.engine, episode_id, config_json)


__all__ = [
    "get_engine",
    "ensure_indexes",
    "ensure_tables",
    "upsert_price_bars",
    "fetch_prices",
    "create_episode",
    "finalize_episode",
    "insert_action",
    "insert_reward",
    "insert_equity_point",
    "upsert_features",
    "upsert_signal_scores",
    "insert_target_weights",
    "set_episode_config",
    "DatabaseManager",
]

