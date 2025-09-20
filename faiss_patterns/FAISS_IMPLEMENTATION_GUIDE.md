# FAISS Implementation Guide (PostgreSQL + Scanner Integration)

This document explains how to use the FAISS module in this repository with Railway PostgreSQL for persistence, and how to wire it to your scanner output to produce trading signals. All logs and examples use ASCII-only formatting.

## Scope

- Persistence: PostgreSQL only (Railway), no SQLite fallbacks
- Data sources: Live/scanner data, not simulated
- Logging: ASCII-only; use tags like [ERROR], [SUCCESS]
- Security: Use environment variables for secrets

## Prerequisites

- Environment variable DATABASE_URL set to your Railway PostgreSQL URL with flags:
  - sslmode=require
  - gssencmode=disable
  - Example format (mask your real credentials in public docs):
    postgresql://USER:PASS@HOST:PORT/DB?sslmode=require&gssencmode=disable

## Persistence Overview

Module: `src/faiss_engine/railway_database.py` (class `FAISSRailwayDatabase`). Tables created:

- `faiss_trading_patterns`
  - pattern_id (unique), symbol, pattern_type, embedding_vector (BYTEA), metadata (JSONB), confidence_score (FLOAT), timestamps
- `faiss_market_regimes`
  - regime_id (unique), regime_type, characteristics (JSONB), start_time, end_time, confidence
- `faiss_performance_metrics`
  - metric_id (unique), pattern_id, metric_type, metric_value, metadata (JSONB), recorded_at

Data is inserted via parameterized SQL with ON CONFLICT upserts.

## Data Contract

Inputs to persist a pattern:

- symbol: string, e.g., "AAPL"
- pattern_type: string, e.g., "scanner_profile" or "intraday_snapshot"
- embedding_vector: numpy ndarray, dtype float32, shape [D] where D is fixed project-wide
- metadata: dict with scanner fields and context (must be JSON serializable)
- confidence_score: float in [0, 1]

Outputs from a search:

- signals: { symbol, action, confidence, matches }
  - matches: array of { pattern_id, similarity, symbol, metadata }

## Building Embeddings From Scanner Data

Source table: `scanner_results` (or your scanner JSON). For each row or snapshot:

1. Select numeric features (returns, range %, volatility, volume ratios, RSI, MACD hist, ATR %, liquidity flags, etc.)
2. Normalize consistently (e.g., z-score or min-max). Persist the scaler used so online queries use the same transform.
3. Build `embedding_vector` as np.float32 of length D (constant across all entries)
4. Construct `metadata` dict with the important raw fields for later inspection
5. Compute or assign `confidence_score` (optional but recommended)

Persist using `FAISSRailwayDatabase.store_trading_pattern(...)` with a deterministic `pattern_id` (hash of symbol+timestamp+selected fields) to allow idempotent upserts.

## Indexing and Search With FAISS

At service startup:

1. Load rows with `FAISSRailwayDatabase.get_trading_patterns(symbol=..., pattern_type=...)`
2. Stack all `embedding_vector` buffers into a matrix X with shape [N, D], dtype float32
3. Build FAISS index (choose one and stick to it):
   - Cosine-similar behavior: normalize vectors and use IndexFlatIP
   - L2 distance: use IndexFlatL2
4. Keep a side-car mapping of row index to {pattern_id, symbol, metadata}

For a live scanner snapshot:

1. Build query vector q (1, D), float32 using the same feature builder and scaler
2. `D, I = index.search(q, k)` to get top-K neighbors
3. Convert distances or inner products to similarity/confidence
   - IP: similarity = max(0, score)
   - L2: similarity = 1.0 / (1.0 + distance)
4. Retrieve metadata via mapping and aggregate into a decision

## Signals Format

Minimal signal payload returned downstream (JSON example):

```json
{
  "symbol": "AAPL",
  "action": "buy",
  "confidence": 0.78,
  "matches": [
    {"pattern_id": "...", "similarity": 0.83, "symbol": "AAPL", "metadata": {"scanner_rank": 1, "rsi": 62, "atr_pct": 0.018}}
  ]
}
```

Actions can be derived by a simple policy: e.g., if average similarity of top-3 is above a threshold and match metadata aligns with your strategy filters, emit "buy"; otherwise, "hold"; symmetrical logic for "sell" if you store bearish patterns.

## Quick Start Checks

- Connectivity and schema overview: `check_railway_database_tables.py`
- FAISS persistence smoke test: run `python src\faiss_engine\railway_database.py` (creates tables and inserts sample records)
- FAISS tables existence: `python check_faiss_tables.py`

All use DATABASE_URL from environment and add sslmode=require & gssencmode=disable if missing.

## Error Handling and Logging

- ASCII-only logs, use tags like [ERROR], [WARNING], [SUCCESS]
- Validate vector dtype and dimension before insert; reject mismatches
- Normalize DATABASE_URL to include required SSL and GSS flags
- Catch database timeouts and surface concise messages

## Edge Cases

- Empty corpus: build index lazily when N > 0
- Dimension drift: disallow; guard with a stored config value for D
- Large batch load: consider chunked loads when N is large
- Network instability: enable connection pool pre-ping and keepalives (already configured)

## Pipeline Integration

Two common patterns:

1. JSON handoff
   - Service writes signals to `signals.json` for the next pipeline stage
2. Database handoff
   - Service writes to a `faiss_signals` table for downstream components to consume

Prefer database handoff for a centralized source of truth and easier monitoring.

## Minimal Pseudocode (Storage and Search)

Storage

```python
db = FAISSRailwayDatabase()
for row in scanner_rows:
    vec = build_vector(row)  # np.float32, shape [D]
    meta = select_metadata(row)
    pid = hash_id(row)
    db.store_trading_pattern(pid, row.symbol, "scanner_profile", vec, meta, confidence)
```

Index and Query

```python
patterns = db.get_trading_patterns(symbol=None, pattern_type="scanner_profile")
X, meta_map = stack_vectors(patterns)
index = build_faiss_index(X)  # IP or L2
q = build_vector(live_snapshot)
D, I = index.search(q[None, :], k=10)
signals = aggregate_to_signal(I, D, meta_map)
```

## Security and Configuration

- Never hardcode credentials in code
- Use `.env` for local development; do not commit real secrets
- Ensure DATABASE_URL contains `sslmode=require&gssencmode=disable`

## Next Steps

- Add a small service that pulls recent `scanner_results`, builds embeddings, persists patterns, builds an index, and emits signals to the trade plan generator
- Add tests that validate vector dimension checks and basic nearest neighbor semantics
