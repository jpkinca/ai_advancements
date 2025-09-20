"""mvp_pipeline.py
Minimal end-to-end pipeline (Phase 1 MVP):

Steps:
1. Load watchlist symbols.
2. Fetch historical bars (IBKR preferred) for a configurable date range.
3. Persist bars into PostgreSQL (`tt_prices`).
4. (Placeholder) Build basic feature frame (returns & rolling volatility).
5. (Placeholder) Show how environment would be instantiated (without full training loop).

Run:
    python -m mvp_pipeline --start 2024-01-01 --end 2024-06-30 --interval 1d

Environment variables:
    DATABASE_URL must be set (or .env loaded externally).

Note: Actual RL training left out for brevity; focus is ingestion + persistence + feature scaffolding.
"""
from __future__ import annotations

import argparse
import pandas as pd
from dotenv import load_dotenv
import nest_asyncio

# Apply nest_asyncio to allow running asyncio loops in scripts
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Handle both direct execution and module execution
try:
    from db_utils import get_engine, ensure_tables, ensure_indexes, upsert_price_bars
    from watchlist_loader import load_watchlist, fetch_price_history
except ImportError:
    from .db_utils import get_engine, ensure_tables, ensure_indexes, upsert_price_bars
    from .watchlist_loader import load_watchlist, fetch_price_history


def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple derived features per symbol (daily return, rolling volatility)."""
    if df.empty:
        return df
    out = []
    for sym, grp in df.groupby("symbol"):
        g = grp.sort_values("datetime").copy()
        g["return_1"] = g["close"].pct_change()
        g["vol_10"] = g["return_1"].rolling(10).std()
        g["vol_20"] = g["return_1"].rolling(20).std()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TensorTrade MVP data pipeline")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=False, help="End date YYYY-MM-DD (default today)")
    p.add_argument("--interval", default="1d", help="Bar interval (1d,1h,15m,1m)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of symbols (debug)")
    p.add_argument("--rate-limit", type=float, default=0.15, help="Delay between IBKR requests (seconds)")
    p.add_argument("--batch-size", type=int, default=10, help="Symbols per batch before connection check")
    p.add_argument("--retry-attempts", type=int, default=3, help="Retry attempts for failed symbols")
    p.add_argument("--db-url", type=str, default=None, help="Override DATABASE_URL env var (PostgreSQL URI)")
    return p.parse_args()


def main():
    args = parse_args()
    start = args.start
    end = args.end
    interval = args.interval

    symbols = load_watchlist()
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"Loaded {len(symbols)} symbols from watchlist")

    print(f"Fetching historical data from IBKR {start}->{end or 'today'} interval={interval} ...")
    print(f"Rate limiting: {args.rate_limit}s delay, batch size: {args.batch_size}, retries: {args.retry_attempts}")
    bars = fetch_price_history(
        symbols, 
        start=start, 
        end=end, 
        interval=interval, 
        source="ibkr",
        rate_limit_delay=args.rate_limit,
        batch_size=args.batch_size,
        retry_attempts=args.retry_attempts
    )
    if hasattr(bars, "__await__"):
        raise RuntimeError("fetch_price_history returned coroutine/future; run script outside async loop.")
    print(f"Fetched {len(bars)} bar rows")

    engine = get_engine(args.db_url)
    ensure_tables(engine)
    ensure_indexes(engine)
    inserted = upsert_price_bars(engine, bars)
    print(f"Inserted (attempted) {inserted} rows (duplicates skipped silently)")

    feat = compute_basic_features(bars)
    latest = feat.sort_values("datetime").groupby("symbol").tail(1)[["symbol", "close", "return_1", "vol_10", "vol_20"]]
    print("Latest feature snapshot:")
    print(latest.to_string(index=False))

    print("\nEnvironment assembly placeholder (training not executed in MVP script)")
    print(" - Next steps: build Streams from features, instantiate Portfolio & Action/Reward schemes, run PPO training.")


if __name__ == "__main__":
    main()
