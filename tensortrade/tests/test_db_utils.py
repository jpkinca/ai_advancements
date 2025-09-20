import pandas as pd
from sqlalchemy import text
import db_utils as du


def test_ensure_tables_and_indexes(engine):
    du.ensure_tables(engine)
    du.ensure_indexes(engine)
    expected = {"tt_prices", "tt_episode", "tt_action", "tt_reward", "tt_features", "tt_signal_scores", "tt_target_weights", "tt_equity_curve"}
    with engine.begin() as conn:
        res = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name = ANY(:names)"), {"names": list(expected)})
        found = {r[0] for r in res.fetchall()}
    missing = expected - found
    assert not missing, f"Missing tables: {missing}"


def test_upsert_and_fetch(engine, sample_prices_df, cleanup_prices):
    du.ensure_tables(engine)
    du.ensure_indexes(engine)
    inserted = du.upsert_price_bars(engine, sample_prices_df)
    assert inserted == len(sample_prices_df)
    fetched = du.fetch_prices(engine, ["AAA", "BBB"])
    # Ensure at least those rows present
    assert len(fetched) >= len(sample_prices_df)
    # Re-insert to test ON CONFLICT do-nothing behavior (no exception)
    du.upsert_price_bars(engine, sample_prices_df)
