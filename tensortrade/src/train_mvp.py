"""train_mvp.py
Minimal retrospective MVP training script.

Goals:
  - Use first N (default 10) symbols from watchlist.
  - Load 3 months of historical daily bars from IBKR.
  - Persist to PostgreSQL (tt_prices) if not already present.
  - Build a simple TensorTrade environment with:
       * VolatilityTargetedAction (multi-asset)
       * RiskAwareReward
       * DrawdownStopper
  - Train a small PPO model for demonstration.

Assumptions:
  - DATABASE_URL env var points to PostgreSQL with tt_ schema created.
  - IBKR Gateway/TWS available for historical data access.
  - tensortrade_risk_module.py is on PYTHONPATH (same src directory).

Example:
  python -m train_mvp --months 3 --steps 5000 --limit 10

Note: This is intentionally minimal; no hyperparameter tuning or advanced features.
"""
from __future__ import annotations

import argparse
import datetime as dt
import pandas as pd
from typing import List
import sys, pathlib
import site
from dotenv import load_dotenv
load_dotenv() # take environment variables from .env.

import nest_asyncio
nest_asyncio.apply()

USE_TENSORTRADE = True
# Attempt to prioritize site-packages version of tensortrade if a local folder name collision exists.
try:
    site_paths = []
    for p in site.getsitepackages():
        tp = pathlib.Path(p) / 'tensortrade'
        if tp.exists():
            site_paths.append(str(p))
    for sp in site_paths:
        if sp not in sys.path:
            sys.path.insert(0, sp)
    from tensortrade.env.default import create  # type: ignore
    from tensortrade.oms.exchanges.simulated import SimulatedExchange  # type: ignore
    from tensortrade.oms.instruments import USD, Instrument  # type: ignore
    from tensortrade.oms.wallets import Portfolio, Wallet  # type: ignore
except Exception:
    USE_TENSORTRADE = False
    from simple_env import SimpleMultiAssetEnv, SimpleEnvConfig  # type: ignore

try:
    from stable_baselines3 import PPO
except Exception as e:  # pragma: no cover
    raise ImportError("stable-baselines3 required: pip install stable-baselines3") from e


def parse_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MVP RL training over historical data (extended TensorTrade capabilities)")
    p.add_argument("--months", type=int, default=3, help="Number of months of history (approx 30d * months)")
    p.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), default today")
    p.add_argument("--limit", type=int, default=10, help="Limit number of symbols (first N in watchlist)")
    p.add_argument("--steps", type=int, default=10000, help="Total PPO timesteps")
    p.add_argument("--window", type=int, default=30, help="Observer window size")
    p.add_argument("--rate-limit", type=float, default=0.15, help="Delay between IBKR requests (seconds)")
    p.add_argument("--batch-size", type=int, default=10, help="Symbols per batch before connection check")
    p.add_argument("--risk-target", type=float, default=0.01, help="Target daily risk for volatility-targeted action scheme")
    p.add_argument("--max-leverage", type=float, default=1.0, help="Max leverage for volatility-targeted scheme")
    p.add_argument("--max-dd", type=float, default=0.2, help="Max drawdown stopper threshold")
    p.add_argument("--min-equity", type=float, default=None, help="Minimum net worth floor (optional)")
    p.add_argument("--log-interval", type=int, default=1000, help="PPO log interval timesteps")
    p.add_argument("--log-training", action="store_true", help="Wrap env to log per-step actions & rewards to DB during training (not just evaluation)")
    p.add_argument("--config", type=str, default=None, help="Optional JSON/YAML config file overriding CLI params")
    p.add_argument("--no-eval", action="store_true", help="Skip post-training evaluation episode")
    p.add_argument("--eval-episodes", type=int, default=1, help="Number of evaluation episodes after training")
    # Data fetch extensions
    p.add_argument("--advanced-fetch", action="store_true", help="Use advanced historical fetch path (exchange fallbacks, listing date heuristics)")
    p.add_argument("--fallback-wts", type=str, default="MIDPOINT", help="Comma list of fallback whatToShow values (lean mode only)")
    # Feature / feed options
    p.add_argument("--with-features", action="store_true", help="Augment DataFeed with engineered features (returns, rolling vol, normalized volume)")
    # Action / reward scheme selection
    p.add_argument("--action-scheme", choices=["vol_target", "discrete_simple"], default="vol_target", help="Select action scheme")
    p.add_argument("--reward-scheme", choices=["risk_aware", "returns"], default="risk_aware", help="Select reward scheme type")
    p.add_argument("--allow-shorts", action="store_true", help="Enable short positions in discrete simple action scheme")
    # Portfolio / exchange
    p.add_argument("--initial-cash", type=float, default=10000.0, help="Initial portfolio cash in USD")
    # Model path
    p.add_argument("--model-path", type=str, default=None, help="Optional explicit model save path (.zip)")
    p.add_argument("--db-url", type=str, default=None, help="Override DATABASE_URL env var (PostgreSQL URI)")
    return p.parse_args()


def date_range(months: int, end: str | None) -> tuple[str, str]:
    end_date = dt.date.today() if end is None else dt.date.fromisoformat(end)
    start_date = end_date - dt.timedelta(days=30 * months)
    return start_date.isoformat(), end_date.isoformat()


def ensure_history(symbols: List[str], start: str, end: str, rate_limit: float = 0.15, batch_size: int = 10,
                   advanced_fetch: bool = False, fallback_wts: List[str] | None = None, db_url: str | None = None) -> pd.DataFrame:
    from db_utils import get_engine, ensure_tables, ensure_indexes, upsert_price_bars, fetch_prices
    from watchlist_loader import fetch_price_history
    engine = get_engine(db_url)
    # Ensure schema first
    ensure_tables(engine)
    ensure_indexes(engine)
    fetched = fetch_price_history(
        symbols,
        start=start,
        end=end,
        interval="1d",
        source="ibkr",
        rate_limit_delay=rate_limit,
        batch_size=batch_size,
        retry_attempts=3,
        advanced=advanced_fetch,
        fallback_what_to_show=fallback_wts,
    )
    if hasattr(fetched, "__await__"):
        raise RuntimeError("Asynchronous fetch detected; run outside running event loop.")
    upsert_price_bars(engine, fetched)
    combined = fetch_prices(engine, symbols, start, end)
    return combined


def build_env(symbols: List[str], price_df: pd.DataFrame, window_size: int, args):
    """Construct a TensorTrade environment with selected schemes and optional engineered features."""
    exchange = SimulatedExchange(base_instrument=USD)
    instruments = [Instrument(sym, 2, sym) for sym in symbols]
    portfolio = Portfolio(USD, [Wallet(exchange, args.initial_cash * USD)])

    streams_nested = build_price_streams(price_df)
    from tensortrade.feed import Stream, DataFeed
    feature_streams = []
    for sym in symbols:
        if sym not in streams_nested:
            continue
        sym_streams = streams_nested[sym]
        base_close = sym_streams['close']
        feature_streams.append(base_close)
        if args.with_features:
            # Simple engineered features: pct change and rolling volatility (window 5 & 10) + normalized volume
            # We build features in pandas then wrap as Streams (keeps lean, avoids dependency on talib)
            grp = price_df[price_df.symbol == sym].sort_values('datetime')
            closes = grp['close'].values
            volumes = grp['volume'].values
            import numpy as np
            pct = np.concatenate([[0.0], np.diff(closes) / (closes[:-1] + 1e-12)])
            vol5 = pd.Series(pct).rolling(5).std().fillna(0).values
            vol10 = pd.Series(pct).rolling(10).std().fillna(0).values
            vol_norm = (volumes - volumes.mean()) / (volumes.std() + 1e-12)
            # Build streams (align lengths)
            feature_streams.append(Stream.source(pct.tolist(), name=f"{sym}:ret_1"))
            feature_streams.append(Stream.source(vol5.tolist(), name=f"{sym}:vol5"))
            feature_streams.append(Stream.source(vol10.tolist(), name=f"{sym}:vol10"))
            feature_streams.append(Stream.source(vol_norm.tolist(), name=f"{sym}:vol_norm"))

    feed = DataFeed(feature_streams)
    feed.compile()

    # Action scheme selection
    if args.action_scheme == "vol_target":
        action_scheme = VolatilityTargetedAction(
            portfolio=portfolio,
            config=VolTargetedActionConfig(
                target_daily_risk=args.risk_target,
                max_leverage=args.max_leverage,
                vol_window=20,
                use_equity_vol=True,
                instrument_symbols=symbols
            )
        )
    else:
        action_scheme = SimpleDiscreteAction(
            portfolio=portfolio,
            config=SimpleDiscreteActionConfig(
                instrument_symbols=symbols,
                allow_shorts=args.allow_shorts
            )
        )

    # Reward scheme selection
    if args.reward_scheme == "risk_aware":
        reward_scheme = RiskAwareReward(RiskAwareRewardConfig())
    else:
        # Fallback to TensorTrade's default simple returns reward if available
        try:  # pragma: no cover
            from tensortrade.env.default.rewards import RiskAdjustedReturns
            reward_scheme = RiskAdjustedReturns()
        except Exception:
            reward_scheme = RiskAwareReward(RiskAwareRewardConfig())

    stopper = DrawdownStopper(DrawdownStopperConfig(max_drawdown=args.max_dd, min_net_worth=args.min_equity, warmup_steps=window_size))

    if USE_TENSORTRADE:
        env = create(
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            stopper=stopper,
            window_size=window_size,
            feed=feed
        )
    else:
        feature_cols = ["close"]
        for extra in ["return_1", "vol_10", "vol_20"]:
            if extra in price_df.columns:
                feature_cols.append(extra)
        cfg = SimpleEnvConfig(symbols=symbols, feature_cols=feature_cols, window=window_size, allow_shorts=args.allow_shorts)
        env = SimpleMultiAssetEnv(price_df, cfg)

    return env


def main():
    from watchlist_loader import load_watchlist
    args = parse_args()
    # Optional config file override (JSON or YAML minimal support)
    if args.config:
        import json, os
        try:
            if args.config.lower().endswith(('.yaml', '.yml')):
                try:
                    import yaml  # type: ignore
                except Exception as e:  # pragma: no cover
                    raise RuntimeError("PyYAML not installed; install pyyaml or provide JSON config") from e
                with open(args.config, 'r', encoding='utf-8') as fh:
                    cfg_data = yaml.safe_load(fh) or {}
            else:
                with open(args.config, 'r', encoding='utf-8') as fh:
                    cfg_data = json.load(fh)
            # Override simple scalar params present in config
            for k, v in cfg_data.items():
                if hasattr(args, k.replace('-', '_')):
                    setattr(args, k.replace('-', '_'), v)
            print(f"Loaded config overrides from {args.config}")
        except FileNotFoundError:
            print(f"Warning: config file {args.config} not found; proceeding with CLI args")
        except Exception as exc:
            print(f"Warning: failed parsing config file {args.config}: {exc}")
    start, end = date_range(args.months, args.end)
    print(f"Training window: {start} -> {end} ({args.months} months approx)")

    symbols = load_watchlist()[: args.limit]
    print(f"Symbols selected: {symbols}")

    print("Fetching & storing history ...")
    fallback_wts = [w.strip() for w in args.fallback_wts.split(',')] if args.fallback_wts else None
    price_df = ensure_history(symbols, start, end, args.rate_limit, args.batch_size, args.advanced_fetch, fallback_wts, db_url=args.db_url)
    print(f"Historical bars loaded: {len(price_df)} rows")

    for sym in symbols:
        count = (price_df.symbol == sym).sum()
        if count  per instrument logging + P&L tracking
            try:
                import numpy as np
                arr = np.asarray(action).reshape(-1)
                
                # Update current prices (attempt to get from environment)
                try:
                    for i, sym in enumerate(symbols[:len(arr)]):
                        if hasattr(env, 'prices') and sym in env.prices:
                            current_prices[sym] = float(env.prices[sym])
                        elif hasattr(env, '_last_prices') and sym in env._last_prices:
                            current_prices[sym] = float(env._last_prices[sym])
                        # If unable to get current price, keep the last known price
                except Exception:
                    pass
                
                # Process each action for P&L tracking
                for i, sym in enumerate(symbols[:len(arr)]):
                    action_weight = float(arr[i])
                    current_price = current_prices.get(sym, 100.0)
                    
                    # Log action to database
                    insert_action(engine, episode_id, ts, sym, action_weight)
                    
                    # Process P&L tracking
                    try:
                        pnl_tracker.process_action(
                            symbol=sym,
                            new_weight=action_weight,
                            current_price=current_price,
                            timestamp=ts,
                            portfolio_value=100000.0  # Default portfolio size
                        )
                    except Exception as e:
                        print(f"P&L tracking error for {sym}: {e}")
                
                # Turnover (average absolute change across instruments)
                if action_prev is not None and len(arr) == len(action_prev):
                    turnover_acc += float(np.mean(np.abs(arr - action_prev)))
                    turnover_steps += 1
                action_prev = arr.copy()
            except Exception as e:
                print(f"Action processing error: {e}")
                pass
            insert_reward(engine, episode_id, ts, float(reward))
            total_reward += reward
            # Equity tracking for metrics
            try:
                # Common attribute patterns
                eq = None
                for attr in ["net_worth", "portfolio_value", "equity"]:
                    if hasattr(env, attr):
                        eq = getattr(env, attr)
                        if callable(eq):
                            eq = eq()
                        break
                if eq is None and hasattr(env, "portfolio") and hasattr(env.portfolio, "net_worth"):
                    eq = env.portfolio.net_worth
                if eq is not None:
                    eq_f = float(eq)
                    equity_hist.append(eq_f)
                    if eq_f > max_equity:
                        max_equity = eq_f
                    if max_equity > 0:
                        dd = 1.0 - eq_f / max_equity
                        if dd > max_dd:
                            max_dd = dd
            except Exception:
                pass
            if truncated:
                done = True
        
        # Generate P&L report at end of episode
        try:
            print(f"\n🎯 Episode {ep+1} P&L Summary:")
            current_pnl = pnl_tracker.get_current_pnl(current_prices)
            total_unrealized = sum(current_pnl.values())
            print(f"   Current Unrealized P&L: ${total_unrealized:,.2f}")
            
            # Get full P&L summary
            pnl_summary = pnl_tracker.get_pnl_summary()
            print(f"   Total Trades: {pnl_summary.total_trades}")
            print(f"   Net P&L: ${pnl_summary.net_pnl:,.2f}")
            print(f"   Win Rate: {pnl_summary.win_rate:.1%}")
            if pnl_summary.total_trades > 10:  # Detailed report for longer episodes
                pnl_tracker.print_pnl_report()
        except Exception as e:
            print(f"P&L summary error: {e}")
        
        # Compute sharpe-like metric (daily approximation)
        sharpe_ratio = None
        try:
            import numpy as np
            if len(equity_hist) > 1:
                eq_arr = np.asarray(equity_hist)
                rets = np.diff(eq_arr) / (eq_arr[:-1] + 1e-12)
                if rets.std() > 0:
                    sharpe_ratio = float(rets.mean() / rets.std() * (252 ** 0.5))  # annualized approx
                else:
                    sharpe_ratio = 0.0
        except Exception:
            sharpe_ratio = None
        turnover = (turnover_acc / turnover_steps) if turnover_steps > 0 else None
        finalize_episode(
            engine,
            episode_id,
            datetime.utcnow(),
            stop_reason="evaluation_complete",
            max_drawdown=float(max_dd) if max_dd else None,
            sharpe_ratio=sharpe_ratio,
            turnover=turnover,
        )
        print(
            f"Evaluation episode {ep+1}/{args.eval_episodes} total reward: {total_reward:.4f} "
            f"sharpe={sharpe_ratio:.3f if sharpe_ratio is not None else 'NA'} max_dd={max_dd:.3%} "
            f"turnover={turnover:.4f if turnover is not None else 'NA'} (episode_id={episode_id})"
        )


if __name__ == "__main__":
    main()
