"""watchlist_loader.py
Utility functions for loading the unified watchlist, fetching historical prices,
and (optionally) constructing TensorTrade data streams for environment creation.

Designed to be resilient if TensorTrade is not installed (graceful fallbacks).

Usage (basic):
    from watchlist_loader import load_watchlist, fetch_price_history
    symbols = load_watchlist()
    data = fetch_price_history(symbols, start="2023-01-01", end="2024-12-31")

Usage (with TensorTrade streams):
    from watchlist_loader import build_price_streams
    streams = build_price_streams(data)

The resulting `streams` dict can be integrated into a TensorTrade DataFeed.

Dependencies (install manually if needed):
    pip install pandas numpy tensortrade-ng ib_insync

Note: For large symbol sets or intraday data, consider batching requests or caching
in a local database using the tt_ schema.
"""
from __future__ import annotations

import os
import datetime as dt
from typing import List, Dict, Optional

import pandas as pd
import asyncio

# IBKR / ib_insync
try:  # pragma: no cover - requires running IB Gateway / TWS
    from ib_insync import Stock, util  # type: ignore
    ib_available = True
except Exception:  # noqa: E722
    Stock = None  # type: ignore
    util = None  # type: ignore
    ib_available = False

# Standard connection helper from local connect_me
try:
    # Try local connect_me first
    from .connect_me import connect_me  # type: ignore
except Exception:  # pragma: no cover
    try:
        # Try direct import
        from connect_me import connect_me  # type: ignore
    except Exception:  # noqa: E722
        connect_me = None  # type: ignore

# Optional TensorTrade imports (graceful fallback)
try:  # pragma: no cover - environment dependent
    from tensortrade.feed import Stream
except Exception:  # pragma: no cover
    class Stream:  # minimal stub
        def __init__(self, name: str, data):
            self.name = name
            self.data = data
        def rename(self, name: str):
            self.name = name
            return self

WATCHLIST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tensorwatchlist.csv")


def load_watchlist(path: Optional[str] = None) -> List[str]:
    """Load watchlist CSV. Accepts formats: single column 'symbol' or two columns symbol,description.
    Returns list of symbols.
    """
    path = path or WATCHLIST_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Watchlist file not found: {path}")

    # Try to parse flexible CSV: either one-column or multi-column.
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().lower()
    if "symbol" in header:
        df = pd.read_csv(path)
        symbols = [str(s).strip() for s in df.iloc[:, 0].dropna().unique()]
    else:  # fallback: comma-separated line
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        symbols = [s.strip() for s in content.split(',') if s.strip()]
    return symbols


def _duration_str_from_days(duration_days: int) -> str:
    """Convert duration in days to IBKR-compliant duration string.
    
    IBKR requires durations > 365 days to be specified in years.
    """
    if duration_days > 365:
        # Convert to years for requests longer than 1 year
        years = duration_days / 365.25  # Account for leap years
        return f"{int(years)} Y"
    else:
        return f"{duration_days} D"


def fetch_price_history(
    symbols: List[str],
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    source: str = "ibkr",
    **kwargs,
) -> pd.DataFrame:
    """Historical price fetcher using IBKR via ib_insync & connect_me.

    Args:
        symbols: list of ticker symbols.
        start: ISO date (YYYY-MM-DD).
        end: ISO date; defaults to today.
        interval: Bar size alias ("1d", "1h", etc.).
        source: "ibkr" (only supported source).
        **kwargs: extra passthrough (e.g., whatToShow, useRTH, rate_limit_delay, batch_size, retry_attempts).

    Returns:
        Long-format DataFrame with columns [symbol, datetime, open, high, low, close, volume].
    """
    if source.lower() != "ibkr":
        raise ValueError("Only 'ibkr' data source is supported")
    return fetch_price_history_ibkr(symbols, start=start, end=end, interval=interval, **kwargs)


def _bar_size_from_interval(interval: str) -> str:
    mapping = {
        "1d": "1 day",
        "1h": "1 hour",
        "1m": "1 min",
        "5m": "5 mins",
        "15m": "15 mins",
        "30m": "30 mins",
        "1w": "1 week",
    }
    return mapping.get(interval.lower(), "1 day")


def fetch_price_history_ibkr(
    symbols: List[str],
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    whatToShow: str = "TRADES",
    useRTH: bool = True,
    timeout: int = 60,
    component_name: str = "pipeline_1",
    rate_limit_delay: float = 0.15,  # 150ms between requests
    batch_size: int = 10,  # Process symbols in batches
    retry_attempts: int = 3,
    enable_exchange_fallback: bool = True,
    exchange_fallbacks: Optional[List[str]] = None,
    adjust_for_listing: bool = True,
    advanced: bool = False,
    fallback_what_to_show: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fetch historical bars from IBKR using ib_insync & the standardized connect_me helper.

    Enhanced with:
    - Rate limiting between symbol requests (IBKR pacing requirements)
    - Batch processing to prevent connection timeouts
    - Automatic retry logic for failed requests
    - Single connection session for entire dataset
    - Proper connection cleanup

    Constraints:
        - IBKR enforces pacing; this implementation adds 150ms delay between requests
        - Connection reused across all symbols in a single session
        - Failed symbols are retried up to 3 times with exponential backoff

    Args:
        symbols: List of ticker symbols to fetch
        start: Start date in ISO format (YYYY-MM-DD)
        end: End date in ISO format (YYYY-MM-DD), defaults to today
        interval: Bar size ("1d", "1h", etc.)
        whatToShow: Data type to show ("TRADES", "MIDPOINT", etc.)
        useRTH: Use regular trading hours only
        timeout: Connection timeout in seconds
        component_name: Component name for connection tracking
        rate_limit_delay: Delay between requests in seconds
        batch_size: Number of symbols to process before connection check
        retry_attempts: Number of retry attempts for failed symbols

    Returns:
        Long-format DataFrame with columns: symbol, datetime, open, high, low, close, volume
    """
    if not ib_available:
        raise RuntimeError("IBKR data source unavailable: ib_insync not importable.")
    
    # Create direct IB connection if connect_me is not available
    if connect_me is None:
        def simple_connect():
            from ib_insync import IB
            ib = IB()
            try:
                ib.connect('127.0.0.1', 7497, clientId=1)  # TWS paper trading port
                return ib
            except:
                try:
                    ib.connect('127.0.0.1', 7496, clientId=1)  # TWS live trading port
                    return ib
                except:
                    ib.connect('127.0.0.1', 4002, clientId=1)  # IB Gateway paper trading port
                    return ib
        connect_func = simple_connect
    else:
        connect_func = connect_me

    end_date = end or dt.date.today().isoformat()
    bar_size = _bar_size_from_interval(interval)

    # Convert date strings to IBKR-friendly endDateTime & duration
    start_dt = dt.datetime.fromisoformat(start)
    end_dt = dt.datetime.fromisoformat(end_date)
    if end_dt  pd.DataFrame:
            ib = await connect_me(component_name)
            frames: List[pd.DataFrame] = []
            contract_cache: Dict[str, any] = {}
            primary_exchange_cache: Dict[str, str] = {}
            try:
                for i, sym in enumerate(symbols):
                    if i > 0:
                        await asyncio.sleep(rate_limit_delay)
                    success = False
                    # Qualify contract once (fills primaryExchange, conId, etc.)
                    if sym not in contract_cache:
                        try:
                            base = Stock(sym, "SMART", "USD")
                            ib.qualifyContracts(base)  # synchronous in ib_insync
                            contract_cache[sym] = base
                            if getattr(base, 'primaryExchange', None):
                                primary_exchange_cache[sym] = base.primaryExchange  # type: ignore[attr-defined]
                        except Exception as qe:  # pragma: no cover
                            print(f"⚠️ Qualification failed for {sym}: {qe}")
                            contract_cache[sym] = Stock(sym, "SMART", "USD")
                    base_contract = contract_cache[sym]
                    primary_ex = primary_exchange_cache.get(sym)
                    for wts in wt_sequence:
                        try:
                            # First attempt with SMART
                            contract = base_contract
                            bars = await ib.reqHistoricalDataAsync(
                                contract,
                                endDateTime=end_dt,
                                durationStr=duration_str,
                                barSizeSetting=bar_size,
                                whatToShow=wts,
                                useRTH=1 if useRTH else 0,
                                formatDate=1,
                            )
                            # If no data and we have a distinct primary exchange, try that
                            if (not bars) and primary_ex and primary_ex != "SMART":
                                try:
                                    px_contract = Stock(sym, primary_ex, "USD")
                                    bars = await ib.reqHistoricalDataAsync(
                                        px_contract,
                                        endDateTime=end_dt,
                                        durationStr=duration_str,
                                        barSizeSetting=bar_size,
                                        whatToShow=wts,
                                        useRTH=1 if useRTH else 0,
                                        formatDate=1,
                                    )
                                except Exception as pxe:  # pragma: no cover
                                    print(f"⚠️ {sym} primaryEx retry error: {pxe}")
                            if bars:
                                df = util.df(bars)
                                if df.empty:
                                    continue
                                df = df.rename(columns={"date": "datetime"})
                                df["datetime"] = pd.to_datetime(df["datetime"])
                                keep = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
                                df = df[keep].copy()
                                if "volume" not in df.columns:
                                    df["volume"] = 0
                                df["symbol"] = sym
                                df["whatToShow_used"] = wts
                                if primary_ex:
                                    df["primaryExchange"] = primary_ex
                                frames.append(df)
                                print(f"✅ {sym} ({i+1}/{len(symbols)}) {len(df)} bars via {wts}")
                                success = True
                                break
                        except Exception as e:  # pragma: no cover
                            print(f"⚠️ {sym} wts={wts} error: {e}")
                            await asyncio.sleep(rate_limit_delay)
                    if not success:
                        print(f"❌ No data for {sym}")
                if not frames:
                    print("⚠️ No historical data retrieved for any symbols in lean mode.")
                    return pd.DataFrame()  # Return empty DataFrame instead of raising error
                full = pd.concat(frames, ignore_index=True).sort_values(["symbol", "datetime"])
                return full
            finally:
                try:
                    from .connect_me import disconnect_me
                    await disconnect_me(component_name)
                except Exception:
                    pass

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            return asyncio.ensure_future(_lean())  # type: ignore
        return asyncio.run(_lean())  # type: ignore

    # Advanced path (existing rich logic)
    async def _async_fetch_with_rate_limiting() -> pd.DataFrame:
        # Single connection for entire dataset
        ib = await connect_me(component_name)
        frames: List[pd.DataFrame] = []
        failed_symbols: List[str] = []
        no_data_symbols: List[str] = []
        no_data_reasons: Dict[str, str] = {}
        exch_fallbacks = exchange_fallbacks or ["SMART", "NASDAQ", "NYSE", "ARCA", "AMEX"]

        async def fetch_contract_details(sym: str):  # pragma: no cover - network dependent
            try:
                base = Stock(sym, "SMART", "USD")
                cds = await ib.reqContractDetailsAsync(base)
                if cds:
                    return cds[0]
            except Exception:
                return None
            return None

        def listing_date_from_details(cd) -> Optional[dt.date]:  # pragma: no cover
            try:
                # firstTradeDateOrContractMonth sometimes blank; lastTradeDateOrContractMonth is for expiry instruments
                raw = getattr(cd.contract, "firstTradeDateOrContractMonth", "") or getattr(cd.contract, "lastTradeDateOrContractMonth", "")
                if raw and len(raw) >= 8:
                    return dt.datetime.strptime(raw[:8], "%Y%m%d").date()
            except Exception:
                return None
            return None
        
        try:
            # Process symbols with rate limiting
            for i, sym in enumerate(symbols):
                # Add delay between requests for IBKR pacing (except first request)
                if i > 0:
                    await asyncio.sleep(rate_limit_delay)
                
                # Batch checkpoint - verify connection is still active
                if i > 0 and i % batch_size == 0:
                    if not (hasattr(ib, 'isConnected') and ib.isConnected()):
                        print(f"Warning: Connection lost at symbol {i}/{len(symbols)}, reconnecting...")
                        ib = await connect_me(component_name, force_reconnect=True)
                
                success = False
                for attempt in range(retry_attempts):
                    try:
                        # Determine dynamic duration if listing date after requested start
                        dynamic_duration_str = duration_str
                        eff_start_dt = start_dt
                        contract_details = None
                        listing_date = None
                        if adjust_for_listing and attempt == 0:  # fetch details once per symbol initial attempt
                            contract_details = await fetch_contract_details(sym)
                            if contract_details:
                                listing_date = listing_date_from_details(contract_details)
                                if listing_date and listing_date > end_dt.date():
                                    # Symbol listed after requested window
                                    no_data_symbols.append(sym)
                                    no_data_reasons[sym] = "pre_listing_window"
                                    success = True
                                    break
                                if listing_date and listing_date > start_dt.date():
                                    eff_start_dt = dt.datetime.combine(listing_date, dt.time.min)
                                    dyn_days = (end_dt.date() - eff_start_dt.date()).days + 1
                                    dynamic_duration_str = _duration_str_from_days(dyn_days)

                        # Try primary exchange list
                        exchanges_to_try = ["SMART"]
                        if enable_exchange_fallback:
                            exchanges_to_try = exch_fallbacks

                        bars = []
                        last_exc = None
                        for ex in exchanges_to_try:
                            contract = Stock(sym, ex, "USD")
                            try:
                                bars = await ib.reqHistoricalDataAsync(
                                    contract,
                                    endDateTime=end_dt,
                                    durationStr=dynamic_duration_str,
                                    barSizeSetting=bar_size,
                                    whatToShow=whatToShow,
                                    useRTH=1 if useRTH else 0,
                                    formatDate=1,
                                )
                                if bars:
                                    break
                            except Exception as ex_err:  # pragma: no cover network
                                last_exc = ex_err
                                continue
                        
                        if not bars:
                            reason = "no_data"
                            if sym in no_data_reasons:
                                reason = no_data_reasons[sym]
                            elif listing_date and listing_date > start_dt.date():
                                reason = "post_listing_adjusted_empty"
                            print(f"⚠️ No historical data returned for {sym} (reason={reason}).")
                            no_data_symbols.append(sym)
                            no_data_reasons.setdefault(sym, reason)
                            success = True
                            break
                            
                        df = util.df(bars)
                        
                        # Normalize columns
                        rename_map = {"date": "datetime", "volume": "volume"}
                        df = df.rename(columns=rename_map)
                        if "datetime" not in df.columns and "date" in df.columns:
                            df["datetime"] = pd.to_datetime(df["date"])
                        df["datetime"] = pd.to_datetime(df["datetime"])
                        
                        # Keep only required columns
                        keep = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
                        df = df[keep].copy()
                        if "volume" not in df.columns:
                            df["volume"] = 0
                        df["symbol"] = sym
                        frames.append(df)
                        
                        success = True
                        print(f"✅ Fetched {len(df)} bars for {sym} ({i+1}/{len(symbols)})")
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if attempt  Dict[str, Dict[str, Stream]]:
    """Build per-symbol OHLCV streams suitable for a TensorTrade DataFeed.

    Returns a nested dict: { symbol: { 'open': Stream, 'high': Stream, ... } }

    The caller can then flatten or select streams to feed into a DataFeed, e.g.:
        from tensortrade.feed import DataFeed
        price_streams = build_price_streams(df)
        streams = []
        for sym, comps in price_streams.items():
            streams.extend(comps.values())
        feed = DataFeed(streams)
    """
    required = {"symbol", "datetime", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(map(str.lower, price_df.columns))):
        raise ValueError(f"DataFrame missing required columns: {required}")

    # Ensure correct types
    df = price_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    result: Dict[str, Dict[str, Stream]] = {}
    for sym, grp in df.groupby('symbol'):
        grp = grp.sort_values('datetime')
        # Create raw lists (TensorTrade Stream can wrap list/iterable)
        result[sym] = {
            'open': Stream.source(grp['open'].tolist(), name=f"{sym}:open") if hasattr(Stream, 'source') else Stream(f"{sym}:open", grp['open'].tolist()),
            'high': Stream.source(grp['high'].tolist(), name=f"{sym}:high") if hasattr(Stream, 'source') else Stream(f"{sym}:high", grp['high'].tolist()),
            'low': Stream.source(grp['low'].tolist(), name=f"{sym}:low") if hasattr(Stream, 'source') else Stream(f"{sym}:low", grp['low'].tolist()),
            'close': Stream.source(grp['close'].tolist(), name=f"{sym}:close") if hasattr(Stream, 'source') else Stream(f"{sym}:close", grp['close'].tolist()),
            'volume': Stream.source(grp['volume'].tolist(), name=f"{sym}:volume") if hasattr(Stream, 'source') else Stream(f"{sym}:volume", grp['volume'].tolist()),
        }
    return result


def example_pipeline(start: str = "2024-01-01", end: Optional[str] = None, interval: str = "1d"):
    """Convenience function: load watchlist, fetch prices, build streams.
    Returns (symbols, price_df, streams_nested_dict)
    """
    symbols = load_watchlist()
    price_df = fetch_price_history(symbols, start=start, end=end, interval=interval, source="ibkr")
    streams = build_price_streams(price_df)
    return symbols, price_df, streams


if __name__ == "__main__":  # Manual quick test
    try:
        syms, df, streams = example_pipeline()
        print(f"Loaded {len(syms)} symbols, price rows: {len(df)}; built streams for {len(streams)} symbols")
    except Exception as exc:
        print(f"Example pipeline failed: {exc}")
