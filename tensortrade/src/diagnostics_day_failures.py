#!/usr/bin/env python3
"""diagnostics_day_failures.py

Per-day IBKR historical bar availability diagnostic for symbols that returned no data
in bulk fetches. Iterates day-by-day over a date range and attempts to fetch a single
1-day daily bar for each symbol using multiple exchanges and whatToShow fallbacks.

Outputs a table indicating which (symbol, date, weekday) combinations have data and
which do not, plus which exchange / whatToShow succeeded first.

Usage:
    python diagnostics_day_failures.py --symbols ALAB,RDDT --start 2024-01-02 --end 2024-01-10

Optional:
    --exchanges SMART,NASDAQ,NYSE --what-to-show TRADES,MIDPOINT,ADJUSTED_LAST --use-rth 1 --rate-limit 0.2

Note: Requires working IBKR connection (Gateway / TWS) and vendored connect_me.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
from typing import List, Optional, Dict
import pandas as pd

# ib_insync + connect_me
try:  # pragma: no cover
    from ib_insync import Stock  # type: ignore
    from ibkr_api.connect_me import connect_me, disconnect_me  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Required imports unavailable: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-day IBKR data availability diagnostics")
    p.add_argument("--symbols", required=True, help="Comma-separated symbols")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--exchanges", default="SMART,NASDAQ,NYSE", help="Comma-separated exchanges to try in order")
    p.add_argument("--what-to-show", default="TRADES,MIDPOINT,ADJUSTED_LAST", help="Comma-separated whatToShow fallbacks")
    p.add_argument("--use-rth", type=int, default=1, help="Use regular trading hours (1 or 0) for initial attempts")
    p.add_argument("--toggle-rth", action="store_true", help="If no bar found with initial useRTH, retry with opposite")
    p.add_argument("--rate-limit", type=float, default=0.2, help="Delay between IBKR requests (seconds)")
    p.add_argument("--component-name", default="pipeline_1", help="Component name for connect_me")
    return p.parse_args()


def daterange(start_date: dt.date, end_date: dt.date):
    current = start_date
    while current  pd.DataFrame:
    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    exchanges = [e.strip().upper() for e in args.exchanges.split(',') if e.strip()]
    wts_list = [w.strip().upper() for w in args.__dict__["what_to_show"].split(',') if w.strip()]
    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)

    ib = await connect_me(args.component_name)

    # Pre-fetch contract details to determine listing (firstTradeDate) and primary exchange
    listing_dates: Dict[str, Optional[dt.date]] = {}
    primary_exchanges: Dict[str, Optional[str]] = {}
    from ib_insync import util  # type: ignore
    for sym in symbols:
        try:
            cd = await ib.reqContractDetailsAsync(Stock(sym, "SMART", "USD"))
            if cd:
                c = cd[0].contract
                raw = getattr(c, "firstTradeDateOrContractMonth", "") or getattr(c, "lastTradeDateOrContractMonth", "")
                if raw and len(raw) >= 8:
                    try:
                        listing_dates[sym] = dt.datetime.strptime(raw[:8], "%Y%m%d").date()
                    except Exception:
                        listing_dates[sym] = None
                else:
                    listing_dates[sym] = None
                primary_exchanges[sym] = getattr(c, "primaryExchange", None)
            else:
                listing_dates[sym] = None
                primary_exchanges[sym] = None
        except Exception:
            listing_dates[sym] = None
            primary_exchanges[sym] = None

    records: List[dict] = []
    try:
        for sym in symbols:
            for day in daterange(start_date, end_date):
                day_found = False
                used_ex = None
                used_wts = None
                used_rth = None
                bar_count = 0
                note = ""
                # IB endDateTime: we point to day end (23:59:59) to ensure full day retrieval
                end_dt = dt.datetime.combine(day, dt.time(23, 59, 59))
                for ex in exchanges:
                    if day_found:
                        break
                    for wts in wts_list:
                        if day_found:
                            break
                        for rth_flag in ([args.use_rth] + ([1 - args.use_rth] if args.toggle_rth else [])):
                            try:
                                # Skip pre-listing days (classify but do not attempt repeated requests)
                                ldate = listing_dates.get(sym)
                                if ldate and day  None:
    if df.empty:
        print("No records produced.")
        return
    print("\n=== Per-Symbol Summary ===")
    for sym, grp in df.groupby("symbol"):
        total = len(grp)
        found = grp[grp.found].shape[0]
        print(f"{sym}: {found}/{total} days have data ({found/total*100:.1f}%)")
    print("\nMissing Days Detail:")
    missing = df[~df.found]
    if missing.empty:
        print("None")
    else:
        print(missing[["symbol", "date", "weekday", "note"]].to_string(index=False))
    # Weekday aggregation excluding pre_listing
    post_listing_missing = missing[missing.note != "pre_listing"]
    if not post_listing_missing.empty:
        print("\nPost-listing missing counts by weekday:")
        agg = post_listing_missing.groupby(["symbol", "weekday"]).size().reset_index(name="missing_days")
        print(agg.to_string(index=False))
    # Listing date summary
    if "listing_date" in df.columns:
        print("\nListing Dates:")
        lsum = df[["symbol", "listing_date"]].drop_duplicates()
        print(lsum.to_string(index=False))


async def main_async():
    args = parse_args()
    df = await diagnose(args)
    summarize(df)
    # Save CSV for further analysis
    out_name = f"diagnostics_day_failures_{args.start}_{args.end}.csv"
    df.to_csv(out_name, index=False)
    print(f"\nSaved detailed results to {out_name}")


def main():
    try:
        asyncio.run(main_async())
    except RuntimeError:
        # Running inside existing event loop (unlikely for CLI) – fallback
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main_async())


if __name__ == "__main__":
    main()
