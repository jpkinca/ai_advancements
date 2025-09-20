"""Standalone Alpha Factor Validation MVP using Alphalens-Reloaded.

Generates synthetic pricing + factor data with a weak but positive predictive signal,
computes forward returns, produces Alphalens tear sheet artifacts, and saves
summary metrics (IC, IC IR, turnover proxy) to JSON.

Run: python alpha_validator_mvp.py --outdir outputs
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless generation
import matplotlib.pyplot as plt

# Alphalens imports
aimport_error = None
try:
    import alphalens as al
except Exception as e:  # capture to allow graceful message if missing
    aimport_error = e

RNG_SEED = 1337

def generate_synthetic_data(n_days: int = 250, n_assets: int = 50, signal_ic: float = 0.06):
    """Generate synthetic daily close prices and a factor with target IC.

    We simulate log returns ~ N(0, 1%) with a common market component + idiosyncratic noise.
    Factor is constructed as linear combination of future returns (with noise) to embed
    controllable predictive power (approx desired IC) while avoiding perfect foresight at inference.
    """
    np.random.seed(RNG_SEED)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    tickers = [f"AST{i:03d}" for i in range(n_assets)]

    # Market + idiosyncratic returns
    market_ret = np.random.normal(0, 0.004, size=n_days)
    idio = np.random.normal(0, 0.01, size=(n_days, n_assets))
    returns = market_ret[:, None] * 0.3 + idio * 0.7

    prices0 = 100 * (1 + np.random.normal(0, 0.01, size=n_assets))
    price_paths = np.vstack([prices0, prices0 * np.cumprod(1 + returns, axis=0)])
    # Align lengths (n_days) by trimming first row used as starting point
    price_paths = price_paths[1:]
    pricing = pd.DataFrame(price_paths, index=dates, columns=tickers)

    # Future returns (1-day ahead) for embedding signal
    fwd_ret = pd.DataFrame(returns, index=dates, columns=tickers).shift(-1)

    # Factor ~ signal_ic * standardized(future return) + noise
    z_fwd = (fwd_ret - fwd_ret.mean()) / fwd_ret.std()
    noise = np.random.normal(0, 1, size=z_fwd.shape)
    factor_raw = signal_ic * z_fwd + np.sqrt(1 - signal_ic**2) * noise

    factor = factor_raw.stack()
    factor.index.set_names(["date", "asset"], inplace=True)

    return pricing, factor


def compute_metrics(factor_data: pd.DataFrame) -> dict:
    """Compute summary metrics from Alphalens factor_data."""
    ic_series = al.performance.factor_information_coefficient(factor_data)
    # IC series is MultiIndex (date, period), get 1D horizon
    if isinstance(ic_series, pd.Series):
        ic_daily = ic_series.xs('1D', level=1)
    else:
        ic_daily = ic_series['1D'] if '1D' in ic_series.columns else ic_series.iloc[:, 0]
    ic_mean = ic_daily.mean()
    ic_std = ic_daily.std(ddof=1)
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan

    quantile_ret = al.performance.mean_return_by_quantile(factor_data)[0]
    # quantile_ret: index=quantiles (1-5), columns=periods ('1D','5D','10D')
    q_spread = (quantile_ret.loc[quantile_ret.index.max(), :] - quantile_ret.loc[quantile_ret.index.min(), :]).mean()

    # Turnover proxy: use 1D turnover if available
    try:
        turnover = al.performance.factor_turnover(factor_data, period=1).mean().mean()
    except AttributeError:
        turnover = np.nan  # Not available in this version

    return {
        "ic_mean": float(ic_mean),
        "ic_ir": float(ic_ir),
        "q_spread_est": float(q_spread),
        "turnover_proxy": float(turnover),
        "n_obs": int(factor_data.shape[0])
    }


def main():
    parser = argparse.ArgumentParser(description="Alpha factor validation MVP")
    parser.add_argument("--outdir", default="outputs", help="Directory for tear sheet and metrics")
    parser.add_argument("--days", type=int, default=250)
    parser.add_argument("--assets", type=int, default=50)
    parser.add_argument("--target_ic", type=float, default=0.06)
    args = parser.parse_args()

    if aimport_error is not None:
        raise SystemExit(f"Alphalens import failed: {aimport_error}. Did you install dependencies?")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1] Generating synthetic data ...")
    pricing, factor = generate_synthetic_data(args.days, args.assets, args.target_ic)

    # Align factor & pricing index
    print("[2] Building factor_data via Alphalens utils ...")
    factor_data = al.utils.get_clean_factor_and_forward_returns(
        factor,
        pricing,
        quantiles=5,
        periods=(1, 5, 10),
        max_loss=0.35
    )

    print("[3] Creating tear sheet (saved to PNGs) ...")
    # Instead of interactive full tear sheet (lots of plots), selectively create key ones
    al.tears.create_information_tear_sheet(factor_data, by_group=False)
    plt.savefig(outdir / "information_tear_sheet.png", dpi=150, bbox_inches="tight")
    plt.close('all')

    al.tears.create_returns_tear_sheet(factor_data)
    plt.savefig(outdir / "returns_tear_sheet.png", dpi=150, bbox_inches="tight")
    plt.close('all')

    print("[4] Computing metrics ...")
    metrics = compute_metrics(factor_data)

    metrics_path = outdir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[DONE] Metrics written to", metrics_path)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
