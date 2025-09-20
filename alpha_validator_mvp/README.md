# Alpha Validator MVP

A minimal standalone script demonstrating how to validate a synthetic alpha factor using **Alphalens-Reloaded**. It generates synthetic OHLCV close prices and a predictive factor with a controllable target Information Coefficient (IC), runs Alphalens preprocessing, produces selected tear sheet plots, and outputs summary metrics to JSON.

## Features
- Synthetic multi-asset price path generation (market + idiosyncratic components)
- Factor engineered to approximate target IC vs. next-day returns
- Alphalens cleaning + forward returns calculation (1, 5, 10 day horizons)
- Information & returns tear sheet PNG exports (headless)
- Summary metrics: mean IC, IC IR, quantile spread estimate, turnover proxy, observation count

## Quick Start

```bash
# (Optional) Create and activate a virtual environment first
pip install -r requirements.txt
python alpha_validator_mvp.py --outdir outputs --days 300 --assets 75 --target_ic 0.07
```

Outputs:
- `outputs/information_tear_sheet.png`
- `outputs/returns_tear_sheet.png`
- `outputs/metrics_summary.json`

## metrics_summary.json Example
```json
{
  "ic_mean": 0.055,
  "ic_ir": 0.85,
  "q_spread_est": 0.0043,
  "turnover_proxy": 0.62,
  "n_obs": 37500
}
```
(Note: Values depend on RNG seed and parameters.)

## Parameter Tuning
- `--target_ic`: Controls signal strength (0.0–0.15 realistic for synthetic). Higher increases factor-return correlation.
- `--days`: History length (more days → more stable IC estimates).
- `--assets`: Universe size.

## Extending Toward Full Service
1. Replace synthetic generator with DB/market data fetch.
2. Persist factor metrics to a database table (e.g., Postgres) instead of JSON.
3. Add rolling IC monitoring & threshold-based alerts.
4. Integrate sector/regime grouping via `groupby=` parameter.
5. Containerize and expose as an API endpoint.

## Troubleshooting
- ImportError: Ensure you installed `alphalens-reloaded` (not the deprecated original) and that your Python version is supported.
- Empty factor_data: Check `max_loss` or adjust universe size.
- Slow plotting: Reduce horizons or skip heavy tear sheet functions.

## License
MIT (adapt as needed).
