# Proposed Initial Parameter Settings for TensorTrade Paper Trading

- **Initial Cash Balance:**
  - $10,000 (adjust as needed)

- **Watchlist (Instrument List):**
  - To be provided by user (e.g., ["TSLA", "AAPL", "MSFT"])

- **Trading Fees/Commissions:**
  - 0.1% per trade (realistic for most brokers)

- **Leverage Limit:**
  - 1.0 (long-only, no leverage; increase for margin trading)

- **Risk Parameters:**
  - max_drawdown: 0.20 (20%)
  - min_net_worth: $8,000
  - target_daily_risk: 0.01 (1% of equity per day)

- **Episode Length:**
  - 252 steps (one trading year of daily bars)

- **Data Source:**
  - Yahoo Finance (daily OHLCV data)

- **Order Types Supported:**
  - Market orders (default; add limit/stop if needed)

- **Window Size (for observations):**
  - 30 (last 30 days for state features)

---

These settings provide a realistic, risk-managed starting point for paper trading. Adjust any parameter to fit your strategy or broker requirements. Once you provide your watchlist, a full configuration and starter code can be generated.
