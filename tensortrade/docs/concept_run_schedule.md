# Concept Run Schedule for TensorTrade RL Trading Module

## 1. Program Run Frequency

- **Backtesting/Training:**
  - Run through historical data in episodes as needed for model training and evaluation.
  - Frequency is determined by the number of training episodes and data size.

- **Live Trading:**
  - Run at every new data interval (e.g., every minute, hour, or day).
  - At each interval, process new market data, make trading decisions, and execute orders.

---

## 2. Recommended Chart Time Intervals

- **Daily Bars (1D):**
  - Suitable for swing trading, long-term strategies, and research.
  - Lower noise, less frequent trading, ideal for RL experiments and proof-of-concept.

- **Hourly Bars (1H):**
  - Good for intraday strategies and more frequent decision-making.

- **Minute Bars (1M):**
  - For high-frequency trading; requires robust infrastructure and data.

---

## 3. Practical Guidance

- Start with daily or hourly intervals for initial development and RL training.
- Choose the interval based on strategy goals and available data.
- For live trading, synchronize the program to run at the chosen interval (e.g., every day at market close for daily bars, every hour for hourly bars).

---

This schedule helps ensure the module runs efficiently and aligns with your trading strategy and data availability.
