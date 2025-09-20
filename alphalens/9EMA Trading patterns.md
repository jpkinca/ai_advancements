Based on a review of established trading strategies, here are two practical algorithms that incorporate a 1-minute chart, the 9-period Exponential Moving Average (EMA), and candlestick patterns for confirmation. These are geared toward scalping in high-liquidity markets like forex (e.g., EUR/USD) or stocks, where quick entries and exits capitalize on short-term momentum. I’ll outline the setup, rules, and logic for each, along with key concerns. These aren’t foolproof—backtest them rigorously on historical data and demo accounts before live use, as scalping demands low spreads, fast execution, and emotional control to avoid overtrading.
1. 9 EMA Pullback Scalping Algorithm
This strategy focuses on trend continuation after a pullback to the 9 EMA, using candlesticks for entry confirmation. It’s simple, responsive, and suits volatile sessions like London or New York opens.
Setup:
	•	Timeframe: 1-minute chart.
	•	Indicators: Apply only the 9 EMA (close price basis).
	•	Additional: Mark key support/resistance levels (e.g., previous day’s high/low) and optionally add volume for confirmation.
	•	Market: High-liquidity pairs/assets to minimize slippage.
Entry Rules (Long/Buy):
	•	Price must be in an uptrend (trading above the 9 EMA, with recent candles closing above it).
	•	Wait for a pullback where price touches or retraces to the 9 EMA.
	•	Enter long on the close of a bullish candlestick pattern at the 9 EMA, such as:
	◦	Bullish Engulfing (small bearish candle followed by larger bullish one, showing reversal strength).
	◦	Pin Bar (long lower wick rejecting lower prices).
	◦	Doji (indecision resolving upward, if followed by a green candle).
	•	Filter: Avoid if overall higher timeframe (e.g., 5-min) shows downtrend.
Entry Rules (Short/Sell):
	•	Price in downtrend (below 9 EMA, candles closing below).
	•	Pullback/retracement up to the 9 EMA.
	•	Enter short on close of bearish pattern like Bearish Engulfing, Pin Bar (long upper wick), or Doji resolving downward.
Exit Rules:
	•	Take Profit: 5-10 pips (or 1:1 to 1:2 risk-reward ratio based on ATR); trail stop to lock profits if momentum builds.
	•	Stop Loss: Place 5-8 pips beyond the recent swing high/low or the candlestick pattern’s extreme.
	•	Alternative Exit: Close on reversal signs, like opposite candlestick pattern or price crossing back through 9 EMA.
Performance Notes: This leverages the 9 EMA’s sensitivity for early trend detection, with candlesticks reducing false signals. In backtests on forex pairs, win rates can hit 55-65% with tight risk management, but commissions eat into profits on low-vol days. 6 7
2. 9 EMA Breakout-Retracement Algorithm
This builds on breakouts from key levels, using the 9 EMA as a dynamic support/resistance filter and engulfing patterns for precise entries. It’s aggressive for momentum plays but requires strict time filters to avoid chop.
Setup:
	•	Timeframe: 1-minute chart.
	•	Indicators: 9 EMA only.
	•	Additional: Plot previous session’s high/low as breakout levels; consider trading only during high-volume hours (e.g., 10-11 AM EST).
Entry Rules (Long/Buy):
	•	Identify a breakout above previous high, confirming uptrend start.
	•	Wait for retracement back to the 9 EMA.
	•	Enter long on the first bullish engulfing candlestick (larger green body fully covering prior red candle) touching the 9 EMA.
	•	Filter: Volume spike on breakout; skip if no clear trend on 5-min chart.
Entry Rules (Short/Sell):
	•	Breakout below previous low for downtrend.
	•	Retracement up to 9 EMA.
	•	Enter short on bearish engulfing candlestick at the 9 EMA.
Exit Rules:
	•	Take Profit: At next resistance/support or when a Doji forms after 10-20 pips movement (signals exhaustion).
	•	Stop Loss: Above/below the engulfing candle’s high/low (typically 5-10 pips risk).
	•	Alternative: Use Fibonacci extensions for targets or exit on 9 EMA cross in opposite direction.
Performance Notes: The 9 EMA acts as a “bounce zone” post-breakout, with engulfing patterns providing high-probability setups. Historical tests show it performs well in trending markets (e.g., 60%+ accuracy on EUR/USD), but false breakouts are a risk—leading to whipsaws in ranging conditions. 7
Concerns and Advice: Both algorithms are high-frequency, so transaction costs can erode edges; aim for brokers with <1 pip spreads. Risk no more than 0.5-1% per trade to survive drawdowns. Candlestick patterns add subjectivity—combine with objective filters like RSI (>50 for longs) to improve robustness. If you’re automating this (e.g., in Python/TradingView), test for overfitting. Scalping isn’t for everyone; if slippage or stress is an issue, scale up to 5-min charts for similar logic with less noise. Let me know if you need code examples or tweaks for specific assets.

To identify stocks suitable for these 1-minute scalping algorithms (using 9 EMA and candlestick patterns) from a pool of 1000 or more, prioritize those with high liquidity to ensure tight spreads and minimal slippage, sufficient intraday volatility for meaningful moves, and a tendency toward clean trends rather than chop. These algos thrive on quick, high-probability setups, so illiquid or range-bound stocks will lead to false signals, overtrading, and eroded edges. Not every stock will fit—expect to narrow down to 10-50 candidates daily, then manually review or backtest. Here’s a practical, step-by-step method to screen efficiently, based on established day trading criteria.
Step 1: Use a Reliable Stock Screener
Leverage free or low-cost tools like Finviz, TradingView, Yahoo Finance, or StockFetcher to filter en masse. These handle 1000+ stocks easily via presets or custom scans. If you’re on a platform like Thinkorswim (TD Ameritrade) or Interactive Brokers, their built-in scanners are even better for real-time data. Avoid manual sifting—it’s inefficient and error-prone.
	•	Why these tools? They pull from major exchanges (NYSE, NASDAQ) and allow exports for further analysis in Excel or Python. 15 19 
	•	Pro tip: Run scans pre-market (e.g., 8-9 AM EST) for that day’s candidates, focusing on US stocks to align with high-volume sessions.
Step 2: Apply Key Screening Criteria
Set these filters to target stocks optimized for 1-min scalping. Adjust based on your risk tolerance and market conditions (e.g., tighten volume in low-vol environments). Aim for a balance: too volatile risks whipsaws; too stable lacks opportunities. 10 16 18
	•	Liquidity (Top Priority): Average daily volume > 1-2 million shares. This ensures you can enter/exit without moving the price—critical for scalping’s tight stops (5-10 pips/points).
	•	Volatility: Average True Range (ATR, 14-period) > $0.50-$1.00, or >2-3% daily range. Beta >1.0 for market correlation. Avoid extremes (>5% unless you’re experienced, as it amplifies losses). 14 
	•	Price Range: $10-$150/share. Steer clear of pennies (<$5) for manipulation risks, or ultra-high (> $200) for affordability in position sizing.
	•	Float and Market Cap: Float <100M for added volatility (e.g., small-caps), but market cap >$2B for stability. Low float can spike moves but increases gaps/slippages. 17 
	•	Relative Volume (RVOL): >1.5x average (unusual volume) for that day—signals momentum for EMA pullbacks/breakouts.
	•	Sector/Other Filters: Tech, biotech, energy, or consumer discretionary for inherent volatility. Add news/gap filters (e.g., >2% pre-market gap) during earnings seasons. Exclude ETFs unless highly liquid like SPY/QQQ.
	•	Trend Bias: Optionally, screen for stocks above/below 50-day SMA to favor up/down trends, aligning with the algos’ EMA logic.
Example Finviz Scan Setup:
	•	Descriptive: Average Volume >1M
	•	Technical: ATR >0.5, Beta >1
	•	Fundamental: Market Cap >$2B, Float <100M
	•	Performance: Change from Open >2% (for intraday momentum) This might yield 20-50 hits from 1000+; sort by volatility descending. 12 19 
Step 3: Validate and Refine Candidates
	•	Manual Review: From the shortlist, check 1-min charts for recent sessions. Look for clear EMA bounces and candlestick reliability (e.g., frequent engulfings without excessive wicks). Use TradingView’s replay mode to simulate.
	•	Backtest the Algos: Apply your strategies to 1-3 months of historical 1-min data via tools like Backtrader (Python) or TradingView’s strategy tester. Target >55% win rate and >1.5 reward:risk.
	•	Automate if Scaling: If you have a CSV of 1000+ stocks, script a screener in Python using pandas and yfinance (for data pulls—note: requires API access). Example logic: Fetch ATR/volume, filter thresholds, output top 20. I can provide sample code if you share your setup.
	•	Monitor Real-Time: Use alerts for breakouts or volume surges during the day.
Concerns and Viewpoints
This method works well but isn’t foolproof—markets change, and screened stocks can turn choppy post-news (e.g., FOMC days). Commissions/spreads will kill profitability on low-vol picks, so factor in your broker’s costs (aim for <$0.005/share). If your pool includes illiquids, you’ll waste time; focus on S&P 500/ Nasdaq 100 subsets for quality. Over-reliance on screens can miss nuances like sector rotations—always cross-check with broader market trends. If you’re new, start with 5-10 stocks you know well (e.g., AAPL, TSLA) before expanding. What’s your stock pool source or platform? I can tweak this for specifics.

Yes, the algorithms I outlined—and most established 9 EMA scalping strategies—do factor in the distance between the close (or low/high of the candle) and the 9 EMA, but implicitly rather than with rigid pip measurements. This ensures you’re catching “tight” pullbacks or retests for better risk-reward, avoiding entries where price is drifting too far and momentum fades. Here’s the breakdown, pulled from proven setups:
In the Pullback Scalping Algorithm
	•	It requires the candle to touch or dip slightly below/above the 9 EMA during the retracement (e.g., low of a bullish pin bar grazing the EMA line for longs). If the close is more than a “slight” deviation—say, 5-10 pips away on a 1-min forex chart without rejection—it signals a weak setup, and you’d skip to prevent chasing. 11 This proximity acts as a filter for high-probability bounces, keeping stops ultra-tight (e.g., 5 pips beyond the candle extreme).