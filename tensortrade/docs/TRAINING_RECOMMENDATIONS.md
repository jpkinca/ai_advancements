# Training Recommendations (Current MVP)

## Summary of Current Training Capability

You can ingest daily OHLCV via IBKR, persist to PostgreSQL, build a multi‑asset TensorTrade environment, and train a PPO policy using either:

- Volatility‑targeted continuous weight allocation, or
- Simple discrete (flat/long[/short]) allocation scheme.

Risk controls and logging supported:

- RiskAwareReward + optional alternative returns reward.
- DrawdownStopper for max drawdown / net worth floor.
- Optional per‑step DB logging wrapper (actions & rewards).
- Persistence for prices, features, signal scores, target weights, equity curve, episode metrics (Sharpe proxy, max drawdown, turnover, config).

Available engineered features (if enabled):

- 1‑period returns, rolling volatility (5/10/20), normalized volume per symbol.

What training currently achieves:

- Learns coarse exposure modulation across supplied symbols based on short‑horizon return/vol patterns.
- Demonstrates volatility targeting and drawdown protection behaviors.
- Produces a PPO policy that may modestly generalize to immediately subsequent data.

Key limitations:

- Sparse feature space (no factor library, regimes, macro, or cross‑sectional rankings yet).
- Daily bars only → few temporal steps per episode with short history windows.
- No explicit transaction/slippage costs (turnover unrealistically unconstrained).
- Simple reward (risk aware pseudo‑Sharpe) without multi-objective extensions.
- No separate train/validation temporal split embedded in pipeline.

## Assessment of "50 Stocks over 3 Months (Daily)"

- ≈ 60–65 trading days → with window_size=30 leaves ~30–35 effective steps per episode.
- PPO 10k–20k steps would recycle the same short history many times → high overfit risk.
- More symbols increase action/state dimensionality without increasing temporal depth.
- Result would serve only as a functional smoke test, not a meaningful strategy evaluation.

## Recommended First Robust Run

- Symbols: 10–15 (limit complexity while multi‑asset behavior is validated).
- History: 12 months daily (≈ 250 trading days) → ~220 effective steps (after 30‑day window).
- Window size: 30 (enough lookback for short‑term vol / return stats; keep state compact).
- PPO steps: 20k (gives ~90 policy updates if using default batch sizes; adjust if convergence stalls).
- Action scheme: volatility‑targeted (captures risk scaling early).
- Features: enable engineered features flag (`--with-features`).
- Evaluation: at least 1 evaluation episode post‑training; persist metrics.

## Alternative Paths (When Extending)

1. Intraday Expansion: Move to 60‑minute bars for 3–6 months → 400–800 steps per episode (requires intraday fetch support addition).
2. Feature Enrichment: Add cross‑sectional ranks (momentum, volatility, volume), rolling z‑scores, sector/industry dummies.
3. Cost Modeling: Introduce transaction cost penalty (e.g., proportional to turnover) inside reward or subtract from equity.
4. Data Split: Time‑based walk‑forward (train on first 9 months, evaluate on last 3 months) to reduce look‑ahead bias.
5. Hyperparameter Tuning: Systematic grid for PPO (learning rate, batch size, ent coef) once richer feature set exists.

## Quick Command Template

```bash
python -m train_mvp --months 12 --limit 15 --steps 20000 --window 30 \
  --with-features --action-scheme vol_target --reward-scheme risk_aware \
  --risk-target 0.01 --max-leverage 1.0 --max-dd 0.2 --eval-episodes 1
```

(Add `--log-training` if you want per-step action/reward persistence during training.)

## Interpreting Initial Results

- Sharpe proxy near 0 after first run is expected; focus on turnover, max drawdown adherence, and stability of equity curve path.
- If turnover is extreme (>0.2 average absolute action delta per step), consider entropy reduction or explicit turnover penalty.
- If max drawdown approaches threshold quickly, reduce window size or target risk.

## Next Immediate Enhancements (High Impact, Low Complexity)

1. Persist transaction cost assumptions and apply in reward.
2. Add rank-based momentum & volatility decile features.
3. Add simple signal combining into `tt_signal_scores` to seed target weight generation (already tabled).

---

Generated on: 2025-08-17
