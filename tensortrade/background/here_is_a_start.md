Here’s a ready-to-use TensorTrade add-on that boosts outcomes with smarter reward shaping, risk-targeted position sizing, and an early stop guardrail.

What you get

RiskAwareReward — Sharpe-like reward with penalties for drawdown and churn/turnover.

VolatilityTargetedAction — Maps a continuous action 
[
−
1
,
1
]
[−1,1] to a vol-scaled exposure, capping leverage and keeping risk steady.

DrawdownStopper — Ends episodes early if max drawdown or equity floor is breached.

Download the module

Quickstart (pseudocode)
from tensortrade.env.default import create
from tensortrade.oms.exchanges.simulated import SimulatedExchange
from tensortrade.oms.instruments import USD, Instrument
from tensortrade.oms.wallets import Portfolio, Wallet

from tensortrade_risk_module import (
    RiskAwareReward, RiskAwareRewardConfig,
    VolatilityTargetedAction, VolTargetedActionConfig,
    DrawdownStopper, DrawdownStopperConfig
)

# Exchange & portfolio
exchange = SimulatedExchange(base_instrument=USD)
btc = Instrument("BTC", 8, "Bitcoin")
portfolio = Portfolio(USD, [Wallet(exchange, 10000 * USD)])

# Risk-aware reward
reward = RiskAwareReward(RiskAwareRewardConfig(
    window=64, dd_penalty=2.5, turnover_penalty=0.05, risk_free_rate=0.0
))

# Vol-targeted action (continuous a∈[-1,1] → exposure, uses proportion_order)
action = VolatilityTargetedAction(
    portfolio=portfolio,
    config=VolTargetedActionConfig(
        target_daily_risk=0.01, max_leverage=1.0, vol_window=20,
        use_equity_vol=True, main_instrument_symbol="BTC"
    )
)

# Guardrail
stopper = DrawdownStopper(DrawdownStopperConfig(
    max_drawdown=0.20, min_net_worth=8000.0, warmup_steps=10
))

env = create(
    portfolio=portfolio,
    action_scheme=action,
    reward_scheme=reward,
    stopper=stopper,
    window_size=30
)

# Train (example with SB3 PPO)
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)

Why this helps

Outcome-oriented reward: Encourages smooth equity growth (vol-adjusted), discourages deep drawdowns and excessive churn → more stable learning.

Risk parity on autopilot: Position sizing adapts to regime volatility so your policy doesn’t over-risk during turbulence or under-risk in calm markets.

Fail-safe episodes: Early-stop keeps training on-distribution and prevents “death spirals” after big hits.

Tunables that matter

RiskAwareRewardConfig.window — 32–128 is typical (shorter = faster adaptation, noisier σ).

dd_penalty — 1.5–4.0; higher clamps risk harder.

turnover_penalty — 0.02–0.10 to discourage flip-flopping.

VolTargetedActionConfig.target_daily_risk — common picks: 0.5–1.5% per day.

max_leverage — cap exposure (e.g., 1.0 long-only, 2.0 with modest leverage/shorts).

DrawdownStopperConfig.max_drawdown — 0.10–0.25 typical in training.

Notes

The action scheme uses proportion_order. If your TensorTrade version differs, keep the reward + stopper and swap in your preferred ActionScheme.

If your broker/exchange doesn’t support shorts, clamp agent outputs to 
[
0
,
1
]
[0,1] at the policy head.

For multi-asset portfolios, extend VolatilityTargetedAction to route a vector of actions to per-instrument proportions.

Want me to adapt this for your IBKR live flow (post-training) and add PostgreSQL logging + Telegram alerts for stops/breaches? I can bundle a thin inference wrapper around your trained policy next.