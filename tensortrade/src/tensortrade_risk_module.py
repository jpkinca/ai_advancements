
"""
tensortrade_risk_module.py
--------------------------
Drop-in module for TensorTrade to improve trading outcomes via:
1) RiskAwareReward: Sharpe-like, drawdown-penalized reward shaping.
2) VolatilityTargetedAction: Continuous action -> risk-targeted position sizing.
3) DrawdownStopper: Early stop episode on breach of max drawdown or net-worth floor.

This file is designed to be robust to minor TensorTrade API differences across versions.
Where possible, it uses duck-typing and reasonable fallbacks. See usage examples at bottom.

Author: ChatGPT (GPT-5 Thinking)
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Deque, Tuple
from collections import deque
import numpy as np

# --- Optional imports (TensorTrade / Gym). We avoid hard failures if not present. ---
try:
    # Old/Default API paths (pre-1.0 had these namespaces)
    from tensortrade.env.default.rewards import RewardScheme
    from tensortrade.env.default.actions import ActionScheme
    from tensortrade.env.default.stoppers import Stopper
    from tensortrade.oms.orders import Order, proportion_order
    import gym
    from gym import spaces
except Exception:
    # Fallback placeholder interfaces to allow type checking / import in non-TT contexts.
    class RewardScheme:  # type: ignore
        """Fallback RewardScheme stub. TensorTrade will provide the real one at runtime."""
        registered_name = "risk_aware_reward"
        def reset(self): ...
        def get_reward(self, action: Any) -> float: ...

    class ActionScheme:  # type: ignore
        """Fallback ActionScheme stub."""
        registered_name = "volatility_targeted_action"
        @property
        def action_space(self): ...
        def get_action(self, action: Any): ...
        def reset(self): ...

    class Stopper:  # type: ignore
        """Fallback Stopper stub."""
        registered_name = "drawdown_stopper"
        def reset(self): ...
        def stop(self, env: Any) -> bool: ...

    class Order:  # type: ignore
        pass

    def proportion_order(portfolio, instrument, proportion) -> Order:  # type: ignore
        raise RuntimeError("proportion_order is not available without TensorTrade installed.")

    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except Exception:
        class spaces:  # type: ignore
            class Box:
                def __init__(self, low, high, shape=None, dtype=None): ...
        class gym:  # type: ignore
            spaces = spaces


# -------------------- Utility helpers --------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _env_net_worth(env: Any) -> float:
    """
    Attempt to obtain the current net worth from env/portfolio,
    compatible with several TensorTrade versions.
    """
    # Common access patterns
    for attr in ["net_worth", "portfolio_value", "equity"]:
        if hasattr(env, attr):
            return _safe_float(getattr(env, attr))

    # Default TT: env has a portfolio with net_worth
    if hasattr(env, "portfolio"):
        pf = getattr(env, "portfolio")
        for attr in ["net_worth", "performance", "equity"]:
            if hasattr(pf, attr):
                v = getattr(pf, attr)
                if callable(v):
                    try:
                        return _safe_float(v())
                    except Exception:
                        pass
                else:
                    # e.g., portfolio.net_worth
                    return _safe_float(v)

    # RewardScheme usually has _portfolio in TT default envs
    if hasattr(env, "_portfolio"):
        pf = getattr(env, "_portfolio")
        if hasattr(pf, "net_worth"):
            return _safe_float(getattr(pf, "net_worth"))

    return 0.0


def _env_price(env: Any) -> Optional[float]:
    """
    Try to infer a latest traded price for the main instrument.
    Used by the action scheme for ATR/vol calculations if available.
    """
    # Many TT examples keep price streams inside observation state rather than env.
    # We return None if we cannot find it. The action scheme gracefully degrades.
    for attr in ["price", "last_price", "close"]:
        if hasattr(env, attr):
            return _safe_float(getattr(env, attr), default=None)  # type: ignore
    # Try portfolio/exchange route
    try:
        ex = getattr(env, "exchange", None) or getattr(env, "_exchange", None)
        if ex is not None and hasattr(ex, "price"):
            return _safe_float(ex.price, default=None)  # type: ignore
    except Exception:
        pass
    return None


# -------------------- Risk-aware Reward --------------------

@dataclass
class RiskAwareRewardConfig:
    window: int = 64                  # number of steps for rolling volatility / turnover
    risk_free_rate: float = 0.0       # daily risk-free rate for excess return (if known)
    dd_penalty: float = 2.5           # penalty multiplier for current drawdown (0..1)
    turnover_penalty: float = 0.05    # penalty per unit turnover in window
    clip_reward: Optional[float] = 5.0  # clip absolute reward to reduce outliers
    epsilon: float = 1e-9


class RiskAwareReward(RewardScheme):
    """
    Reward = (Δequity - rf) / (σ_equity + ε)  -  λ_dd * drawdown  -  λ_to * turnover

    Intuition:
    - Incentivizes smoother, volatility-adjusted equity growth (Sharpe-like).
    - Penalizes being in drawdown and churning positions (turnover penalty).
    - Stable across different instruments and episode lengths.
    """

    registered_name = "risk_aware_reward"

    def __init__(self, config: Optional[RiskAwareRewardConfig] = None):
        super().__init__()
        self.cfg = config or RiskAwareRewardConfig()
        self._returns: Deque[float] = deque(maxlen=self.cfg.window)
        self._actions: Deque[float] = deque(maxlen=self.cfg.window)
        self._last_equity: Optional[float] = None
        self._peak_equity: float = 0.0
        self._prev_action: float = 0.0

    # TensorTrade calls reset() at the start of each episode.
    def reset(self) -> None:
        self._returns.clear()
        self._actions.clear()
        self._last_equity = None
        self._peak_equity = 0.0
        self._prev_action = 0.0

    def _compute_reward(
        self,
        equity_now: float,
        action_scalar: Optional[float] = None
    ) -> float:
        cfg = self.cfg

        if self._last_equity is None or self._last_equity  1 else 0.0

        # Update drawdown
        self._peak_equity = max(self._peak_equity, equity_now)
        drawdown = 0.0
        if self._peak_equity > 0:
            drawdown = max(0.0, 1.0 - equity_now / self._peak_equity)

        # Turnover via action-change proxy
        turnover = 0.0
        if action_scalar is not None:
            a = float(action_scalar)
            turnover = abs(a - self._prev_action)
            self._prev_action = a
            self._actions.append(a)

        # Compose reward
        risk_adj = excess_ret / (vol + cfg.epsilon)
        reward = risk_adj - cfg.dd_penalty * drawdown - cfg.turnover_penalty * turnover

        if cfg.clip_reward is not None:
            reward = float(np.clip(reward, -cfg.clip_reward, cfg.clip_reward))

        self._last_equity = equity_now
        return float(reward)

    def get_reward(self, action: Any) -> float:
        """
        TT typically passes the raw agent action here.
        We attempt to coerce it to a scalar for turnover computation when possible.
        """
        # Extract scalar from various action types
        action_scalar: Optional[float] = None
        try:
            if isinstance(action, (int, float)):
                action_scalar = float(action)
            elif hasattr(action, "item"):
                action_scalar = float(action.item())
            elif isinstance(action, (list, tuple)) and len(action) > 0:
                action_scalar = float(action[0])
            elif hasattr(action, "__array__"):
                arr = np.asarray(action)
                if arr.size > 0:
                    action_scalar = float(arr.flat[0])
        except Exception:
            action_scalar = None

        # Get env or portfolio to compute equity
        env = getattr(self, "env", None) or getattr(self, "_env", None)
        if env is None:
            # Some TT versions set _portfolio on the RewardScheme
            env = getattr(self, "_portfolio", None)
        equity_now = 0.0
        try:
            equity_now = _env_net_worth(env) if env is not None else 0.0
        except Exception:
            equity_now = 0.0

        return self._compute_reward(equity_now=equity_now, action_scalar=action_scalar)


# -------------------- Volatility-targeted Action Scheme --------------------

@dataclass
class VolTargetedActionConfig:
    target_daily_risk: float = 0.01    # Target daily equity volatility (1% default)
    max_leverage: float = 1.0          # Clamp exposure in [-max, max]
    vol_window: int = 20               # Rolling window for volatility estimate (equity or price proxy)
    use_equity_vol: bool = True        # If True, estimate σ from equity returns; else fallback to price returns if available.
    epsilon: float = 1e-9
    instrument_symbols: Optional[list[str]] = None  # List of instrument symbols for multi-asset support. If None, auto-detect.


class VolatilityTargetedAction(ActionScheme):
    """
    Multi-asset version: Each action a_i ∈ [-1, 1] is mapped to a target exposure e_i ∈ [-L, L] for each instrument.
    e_i = a_i * min(L, target_daily_risk / (σ_i + ε))
    σ_i is estimated rolling volatility of equity (preferred) or price for each instrument.

    Orders are placed as *proportion_order* for each instrument, rebalancing to the new target exposures.
    """

    registered_name = "volatility_targeted_action"

    def __init__(self, portfolio, config: Optional[VolTargetedActionConfig] = None):
        super().__init__()
        self.portfolio = portfolio
        self.cfg = config or VolTargetedActionConfig()

        # Detect instruments
        base = getattr(portfolio, "base_instrument", None) or getattr(portfolio, "base_symbol", None)
        holdings = getattr(portfolio, "holdings", [])
        if self.cfg.instrument_symbols is not None:
            self._instruments = [h.instrument for h in holdings if getattr(h, "instrument", None) and str(h.instrument.symbol) in self.cfg.instrument_symbols]
        else:
            self._instruments = [h.instrument for h in holdings if getattr(h, "instrument", None) and (base is None or str(h.instrument) != str(base))]

        self._equity_hists = {instr.symbol: deque(maxlen=self.cfg.vol_window + 1) for instr in self._instruments}
        self._price_hists = {instr.symbol: deque(maxlen=self.cfg.vol_window + 1) for instr in self._instruments}

        # Action space is vector in [-1, 1] for each instrument
        n = len(self._instruments)
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        for hist in self._equity_hists.values():
            hist.clear()
        for hist in self._price_hists.values():
            hist.clear()

    def get_action(self, action: Any):
        # Convert action to vector in [-1, 1]
        try:
            arr = np.asarray(action, dtype=float).reshape(-1)
            acts = np.clip(arr, -1.0, 1.0)
        except Exception:
            acts = np.zeros(len(self._instruments))

        env = getattr(self, "env", None) or getattr(self, "_env", None)

        orders = []
        for i, instr in enumerate(self._instruments):
            # For each instrument, estimate volatility
            eq_hist = self._equity_hists[instr.symbol]
            pr_hist = self._price_hists[instr.symbol]

            # Try to get instrument-specific equity and price
            equity_now = None
            price_now = None
            # If env provides per-instrument equity, use it; else fallback to portfolio net worth
            if hasattr(env, "get_instrument_equity"):
                try:
                    equity_now = env.get_instrument_equity(instr)
                except Exception:
                    equity_now = None
            if equity_now is None:
                equity_now = _env_net_worth(env)
            # Try to get instrument price
            if hasattr(env, "get_instrument_price"):
                try:
                    price_now = env.get_instrument_price(instr)
                except Exception:
                    price_now = None
            if price_now is None:
                price_now = _env_price(env)

            if equity_now is not None and equity_now > 0:
                eq_hist.append(float(equity_now))
            if price_now is not None and price_now > 0:
                pr_hist.append(float(price_now))

            # Compute σ
            sigma = None
            if self.cfg.use_equity_vol and len(eq_hist) >= 2:
                eq = np.asarray(eq_hist, dtype=float)
                rets = np.diff(eq) / (eq[:-1] + self.cfg.epsilon)
                if len(rets) > 1:
                    sigma = float(np.std(rets))
            if sigma is None and len(pr_hist) >= 2:
                pr = np.asarray(pr_hist, dtype=float)
                rets = np.diff(pr) / (pr[:-1] + self.cfg.epsilon)
                if len(rets) > 1:
                    sigma = float(np.std(rets))
            if sigma is None:
                sigma = 0.0

            # Target exposure based on volatility targeting
            max_e = float(self.cfg.max_leverage)
            if sigma  bool:
        self._steps += 1
        if self._steps  0:
            dd = max(0.0, 1.0 - equity / self._peak_equity)
        if dd >= self.cfg.max_drawdown:
            return True

        # Check absolute floor
        if self.cfg.min_net_worth is not None and equity = 0 at the policy level or clamp agent outputs.
# - For multi-instrument portfolios, extend VolatilityTargetedAction to route actions to multiple instruments.
"""
