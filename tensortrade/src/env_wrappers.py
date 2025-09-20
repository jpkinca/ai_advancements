"""env_wrappers.py
Utility Gym wrappers for the TensorTrade MVP.

Current focus:
  - ActionRewardDBLogger: Logs per-step actions and rewards into the tt_ tables
    using the helpers in `db_utils` (insert_action / insert_reward / create_episode / finalize_episode).

Design notes:
  - Keeps a running `episode_id` in the DB. When env.reset() is called after a
    terminal step, it finalizes the prior episode and creates a new one.
  - Handles both old Gym (obs, reward, done, info) and new Gymnasium style
    (obs, reward, terminated, truncated, info) step signatures by introspection.
  - Extracts scalar actions or action vectors (numpy / list). If vector, maps
    indices to `symbols` list provided at construction.
  - Lightweight and only used when `--log-training` flag is passed to
    `train_mvp.py` (to avoid overhead when not desired).

Usage:
    from env_wrappers import ActionRewardDBLogger
    wrapped_env = ActionRewardDBLogger(env, engine, symbols)

Limitations:
  - Does not currently log observations (tt_observation); can be extended later.
  - Assumes UTC timestamps; for simplicity uses datetime.utcnow().
"""
from __future__ import annotations

from typing import Any, Sequence, Optional
from datetime import datetime

try:
    import gym
except Exception:  # pragma: no cover
    gym = None  # type: ignore

import numpy as np

from db_utils import (
    create_episode,
    finalize_episode,
    insert_action,
    insert_reward,
    get_engine,
    ensure_indexes,
)


class ActionRewardDBLogger:
    """Generic wrapper that intercepts reset/step to persist actions & rewards.

    Parameters
    ----------
    env : gym.Env
        The underlying environment (TensorTrade default env implementing Gym API).
    engine : sqlalchemy.Engine | None
        SQLAlchemy engine. If None, will call get_engine() at init.
    symbols : Sequence[str]
        Ordered list of symbols to map action vector positions to instruments.
    log_every : int (default=1)
        Log every N steps (can be increased to reduce overhead).
    episode_metadata : dict | None
        Reserved for future enrichment (e.g., hyperparams) at episode creation.
    """

    def __init__(
        self,
        env: Any,
        symbols: Sequence[str],
        engine=None,
        log_every: int = 1,
        episode_metadata: Optional[dict] = None,
    ) -> None:
        if gym is not None and not isinstance(env, gym.Env):  # type: ignore
            # We don't hard fail—some TensorTrade envs may not subclass directly.
            pass
        self.env = env
        self.symbols = list(symbols)
        self.engine = engine or get_engine()
        ensure_indexes(self.engine)
        self.log_every = max(1, int(log_every))
        self.episode_metadata = episode_metadata or {}
        self._episode_id: Optional[int] = None
        self._step_in_episode = 0

    # ------------- Internal helpers -------------
    def _start_new_episode(self):
        self._episode_id = create_episode(self.engine, datetime.utcnow())
        self._step_in_episode = 0

    def _finalize_episode(self, stop_reason: str = "episode_complete"):
        if self._episode_id is not None:
            finalize_episode(self.engine, self._episode_id, datetime.utcnow(), stop_reason=stop_reason)
            self._episode_id = None

    def _log(self, action: Any, reward: float):
        if self._episode_id is None:
            return
        ts = datetime.utcnow()
        # Insert reward row
        try:
            insert_reward(self.engine, self._episode_id, ts, float(reward))
        except Exception:
            pass
        # Attempt to parse action vector
        try:
            if isinstance(action, (int, float)):
                arr = np.array([float(action)])
            else:
                arr = np.asarray(action).reshape(-1)
            for i, val in enumerate(arr):
                if i >= len(self.symbols):
                    break
                try:
                    insert_action(self.engine, self._episode_id, ts, self.symbols[i], float(val))
                except Exception:
                    continue
        except Exception:
            # Non-numeric / unsupported action structure.
            pass

    # ------------- Gym API -------------
    def reset(self, *args, **kwargs):  # noqa: D401
        # Finalize previous episode (if any) and start a new one.
        if self._episode_id is not None:
            self._finalize_episode("reset_called")
        self._start_new_episode()
        return self.env.reset(*args, **kwargs)

    def step(self, action):  # noqa: D401
        result = self.env.step(action)
        # Support both API styles
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:  # (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
        self._step_in_episode += 1
        if (self._step_in_episode % self.log_every) == 0:
            try:
                self._log(action, reward)
            except Exception:
                pass
        if terminated or truncated:
            self._finalize_episode("terminated" if terminated else "truncated")
        # Reconstruct original tuple shape
        if len(result) == 4:
            return obs, reward, terminated or truncated, info
        return obs, reward, terminated, truncated, info

    # ------------- Attribute forwarding -------------
    def __getattr__(self, item):  # Forward unknown attrs to underlying env
        return getattr(self.env, item)

    def close(self):  # Graceful close
        try:
            self._finalize_episode("env_closed")
        except Exception:
            pass
        if hasattr(self.env, "close"):
            return self.env.close()

__all__ = ["ActionRewardDBLogger"]
