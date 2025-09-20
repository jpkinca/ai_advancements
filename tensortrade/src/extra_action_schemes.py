"""extra_action_schemes.py
Custom lightweight action schemes to expand TensorTrade capabilities for the MVP.

Currently includes:
  - SimpleDiscreteAction: For each instrument choose one of {FLAT, LONG, (SHORT optional)}.

Why:
  The volatility-targeted continuous scheme is powerful but sometimes a simple
  discrete scheme is desirable for quick experimentation, baseline comparison,
  or debugging reward behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Any
import numpy as np

try:  # pragma: no cover - depends on tensortrade
    from tensortrade.env.default.actions import ActionScheme
    from tensortrade.oms.orders import proportion_order
except Exception:  # pragma: no cover
    class ActionScheme:  # type: ignore
        def reset(self): ...
        @property
        def action_space(self): ...
        def get_action(self, action): ...
    def proportion_order(*args, **kwargs):  # type: ignore
        raise RuntimeError("TensorTrade not installed; proportion_order unavailable")

try:  # pragma: no cover
    from gym import spaces
except Exception:  # pragma: no cover
    class spaces:  # type: ignore
        class Discrete:
            def __init__(self, n): self.n = n

@dataclass
class SimpleDiscreteActionConfig:
    instrument_symbols: Optional[List[str]] = None
    allow_shorts: bool = False  # If True: actions = {FLAT, LONG, SHORT}; else {FLAT, LONG}


class SimpleDiscreteAction(ActionScheme):
    """Simple per-instrument discrete action scheme.

    For N instruments, action is an integer in [0, base**N - 1] representing the cartesian product.
    base = 2 (FLAT, LONG) or 3 (FLAT, LONG, SHORT) depending on allow_shorts.

    Decoding strategy:
        action -> base representation -> per instrument directive.

    Each directive maps to target proportion:
        FLAT -> 0
        LONG -> +1/N (equal-weight long exposure)
        SHORT -> -1/N (if allowed)

    This keeps total gross exposure  List[int]:
        instr_count = len(self._instruments)
        digits = []
        x = int(a)
        for _ in range(instr_count):
            digits.append(x % self._base)
            x //= self._base
        while len(digits) < instr_count:
            digits.append(0)
        return digits

    def get_action(self, action: Any):
        try:
            a_int = int(action)
        except Exception:
            a_int = 0
        directives = self._decode(a_int)
        n = len(self._instruments) or 1
        unit = 1.0 / n
        orders = []
        for i, instr in enumerate(self._instruments):
            d = directives[i]
            if d == 0:  # FLAT
                target = 0.0
            elif d == 1:  # LONG
                target = unit
            else:  # SHORT
                target = -unit if self.cfg.allow_shorts else unit
            try:
                o = proportion_order(portfolio=self.portfolio, instrument=instr, proportion=target)
                orders.append(o)
            except Exception:  # pragma: no cover
                continue
        return orders
