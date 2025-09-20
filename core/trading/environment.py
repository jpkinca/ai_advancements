from dataclasses import dataclass
import pandas as pd

@dataclass
class StepResult:
    next_index: int
    reward: float
    done: bool

class SimpleTradingEnv:
    """A minimal non-gym trading env for quick tests."""

    def __init__(self, data: pd.DataFrame, price_col: str = "Close"):
        self.data = data.reset_index(drop=True)
        self.price_col = price_col
        self.i = 1
        self.position = 0  # -1 short, 0 flat, 1 long
        self.entry_price = None

    def reset(self):
        self.i = 1
        self.position = 0
        self.entry_price = None
        return self._obs()

    def _obs(self):
        row = self.data.iloc[self.i]
        return row.to_dict()

    def step(self, action: int) -> StepResult:
        # actions: 0 hold, 1 long, 2 short, 3 close
        price_prev = float(self.data.iloc[self.i - 1][self.price_col])
        price = float(self.data.iloc[self.i][self.price_col])
        reward = 0.0

        if action == 1 and self.position == 0:  # open long
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:  # open short
            self.position = -1
            self.entry_price = price
        elif action == 3 and self.position != 0:  # close
            if self.position == 1:
                reward = (price - self.entry_price) / self.entry_price
            else:
                reward = (self.entry_price - price) / self.entry_price
            self.position = 0
            self.entry_price = None
        else:
            # mark-to-market reward for holding
            if self.position == 1:
                reward = (price - price_prev) / price_prev
            elif self.position == -1:
                reward = (price_prev - price) / price_prev

        self.i += 1
        done = self.i >= len(self.data) - 1
        return StepResult(self.i, reward, done)
