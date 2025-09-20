import pandas as pd
from core.data.preprocessing import add_basic_indicators
from core.trading.environment import SimpleTradingEnv

# Use random walk data for a quick smoke test
import numpy as np
np.random.seed(0)
price = 100 + np.cumsum(np.random.randn(300))
df = pd.DataFrame({"Close": price})
df = add_basic_indicators(df)

env = SimpleTradingEnv(df)
obs = env.reset()
ret = []
done = False
while not done:
    action = 0  # hold strategy as a baseline
    step = env.step(action)
    ret.append(step.reward)
    done = step.done

print(f"Steps: {len(ret)}, mean daily return: {np.mean(ret):.6f}")
