# TensorTrade Unified Risk Module: Outputs & Usage

## Outputs

1. **Orders (Actions):**
   - For each step, the module outputs a list of `Order` objects, one per stock, specifying the target portfolio proportion for each instrument.
   - These orders are executed by the environment to rebalance the portfolio according to the agent’s actions.

2. **Reward Values:**
   - After each step, the module computes a risk-aware reward value, which is used by the RL agent to learn and optimize its trading strategy.
   - The reward reflects volatility-adjusted returns, penalizes drawdown, and discourages excessive turnover.

3. **Episode Stop Signal:**
   - The module outputs a boolean signal (via the stopper) indicating whether the episode should end early due to excessive drawdown or net worth breach.

---

## What Do You Do With These Outputs?

- **Orders:**  
  The environment executes these orders, updating the portfolio holdings (buying/selling stocks) based on the agent’s decisions.

- **Reward Values:**  
  The RL agent (e.g., PPO from Stable Baselines3) uses these rewards to update its policy, learning to maximize risk-adjusted returns while managing drawdown and turnover.

- **Episode Stop Signal:**  
  If triggered, the environment ends the current episode, resets the state, and starts a new training or evaluation run.

---

## Typical Workflow

1. **Training:**  
   - The RL agent interacts with the environment, receives observations, outputs actions (orders), and gets rewards.
   - The agent learns to optimize its trading strategy over many episodes.

2. **Backtesting/Evaluation:**  
   - After training, you run the agent on historical data to evaluate performance (equity curve, drawdown, Sharpe ratio, etc.).

3. **Live Trading (Optional):**  
   - With further integration, the module’s outputs can be used to place real trades via a broker API.

---

## Visualization & Analysis

- Use the environment’s logs and outputs to plot:
  - Portfolio equity curve
  - Drawdown over time
  - Position sizes per stock
  - Performance metrics (Sharpe, Sortino, turnover)

---

In summary:  
You use the module’s outputs to train, evaluate, and (optionally) deploy RL-based trading strategies, with built-in risk management and multi-stock support.
