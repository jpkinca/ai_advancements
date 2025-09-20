# TensorTrade RL Trading Data Model

All tables are prefixed with `tt_` for clarity and organization.

---

## 1. tt_prices
Stores historical price data for each instrument.

| Column           | Type        | Description                       |
|------------------|------------|-----------------------------------|
| id               | INTEGER PK | Unique row ID                     |
| instrument       | TEXT       | Stock symbol (e.g., TSLA)         |
| timestamp        | DATETIME   | Time of price record              |
| open             | FLOAT      | Open price                        |
| high             | FLOAT      | High price                        |
| low              | FLOAT      | Low price                         |
| close            | FLOAT      | Close price                       |
| volume           | FLOAT      | Trading volume                    |

---

## 2. tt_portfolio
Stores portfolio state at each timestep.

| Column           | Type        | Description                       |
|------------------|------------|-----------------------------------|
| id               | INTEGER PK | Unique row ID                     |
| episode_id       | INTEGER FK | Reference to tt_episode           |
| timestamp        | DATETIME   | Time of record                    |
| net_worth        | FLOAT      | Portfolio net worth               |
| cash_balance     | FLOAT      | Cash in portfolio                 |

---

## 3. tt_holdings
Stores holdings per instrument at each timestep.

| Column           | Type        | Description                       |
|------------------|------------|-----------------------------------|
| id               | INTEGER PK | Unique row ID                     |
| portfolio_id     | INTEGER FK | Reference to tt_portfolio         |
| instrument       | TEXT       | Stock symbol                      |
| quantity         | FLOAT      | Number of shares held             |

---

## 4. tt_action
Stores agent actions at each timestep.

| Column           | Type        | Description                       |
|------------------|------------|-----------------------------------|
| id               | INTEGER PK | Unique row ID                     |
| episode_id       | INTEGER FK | Reference to tt_episode           |
| timestamp        | DATETIME   | Time of action                    |
| instrument       | TEXT       | Stock symbol                      |
| action_value     | FLOAT      | Action taken (proportion/target)  |

---

## 5. tt_order
Stores executed orders.

| Column           | Type        | Description                       |
|------------------|------------|-----------------------------------|
| id               | INTEGER PK | Unique row ID                     |
| episode_id       | INTEGER FK | Reference to tt_episode           |
| timestamp        | DATETIME   | Time of order                     |
| instrument       | TEXT       | Stock symbol                      |
| order_type       | TEXT       | Type (buy/sell/hold)              |
| size             | FLOAT      | Order size                        |
| price            | FLOAT      | Execution price                   |

---

## 6. tt_reward
Stores reward values at each timestep.

| Column           | Type        | Description                       |
|------------------|------------|-----------------------------------|
| id               | INTEGER PK | Unique row ID                     |
| episode_id       | INTEGER FK | Reference to tt_episode           |
| timestamp        | DATETIME   | Time of reward                    |
| reward_value     | FLOAT      | Computed reward                   |

---

## 7. tt_episode
Stores metadata for each episode.

| Column           | Type        | Description                       |
|------------------|------------|-----------------------------------|
| id               | INTEGER PK | Unique episode ID                 |
| start_time       | DATETIME   | Episode start time                |
| end_time         | DATETIME   | Episode end time                  |
| stop_reason      | TEXT       | Reason for episode stop           |
| max_drawdown     | FLOAT      | Max drawdown in episode           |
| sharpe_ratio     | FLOAT      | Sharpe ratio for episode          |
| turnover         | FLOAT      | Turnover for episode              |

---

## 8. tt_observation
Stores agent observations (features) at each timestep.

| Column           | Type        | Description                       |
|------------------|------------|-----------------------------------|
| id               | INTEGER PK | Unique row ID                     |
| episode_id       | INTEGER FK | Reference to tt_episode           |
| timestamp        | DATETIME   | Time of observation               |
| features         | TEXT/JSON  | Serialized feature vector         |

---

This schema supports efficient training, analysis, and experiment tracking for RL trading strategies. You can implement it in SQLite, PostgreSQL, or any relational database.
