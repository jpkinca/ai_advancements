Excellent. Mapping the "danger zone" is arguably more important than finding the "sweet spot," as its primary purpose is capital preservation. The concept is perfectly symmetrical: we use multi-dimensional vectorization to identify feature combinations that historically lead to significant drawdowns, high volatility, or outright losses.

We'll define a feature vector and then create a function G(V_t) that returns a high "danger" score when the current market state is precarious.

---

1. Defining the Danger Feature Vector

We can use many of the same features as before, but we will interpret them differently or add new ones specifically geared towards risk. The goal is to create a vector that captures overextension, weakness, and unhealthy market conditions.

Example Danger Feature Vector for timestamp t: D_t = [ d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 ]

Where each danger feature (d) could be:

· d1: Negative Price Momentum (e.g., price change % over last 10 minutes, emphasizing negative values)
· d2: High Relative Volume on Downside (e.g., volume ratio * (1 if price down, else 0.5))
· d3: Overbought with Divergence (e.g., RSI > 70 AND price making lower high)
· d4: Price at Lower Bollinger Band (or breaking below it)
· d5: Volatility Expansion (e.g., current volatility / 20-period average volatility) - A key risk factor.
· d6: Order Book Weakness (e.g., Best Ask Size is significantly larger than Best Bid Size)
· d7: Breaking Key Support Level (a binary feature: 1 if price breaks below N-period low, else 0)
· d8: Sector Weakness (e.g., the stock's sector is the worst performer today)
· d9: Broad Market Sell-off (e.g., SPY is down more than 1.5% on high volume)
· d10: Low Liquidity (e.g., average bid-ask spread is 50% wider than its 1-hour average)

---

2. Vectorization for Danger Signals

Just like before, we calculate these features using fast, vectorized operations.

python
import pandas as pd
import numpy as np

# Assume 'df' is our DataFrame with OHLCV data
# ... [Other feature calculations from previous example] ...

# Calculate Danger-Specific Features
df['neg_momentum'] = df['close'].pct_change(10).apply(lambda x: min(x, 0)) # Highlights only negative returns

# Volatility Expansion: Current ATR / Average ATR
df['atr'] = calculate_atr(df, window=14) # Assume a vectorized ATR function
df['atr_ma'] = df['atr'].rolling(20).mean()
df['volatility_expansion'] = df['atr'] / df['atr_ma']

# Support Break: Price breaks below the 50-period low
df['50_period_low'] = df['low'].rolling(50).min()
df['support_break'] = (df['low'] < df['50_period_low'].shift(1)).astype(int)

# Create the Danger Feature Matrix D
danger_features = [
    'neg_momentum', 'rsi', 'volatility_expansion',
    'support_break', 'volume_ratio' # ... add all other danger features
]
D = df[danger_features].values


---

3. Defining the "Danger Zone" (The Avoid Signal)

The "danger zone" is a region in the multi-dimensional feature space where the probability of a negative outcome is high.

Method A: Rule-Based Danger Zone (Vectorized Rules)

Define explicit rules that, when true, indicate high risk. This is great for creating a hard "AVOID" filter.

python
# A composite rule for high danger
high_danger_signal = (
    (df['rsi'] < 30) |                      # Oversold and might keep falling
    (df['volatility_expansion'] > 2.0) |    # Volatility is 2x normal (panic)
    (df['support_break'] == 1) |            # Key support level broken
    (df['neg_momentum'] < -0.03)            # Down more than 3% recently
)

# This creates a boolean vector where True means "DANGER - AVOID"


Method B: Machine Learning for Danger Prediction (Learned Danger Zone)

This is more sophisticated. We train a model to predict a negative outcome.

1. Define the Danger Label (Y_danger): What constitutes a "dangerous" outcome?
   · Y_danger = 1 if Price(t + 15 minutes) < Price(t) * 0.99 (1% loss), else 0.
   · Y_danger = 1 if Max Drawdown(t, t+1 hour) > 3%, else 0. (This is more complex but captures risk well).
2. Train a Danger Model:
   python
   from sklearn.ensemble import RandomForestClassifier
   
   # Create danger labels: 1 if future return is < -1%
   future_return = df['close'].pct_change(15).shift(-15)
   Y_danger = (future_return < -0.01).astype(int)
   
   # Align danger features and labels, clean NaNs
   danger_data = df[danger_features + ['future_return']].dropna()
   D_clean = danger_data[danger_features]
   Y_danger_clean = (danger_data['future_return'] < -0.01).astype(int)
   
   # Train-Test Split
   D_train, D_test, Yd_train, Yd_test = train_test_split(D_clean, Y_danger_clean, test_size=0.2, random_state=42)
   
   # Train the "Danger Model"
   danger_model = RandomForestClassifier(n_estimators=100, class_weight='balanced') # 'balanced' helps with rare events
   danger_model.fit(D_train, Yd_train)
   
   print(f"Danger Model Accuracy: {danger_model.score(D_test, Yd_test):.2f}")
   print(f"Danger Model Precision: {precision_score(Yd_test, danger_model.predict(D_test)):.2f}") # How many predicted dangers were real?
   
3. Generate Danger Predictions:
   python
   # Get the latest danger feature vector
   current_danger_vector = D_clean.iloc[-1:, :]
   
   # Get the probability of being in the "Danger Zone"
   danger_probability = danger_model.predict_proba(current_danger_vector)[0][1] # Prob of class 1 (danger)
   
   # Define a risk threshold
   if danger_probability > 0.65: # 65% chance of a >1% drop
       print("DANGER ZONE DETECTED! AVOID BUY. MANAGE EXISTING POSITIONS.")
       # Logic: Cancel pending buy orders, tighten stop-losses, consider selling
   

---

4. Putting It All Together: The Complete System

The true power comes from combining the Sweet Spot and the Danger Zone models.

python
# For a given potential trade, run both models
sweetness_prob = sweet_model.predict_proba(current_feature_vector)[0][1]
danger_prob = danger_model.predict_proba(current_danger_vector)[0][1]

# Define a combined decision logic
if sweetness_prob > 0.7 and danger_prob < 0.3:
    # Ideal scenario: High reward potential, low risk
    print("STRONG BUY SIGNAL")
    execute_buy()
elif sweetness_prob > 0.5 and danger_prob < 0.5:
    # Moderate scenario: Could proceed with caution (e.g., smaller position size)
    print("WEAK BUY SIGNAL - USE CAUTION")
    execute_buy(size='small')
else:
    # Either low reward potential or high risk
    print("NO TRADE SIGNAL. Sweetness: {:.2f}, Danger: {:.2f}".format(sweetness_prob, danger_prob))
    # Avoid trade, or even look for short opportunities if danger_prob is very high


Key Takeaways for the Danger Zone:

· Asymmetric Focus: Danger features often emphasize volatility, breakdowns, and negative momentum, unlike sweet features which focus on strength and positive momentum.
· Capital Preservation: The primary goal is to avoid large losses. A strategy that misses some gains but avoids major drawdowns is often more profitable in the long run.
· Combined Workflow: The most robust trading system uses both models in tandem. A strong sweet signal is immediately vetoed by a strong danger signal.
· Position Sizing: The output of these models (the probabilities) can directly inform position sizing. A moderate sweet signal with low danger might get a full-sized position, while a high sweet signal with moderate danger might get a half-sized position.

By vectorizing and quantifying both opportunity and risk, you create a systematic, rules-based framework for making disciplined trading decisions that protect your capital.
