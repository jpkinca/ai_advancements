# Sweet Spot & Danger Zone Algorithm - Data Requirements

## 📊 Data Consumption Requirements

### **Minimum Data Volume**
- **100 bars minimum** - The algorithm requires at least 100 data points to train the ML models
- **Up to 500 bars loaded** - The system attempts to load up to 500 historical bars for analysis

### **Time Interval**
- **Weekly bars (1week)** - The algorithm primarily uses weekly timeframe data
- **Timeframe preference order**: `['1week', '1month', '1min', '5min']` (weekly is preferred when available)

### **Trading Days Coverage**
- **~3 years of data** - Successful stocks had 157 weekly bars
- **Date range**: September 2022 to September 2025 (current date)
- **157 weeks** ≈ **3 years** of weekly trading data

### **Data Fields Required**
The algorithm consumes OHLCV data:
- **Open, High, Low, Close prices**
- **Volume** (for volume-based features)
- **Timestamps** (for time-based features)

### **Real-World Example**
From the successful analysis:
- **Stocks analyzed**: ACM, ADSK, AEIS, AEM, AFRM
- **Each had 157 weekly bars** covering September 2022 - September 2025
- **Failed stocks**: AAPL (62 bars), NVDA (63 bars), PLTR (1 bar) - insufficient data

### **Feature Engineering Windows**
The algorithm creates features using various rolling windows:
- RSI: 14 periods
- Bollinger Bands: 20 periods
- Volume ratios: 5 and 20 periods
- Volatility: 10 and 20 periods
- Trend strength: 20 and 50 periods

### **Summary**
The algorithm consumes **weekly OHLCV data** for **minimum 100 weeks (2 years)**, **preferably 157+ weeks (3+ years)** to achieve optimal ML model training and reliable signal generation.