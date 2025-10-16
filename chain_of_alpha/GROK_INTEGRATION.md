# Chain-of-Alpha MVP - Grok API Integration

## 🚀 Real LLM Alpha Generation is Ready!

Your Chain-of-Alpha MVP now supports **Grok API integration** for real AI-driven alpha factor discovery. This upgrade transforms the system from proof-of-concept to genuine alpha generation engine.

## ⭐ What's New

### 🧠 **Enhanced LLM Integration**
- **Grok API Support**: Finance-aware reasoning with xAI's Grok model
- **Structured Prompting**: JSON schema enforcement for consistent factor output  
- **Market Context**: Auto-detection of market regimes and economic conditions
- **Category Diversity**: Ensures factors span momentum, mean-reversion, volatility, etc.

### 📊 **Advanced Factor Generation**
- **Economic Rationale**: Each factor includes behavioral finance reasoning
- **Market Neutrality**: Beta-neutral factor design principles  
- **Originality Checking**: Avoids duplicate or highly correlated factors
- **Confidence Scoring**: LLM provides confidence estimates for each factor

### 🔧 **Production Features**
- **Environment Variable Support**: Secure API key handling
- **Error Recovery**: Robust fallback mechanisms
- **Validation Pipeline**: Automatic syntax and logic checking
- **Performance Tracking**: Real-time factor performance monitoring

## 🏃‍♂️ Quick Start (5 minutes)

### 1. Get Grok API Access
```bash
# Go to https://console.x.ai/
# Sign in with X account
# Create API key
# Copy the key for next step
```

### 2. Setup API Key
```bash
python setup_grok_api.py
# Follow prompts to enter your API key
# Script will test connection automatically
```

### 3. Validate Integration  
```bash
python validate_factor_generation.py
# Tests structured factor generation
# Validates 3 sample factors
# Confirms syntax and logic
```

### 4. Run Full Pipeline
```bash
python chain_of_alpha_mvp.py
# Generates 10+ real alpha factors
# Backtests with VectorBT  
# Exports results to results/ directory
```

## 📈 Expected Performance Improvements

### With Grok API vs Mock LLM:
- **Factor Novelty**: 5-10x more creative and non-obvious relationships
- **Economic Validity**: Real behavioral finance insights vs random expressions  
- **Performance Potential**: Target Sharpe >0.8 vs baseline 0.3-0.5
- **Diversification**: Better factor category coverage and correlation structure

### Validation Targets:
- ✅ **Factor Generation**: >80% success rate (vs 100% mock but meaningless)
- ✅ **Syntax Validation**: >95% executable expressions
- ✅ **Performance**: Average Sharpe >0.5, best factor >0.8
- ✅ **Originality**: <0.7 correlation between generated factors

## 🔧 Configuration Options

### Environment Variables
```bash
# Set in .env file or system environment
GROK_API_KEY=your_grok_api_key_here
```

### Config.py Settings
```python
LLM_CONFIG = {
    'llm_model': 'grok',          # Use Grok API
    'temperature': 0.7,           # Higher = more creative factors  
    'max_tokens': 1500,          # Allow longer explanations
}

GENERATION_CONFIG = {
    'num_factors': 15,            # Generate more factors for selection
    'factor_complexity': 'medium', # Balance complexity vs interpretability
}
```

## 📊 Sample Factor Output

```json
{
  "factor_expression": "((df['rsi'] - 50) / 50) * (df['volume'] / df['volume'].rolling(20).mean() - 1)",
  "explanation": "RSI momentum combined with volume surge indicates institutional accumulation during oversold conditions, exploiting retail panic selling",
  "expected_signal": "bullish", 
  "confidence": 0.75,
  "category": "momentum"
}
```

## 🚨 Cost Management

### Grok API Pricing (Estimate)
- **Input**: ~$0.01 per factor generation (500-1000 tokens)
- **Daily Budget**: $5-10 for 20-50 factors
- **Monthly**: $150-300 for serious factor research

### Cost Optimization Tips:
```python
# In config.py - Reduce costs while testing
LLM_CONFIG = {
    'temperature': 0.5,     # Lower temperature = more focused
    'max_tokens': 800,     # Shorter responses  
}

GENERATION_CONFIG = {
    'num_factors': 10,     # Start with fewer factors
    'optimization_iterations': 2,  # Reduce optimization rounds
}
```

## 🔍 Debugging & Troubleshooting

### Common Issues:

**"Grok API key not provided"**
```bash
# Check environment variable
echo $GROK_API_KEY

# Or set manually
export GROK_API_KEY="your_key_here"
```

**"Factor syntax validation failed"**  
- Check available data columns in prompt
- Verify pandas DataFrame syntax: `df['column']`
- Review factor complexity settings

**"Low factor success rate"**
- Increase `temperature` for more creativity
- Adjust `max_tokens` for longer responses
- Review market context in prompts

### Debug Mode:
```python
# In config.py
LOGGING_CONFIG = {
    'level': 'DEBUG',  # See detailed LLM interactions
}
```

## 🎯 Success Metrics & Next Steps

### Week 1 Targets:
- [ ] ✅ Grok API integration working (>3 factors generated)
- [ ] 📊 Backtest validation (average Sharpe >0.3, best >0.6) 
- [ ] 🔄 Factor optimization loop working
- [ ] 📈 Out-of-sample testing on 2023-2024 data

### Week 2 Production:
- [ ] 🏭 IBKR paper trading integration
- [ ] 📺 TradingView Pine Script export
- [ ] 📊 Performance monitoring dashboard  
- [ ] 📚 Factor library and cataloging

## 💡 Advanced Usage

### Custom Factor Categories:
```python
# In factor_generation.py - Add your own categories
custom_categories = [
    "Options flow sentiment",
    "Earnings surprise momentum",  
    "Insider trading patterns",
    "Social media sentiment drift"
]
```

### Multi-Asset Support:
```python  
# In config.py - Test on different asset classes
DATA_CONFIG = {
    'tickers': ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT'],  # ETFs
    # Or: ['EURUSD=X', 'GBPUSD=X', '^TNX']        # FX & Bonds
}
```

## 🎉 You're Ready for Real Alpha Discovery!

The MVP is now equipped with genuine AI-powered factor generation. The next run should produce factors with real economic insight and market-beating potential.

**Time to generate your first AI-discovered alpha factors! 🚀**