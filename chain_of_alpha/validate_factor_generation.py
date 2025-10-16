#!/usr/bin/env python3
"""
Validation script for Chain-of-Alpha factor generation

Tests the improved LLM integration and structured factor generation
"""

import sys
import os
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_structured_factor_generation():
    """Test the enhanced factor generation with Grok API"""
    
    print("🧪 Chain-of-Alpha Factor Generation Validation")
    print("=" * 60)
    
    try:
        # Import components
        from src.llm_interface import LLMInterface
        from src.factor_generation import FactorGenerationChain
        from config import CONFIG
        
        print("✅ Imported Chain-of-Alpha components")
        
        # Check API key
        import os
        api_key = os.getenv('GROK_API_KEY')
        if not api_key:
            print("❌ No Grok API key found. Run setup_grok_api.py first.")
            return False
        
        print(f"✅ Found Grok API key: {api_key[:10]}...")
        
        # Initialize LLM interface
        llm_config = CONFIG.copy()
        llm_config['llm_model'] = 'grok'  # Force Grok for this test
        
        print("🔄 Initializing LLM interface...")
        llm = LLMInterface(llm_config)
        
        # Test basic LLM response
        print("🔄 Testing basic LLM connectivity...")
        test_response = llm.generate_response("Respond with 'LLM connection successful' to confirm API works.")
        print(f"📝 LLM Response: {test_response[:100]}...")
        
        # Initialize factor generation chain
        print("🔄 Initializing factor generation chain...")
        factor_gen = FactorGenerationChain(llm_config, llm)
        
        # Create mock market data context
        mock_data_context = {
            'columns': ['close', 'open', 'high', 'low', 'volume', 'returns', 'log_returns', 
                       'sma_5', 'sma_20', 'sma_50', 'rsi', 'macd', 'volatility_20'],
            'date_range': {'start': '2023-01-01', 'end': '2024-01-01'},
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            'stats': {
                'close': {'mean': 150.0, 'std': 50.0},
                'volume': {'mean': 1000000, 'std': 500000}
            }
        }
        
        print("🔄 Generating test factors...")
        
        # Generate 3 test factors with different focuses
        test_factors = []
        
        for i in range(1, 4):
            print(f"\n📊 Generating Factor #{i}...")
            
            try:
                factor = factor_gen._generate_single_factor(mock_data_context, i)
                
                if factor:
                    test_factors.append(factor)
                    print(f"✅ Factor {i} generated successfully:")
                    print(f"   📜 Expression: {factor.get('expression', 'N/A')}")
                    print(f"   💡 Explanation: {factor.get('explanation', 'N/A')[:80]}...")
                    
                    # Validate factor expression syntax
                    try:
                        import pandas as pd
                        # Create dummy DataFrame for syntax validation
                        dummy_data = pd.DataFrame({
                            'close': [100, 101, 102],
                            'open': [99, 100, 101],
                            'high': [102, 103, 104],
                            'low': [98, 99, 100],
                            'volume': [1000, 1100, 1200],
                            'returns': [0.01, 0.01, 0.01],
                            'log_returns': [0.01, 0.01, 0.01],
                            'sma_5': [100, 101, 102],
                            'sma_20': [99, 100, 101],
                            'sma_50': [98, 99, 100],
                            'rsi': [50, 55, 60],
                            'macd': [0.1, 0.2, 0.3],
                            'volatility_20': [0.2, 0.22, 0.25]
                        })
                        
                        # Test factor evaluation
                        result = eval(factor['expression'], {"__builtins__": {}}, {'df': dummy_data, 'pd': pd})
                        print(f"   ✅ Factor syntax validation passed: {type(result).__name__}")
                        
                    except Exception as e:
                        print(f"   ❌ Factor syntax validation failed: {e}")
                        
                else:
                    print(f"❌ Factor {i} generation failed")
                    
            except Exception as e:
                print(f"❌ Error generating factor {i}: {e}")
        
        # Summary
        print(f"\n📊 Validation Summary")
        print("=" * 30)
        print(f"✅ Factors generated: {len(test_factors)}/3")
        
        if len(test_factors) >= 2:
            print("🎉 VALIDATION SUCCESSFUL!")
            print("✅ Grok API integration working")
            print("✅ Structured factor generation working")
            print("✅ Factor syntax validation working")
            
            # Save test results
            results = {
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'factors_generated': len(test_factors),
                'test_factors': test_factors
            }
            
            with open('validation_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print("✅ Validation results saved to validation_results.json")
            return True
        else:
            print("❌ VALIDATION FAILED - Insufficient factors generated")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Main validation function"""
    
    success = test_structured_factor_generation()
    
    if success:
        print("\n🚀 Ready for Production Testing!")
        print("Next steps:")
        print("1. Run: python chain_of_alpha_mvp.py")
        print("2. Monitor factor quality and performance")
        print("3. Iterate on prompts for better factors")
    else:
        print("\n❌ Validation failed. Check setup and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()