#!/usr/bin/env python3
"""
Authentication test script for Chain-of-Alpha MVP

Tests Hugging Face authentication and model access.
"""

import os
import sys
from dotenv import load_dotenv

def test_huggingface_auth():
    """Test Hugging Face authentication"""
    
    print("Testing Hugging Face Authentication")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check for token
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    
    if not hf_token:
        print("❌ HUGGINGFACE_TOKEN not found in environment variables")
        print("\nTo fix this:")
        print("1. Get a token from: https://huggingface.co/settings/tokens")
        print("2. Set environment variable: $env:HUGGINGFACE_TOKEN='hf_your_token_here'")
        print("3. Or create a .env file with: HUGGINGFACE_TOKEN=hf_your_token_here")
        return False
    
    print(f"✅ Found Hugging Face token: {hf_token[:10]}...")
    
    # Test API access
    try:
        from transformers import AutoTokenizer
        
        print("Testing model access...")
        
        # Test with a simple model first
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            token=hf_token
        )
        
        print("✅ Successfully accessed Llama model!")
        print("✅ Authentication working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        
        if "401" in str(e) or "authentication" in str(e).lower():
            print("\nPossible fixes:")
            print("- Check your token is correct")
            print("- Request access at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
        elif "gated" in str(e).lower():
            print("\nYou need to request access to the Llama model:")
            print("1. Go to: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
            print("2. Click 'Request access to this model'")
            print("3. Accept the license agreement")
            
        return False

def test_mock_mode():
    """Test mock mode as fallback"""
    
    print("\nTesting Mock Mode")
    print("=" * 20)
    
    try:
        sys.path.append(os.path.dirname(__file__))
        from src.llm_interface import LLMInterface
        
        config = {'llm_model': 'mock'}
        llm = LLMInterface(config)
        
        response = llm.generate_response("Test prompt")
        print(f"✅ Mock mode working: {response[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Mock mode failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("Chain-of-Alpha Authentication Test")
    print("=" * 50)
    
    # Test HF auth
    hf_success = test_huggingface_auth()
    
    # Test mock mode
    mock_success = test_mock_mode()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    
    if hf_success:
        print("✅ Hugging Face: Ready to use Llama models")
        print("   You can set llm_model='llama-3-8b' in config.py")
    else:
        print("❌ Hugging Face: Authentication needed")
        print("   See HUGGINGFACE_SETUP.md for instructions")
    
    if mock_success:
        print("✅ Mock Mode: Available as fallback")
        print("   Use llm_model='mock' in config.py")
    else:
        print("❌ Mock Mode: Issues detected")
    
    print("\nRecommendation:")
    if hf_success:
        print("- Use Llama models for best results")
    elif mock_success:
        print("- Use mock mode for testing and development")
        print("- Set up HF authentication later for production")
    else:
        print("- Check your installation: pip install -r requirements.txt")
    
    return hf_success or mock_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)