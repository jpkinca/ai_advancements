#!/usr/bin/env python3
"""
Setup script for Grok API integration with Chain-of-Alpha MVP

Run this script to configure your Grok API key and test the connection.
"""

import os
import sys
from dotenv import load_dotenv

def setup_grok_api():
    """Interactive setup for Grok API"""
    
    print("🚀 Chain-of-Alpha MVP - Grok API Setup")
    print("=" * 50)
    
    # Load existing .env file
    load_dotenv()
    
    # Check if API key already exists
    existing_key = os.getenv('GROK_API_KEY')
    if existing_key:
        print(f"✅ Found existing Grok API key: {existing_key[:10]}...")
        if input("Use existing key? (y/n): ").lower().startswith('y'):
            return test_grok_connection(existing_key)
    
    # Get API key from user
    print("\n📋 Grok API Key Setup")
    print("1. Go to: https://console.x.ai/")
    print("2. Sign in with your X account")
    print("3. Navigate to API Keys section")
    print("4. Create a new API key")
    print("5. Copy the key below")
    
    api_key = input("\n🔑 Enter your Grok API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return False
    
    # Save to .env file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    
    try:
        with open(env_path, 'a') as f:
            f.write(f"\n# Grok API Configuration\n")
            f.write(f"GROK_API_KEY={api_key}\n")
        
        print(f"✅ API key saved to {env_path}")
        
        # Set environment variable for current session
        os.environ['GROK_API_KEY'] = api_key
        
        # Test connection
        return test_grok_connection(api_key)
        
    except Exception as e:
        print(f"❌ Error saving API key: {e}")
        return False

def test_grok_connection(api_key: str) -> bool:
    """Test Grok API connection"""
    
    print("\n🧪 Testing Grok API Connection...")
    
    try:
        import requests
        
        url = "https://api.x.ai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        test_payload = {
            "model": "grok-3",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user", 
                    "content": "Say 'Hello from Grok API!' to confirm the connection works."
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        print("🔄 Sending test request...")
        response = requests.post(url, headers=headers, json=test_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"✅ Connection successful!")
            print(f"📝 Grok response: {message}")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Main setup function"""
    
    try:
        # Check if requests is available
        import requests
    except ImportError:
        print("❌ Missing required package. Installing requests...")
        os.system("pip install requests python-dotenv")
    
    success = setup_grok_api()
    
    if success:
        print("\n🎉 Setup Complete!")
        print("✅ Grok API is configured and working")
        print("✅ You can now run the Chain-of-Alpha MVP with real LLM")
        print("\n🚀 Next steps:")
        print("1. Run: python chain_of_alpha_mvp.py")
        print("2. Check results in the results/ directory")
        print("3. Monitor factor performance and iterate")
        
        # Update config to use Grok
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.py')
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            # Update LLM model to grok if it's still mock
            if "'llm_model': 'mock'" in config_content:
                config_content = config_content.replace("'llm_model': 'mock'", "'llm_model': 'grok'")
                with open(config_path, 'w') as f:
                    f.write(config_content)
                print("✅ Updated config.py to use Grok API")
                
        except Exception as e:
            print(f"⚠️ Could not auto-update config.py: {e}")
            print("💡 Manual step: Change 'llm_model' from 'mock' to 'grok' in config.py")
    else:
        print("\n❌ Setup failed. Please check your API key and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()