#!/usr/bin/env python3
"""
Twitter API Setup Helper

This script helps you configure the Twitter Bearer Token for sentiment analysis
in the AI Trading Advancements system.

Author: AI Trading Development Team
Date: August 31, 2025
Version: 1.0.0
"""

import os
import sys
from pathlib import Path
from typing import Optional

def print_separator():
    """Print a visual separator."""
    print("=" * 70)

def print_step(step_num: int, title: str):
    """Print a formatted step header."""
    print(f"\n[STEP {step_num}] {title}")
    print("-" * 50)

def test_twitter_connection(bearer_token: str) -> bool:
    """
    Test the Twitter connection with the provided bearer token.
    
    Args:
        bearer_token: The Twitter Bearer Token to test
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        import tweepy
        
        client = tweepy.Client(bearer_token=bearer_token)
        
        # Test with a simple search (limited results to avoid rate limits)
        tweets = client.search_recent_tweets("trading", max_results=5)
        
        if tweets and tweets.data:
            print(f"[SUCCESS] Found {len(tweets.data)} test tweets")
            return True
        else:
            print("[WARNING] No tweets found in test search")
            return True  # Connection is still valid
            
    except tweepy.Unauthorized:
        print("[ERROR] Unauthorized - Bearer Token is invalid")
        return False
    except tweepy.Forbidden:
        print("[ERROR] Forbidden - Check your app permissions")
        return False
    except tweepy.TooManyRequests:
        print("[WARNING] Rate limit exceeded - but token is valid")
        return True
    except Exception as e:
        print(f"[ERROR] Connection test failed: {e}")
        return False

def get_env_file_path() -> Path:
    """Get the path to the .env file."""
    # Check current directory first
    current_dir = Path.cwd()
    env_file = current_dir / ".env"
    
    if env_file.exists():
        return env_file
    
    # Check parent directory (TradeAppComponents_fresh)
    parent_dir = current_dir.parent
    env_file = parent_dir / ".env"
    
    if env_file.exists():
        return env_file
    
    # Default to current directory
    return current_dir / ".env"

def update_env_file(bearer_token: str) -> bool:
    """
    Update the .env file with the Twitter Bearer Token.
    
    Args:
        bearer_token: The Twitter Bearer Token to add
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        env_file = get_env_file_path()
        
        # Read existing content
        lines = []
        if env_file.exists():
            with open(env_file, 'r') as f:
                lines = f.readlines()
        
        # Check if TWITTER_BEARER_TOKEN already exists
        token_line_found = False
        for i, line in enumerate(lines):
            if line.strip().startswith('TWITTER_BEARER_TOKEN='):
                lines[i] = f'TWITTER_BEARER_TOKEN={bearer_token}\n'
                token_line_found = True
                break
        
        # If not found, add it
        if not token_line_found:
            lines.append(f'\n# Twitter API Configuration\n')
            lines.append(f'TWITTER_BEARER_TOKEN={bearer_token}\n')
        
        # Write back to file
        with open(env_file, 'w') as f:
            f.writelines(lines)
        
        print(f"[SUCCESS] Updated {env_file}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to update .env file: {e}")
        return False

def main():
    """Main setup function."""
    print_separator()
    print("       TWITTER API SETUP FOR AI TRADING ADVANCEMENTS")
    print_separator()
    
    print("\nThis script will help you configure Twitter API access for sentiment analysis.")
    print("You'll need a Twitter Developer Account and Bearer Token.")
    
    # Step 1: Check if tweepy is installed
    print_step(1, "Checking Dependencies")
    
    try:
        import tweepy
        print(f"[SUCCESS] Tweepy is installed (version: {tweepy.__version__})")
    except ImportError:
        print("[ERROR] Tweepy is not installed")
        print("[DATA] Install with: pip install tweepy")
        sys.exit(1)
    
    # Step 2: Get Bearer Token
    print_step(2, "Twitter Developer Account Setup")
    
    print("\nTo get a Twitter Bearer Token:")
    print("1. Go to https://developer.twitter.com/")
    print("2. Create a developer account (if you haven't already)")
    print("3. Create a new project and app")
    print("4. Navigate to 'Keys and Tokens' section")
    print("5. Generate your Bearer Token")
    print("\nNOTE: Keep your Bearer Token secure and never share it publicly!")
    
    # Step 3: Input Bearer Token
    print_step(3, "Bearer Token Configuration")
    
    # Check if token already exists
    existing_token = os.getenv('TWITTER_BEARER_TOKEN')
    if existing_token:
        print(f"[DATA] Found existing token: {existing_token[:20]}...")
        
        while True:
            use_existing = input("\nUse existing token? (y/n): ").lower().strip()
            if use_existing in ['y', 'yes']:
                bearer_token = existing_token
                break
            elif use_existing in ['n', 'no']:
                bearer_token = input("\nEnter your Twitter Bearer Token: ").strip()
                break
            else:
                print("Please enter 'y' or 'n'")
    else:
        bearer_token = input("\nEnter your Twitter Bearer Token: ").strip()
    
    if not bearer_token:
        print("[ERROR] Bearer Token cannot be empty")
        sys.exit(1)
    
    # Step 4: Test Connection
    print_step(4, "Testing Connection")
    
    print("[PROCESSING] Testing Twitter API connection...")
    
    if test_twitter_connection(bearer_token):
        print("[SUCCESS] Twitter API connection successful!")
    else:
        print("[ERROR] Twitter API connection failed")
        print("[DATA] Please check your Bearer Token and try again")
        sys.exit(1)
    
    # Step 5: Save Configuration
    print_step(5, "Saving Configuration")
    
    if update_env_file(bearer_token):
        print("[SUCCESS] Configuration saved to .env file")
    else:
        print("[ERROR] Failed to save configuration")
        print("[DATA] You can manually add this line to your .env file:")
        print(f"TWITTER_BEARER_TOKEN={bearer_token}")
    
    # Step 6: Verify Setup
    print_step(6, "Final Verification")
    
    # Reload environment
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    # Test with AI trading system
    try:
        # Add current directory to path for imports
        sys.path.insert(0, str(Path.cwd()))
        
        import ai_advancements
        
        print("[PROCESSING] Testing AI trading system integration...")
        
        # Get configuration
        core = ai_advancements.get_core_components()
        config = core['get_config']()
        
        if config.api_keys.twitter_bearer_token:
            print("[SUCCESS] Twitter Bearer Token loaded in AI system")
            print(f"[DATA] Token: {config.api_keys.twitter_bearer_token[:20]}...")
        else:
            print("[WARNING] Twitter Bearer Token not found in AI system")
            print("[DATA] You may need to restart your application")
        
        # Check sentiment feature status
        if config.features.get('sentiment_analysis', False):
            print("[SUCCESS] Sentiment analysis feature is enabled")
        else:
            print("[WARNING] Sentiment analysis feature is disabled")
            print("[DATA] You can enable it by setting ENABLE_SENTIMENT=true in .env")
        
    except Exception as e:
        print(f"[WARNING] Could not test AI system integration: {e}")
        print("[DATA] This is normal if the system is not fully set up")
    
    # Final instructions
    print_separator()
    print("                        SETUP COMPLETE!")
    print_separator()
    
    print("\n[SUCCESS] Twitter API is now configured for AI Trading Advancements")
    print("\nNext steps:")
    print("1. Enable sentiment analysis: Set ENABLE_SENTIMENT=true in your .env file")
    print("2. Restart your application to load the new configuration")
    print("3. Test sentiment analysis in your trading strategies")
    print("\nFor advanced configuration, see the documentation in docs/")
    
    print("\n[DATA] Security reminder:")
    print("- Never commit your .env file to version control")
    print("- Keep your Bearer Token secure")
    print("- Monitor your API usage to avoid rate limits")
    
    print(f"\n[SUCCESS] Setup completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[DATA] Setup cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        sys.exit(1)
