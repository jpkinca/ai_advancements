"""initialize.py
Centralized initialization for TensorTrade MVP project.
Handles Python path setup and environment variable loading.
"""
import os
import sys
import pathlib
from dotenv import load_dotenv

# Get project root directory (parent of src)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Add project root and src directory to Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Set DATABASE_URL if not already set
if not os.getenv("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway"

print(f"✅ Initialized TensorTrade environment")
print(f"   Project root: {PROJECT_ROOT}")
print(f"   Database URL: {os.getenv('DATABASE_URL')[:50]}...")
